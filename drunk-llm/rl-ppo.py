import os, wandb
import torch
import joblib
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from sentence_transformers import SentenceTransformer
from datasets import Dataset

# TODO: change as per model etc
os.environ["WANDB_RUN_NAME"] = "ppo-drunk-llama"  # custom run name
EPOCHS = 1
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

LORA_ADAPTER = f"/data/epochs-{EPOCHS}-lora-llama-2-7b-chat-hf-adapter"
print("LORA: ", LORA_ADAPTER)
model_name = MODEL_NAME
lora_adapter_path = LORA_ADAPTER
log_reg_path = "logreg_sbert_model.pkl"  # TODO: drunk text clf
sbert_name = "all-MiniLM-L6-v2"



# 1. REWARD MODEL (SBERT + LogReg)
# Bridges your Scikit-Learn classifier to the PPO loop
class SBERTRewardModel:
    def __init__(self, sbert_path, log_reg_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sbert = SentenceTransformer(sbert_path).to(self.device)
        self.classifier = joblib.load(log_reg_path)

    def get_rewards(self, texts):
        embeddings = self.sbert.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        scores = self.classifier.predict_proba(embeddings)[:, 1]
        return [torch.tensor(score, dtype=torch.float32) for score in scores]



torch_dtype = torch.bfloat16
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    device_map="auto"
)

model.pretrained_model = PeftModel.from_pretrained(
    model.pretrained_model,
    lora_adapter_path,
    is_trainable=True
)

reward_wrapper = SBERTRewardModel(sbert_name, log_reg_path)

prompts = list(pd.read_csv("../dataset/synthetic-prompts-heldout-drunk-texts-data.csv")['prompt'])

dataset = Dataset.from_dict({"query": prompts})
dataset = dataset.map(
    lambda x: tokenizer(x["query"], truncation=True, max_length=256),
    batched=False,
    remove_columns=["query"]
)
dataset.set_format(type="torch")


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


config = PPOConfig(
    learning_rate=5e-6,  # Lower LR for stability
    batch_size=128,  # 128 prompts per "step"
    mini_batch_size=16,  # Process 16 at a time on the A100
    ppo_epochs=4,  # Repeat optimization 4 times per batch
    optimize_cuda_cache=True,
    # init_kl_coeff=0.2,         # Start with a stricter KL penalty
    target_kl=0.1,  # The "Goldilocks" zone for KL
    log_with="wandb",
)

ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=None,  # TRL handles reference model for PEFT automatically
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
)

generation_kwargs = {
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 1.1,  # Added for a bit more "drunk" variety
    "repetition_penalty": 1.1,  # Added to prevent loops
    "max_new_tokens": 128,
    "pad_token_id": tokenizer.pad_token_id,
}

merged_config = config.__dict__ | generation_kwargs
wandb.config.update(merged_config)

print(f"Starting PPO with {len(dataset)} prompts...")
total_steps = len(dataset) // config.batch_size
pbar = tqdm(enumerate(ppo_trainer.dataloader), total=total_steps)
for epoch, batch in pbar:
    query_tensors = [q for q in batch["input_ids"]]

    # Step A: Generate
    response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)

    # Step B: Score
    batch_texts = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
    rewards = reward_wrapper.get_rewards(batch_texts)
    batch["query"] = [tokenizer.decode(q, skip_special_tokens=True) for q in query_tensors]
    batch["response"] = batch_texts

    rewards = [r.to(ppo_trainer.accelerator.device) for r in rewards]

    # Step C: Step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    stats["env/reward_drunk_only"] = torch.mean(torch.tensor([r for r in rewards]))  # Simplified
    stats["env/response_length"] = torch.mean(torch.tensor([float(len(t.split())) for t in batch_texts]))
    ppo_trainer.log_stats(stats, batch, rewards)

    if epoch % 1 == 0:
        mean_reward = torch.mean(torch.stack(rewards)).item()
        # Get KL divergence (how much the model changed)
        kl_div = stats['ppo/policy/approxkl'].item()

        print(f"\n[Step {epoch}]")
        print(f"Mean Reward: {mean_reward:.4f}")
        print(f"KL Div: {kl_div:.4f}")
        print(f"Sample Prompt: {batch_texts[0][:50]}...")
        print(f"Sample Response: {batch_texts[0][len(batch['query'][0]):]}")  # Just the answer
        print("-" * 30)

# 7. FINAL SAVE
ppo_trainer.save_pretrained("./filtered-ppo_drunk_llama2")