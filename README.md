> [!WARNING]
> **Some part of repo contains offensive/hateful speech, profanity, and other potentially triggering content.**

Code and dataset for our paper - _"In Vino Veritas and Vulnerabilities: Examining LLM Safety via Drunk Language Inducement"_ (Accepted INLG 2026)

**arXiv (Pre-print) version:** [link](https://arxiv.org/abs/2601.22169)

<br><br><br>
<p align="center">
<img width="401" height="214" alt="image" src="https://github.com/user-attachments/assets/a6cb3b65-ce00-4149-b6bd-f401f8e22171" />
</p>

The DrunkText dataset could be could found in [/dataset](./dataset) folder. We sourced this from TFLN and Reddit; so we also provide complete datasets for these. 

<br>
<br>

<p align="center">
<img width="839" height="342" alt="image" src="https://github.com/user-attachments/assets/1cc37cd6-3888-4c3e-a1e3-0a796f220e07" />
</p>
Different LLM drunk language inducement methods, drunk text classifier, and adapted LLM-as-Judge jailbreak classifier can be found in [/drunk-llm](./drunk-llm) folder.

<br><br>


<p align="center">
<img width="411" height="240" alt="image" src="https://github.com/user-attachments/assets/3316ecad-a5be-4c99-a05c-3af6f9083c06" />
</p>

For vulnerability evaluation; we re-use:
* Security - JailbreakBench [https://github.com/JailbreakBench/jailbreakbench]
* Privacy - ConfAIde [https://github.com/skywalker023/confaide]



<br><br><br>
### Citing
```
@misc{shetty2026vinoveritasvulnerabilitiesexamining,
      title={In Vino Veritas and Vulnerabilities: Examining LLM Safety via Drunk Language Inducement}, 
      author={Anudeex Shetty and Aditya Joshi and Salil S. Kanhere},
      year={2026},
      eprint={2601.22169},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.22169}, 
}
```
