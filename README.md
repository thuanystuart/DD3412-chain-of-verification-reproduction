# DD3412-chain-of-verification-reproduction

## :fire: Quickstart

First, create a python virtual environment and install the requirements:
```bash
conda create -n CoVe python=3.10
conda activate CoVe
```
To install requirements:
```bash
pip install -r requirements.txt
```
Then, create a file named `.configuration` to store the confidential data, such as OpenAI and HuggingFace API keys. This file should look like:
```
OPENAI_API_KEY=sk-abc.....
HF_API_KEY=hf_abc.......
```
💻 Run the experiments
==========
## 🎯 Running with Bash Script 
To execute CoVe, use the command below:

```bash
>> bash scripts/wikidata.sh # for wikidata task, and Llama2-70b model
```

- Available models are: `llama2`, `llama2_70b`, `llama-65b`, `gpt3`
- Available settings are: `joint`, `two_step`, `factored`
- Available tasks are: `wikidata`, `wikidata_category`, `multispanqa` 

## 📦 Running with Apptainer
You can run the experiments using Apptainer with the following steps:

1. **Build the Apptainer container** by utilizing the provided `my_apptainer.def` file:
```bash
# Build the Apptainer container
apptainer build my_container.sif my_apptainer.def
```
2. **Run the CoVe experiments within the Apptainer container** by executing the `apptainer_job`:
```bash
SBATCH apptainer_job.sh
```
Alternatively, you can use the following command:
```bash
apptainer exec --nv my_apptainer.sif ./scripts/wikidata.sh # for Wikidata task and Llama2-70b
```
