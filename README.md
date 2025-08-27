# hardPrompt2softPrompt

HardPrompt2SoftPrompt is a research prototype that converts a fixed **hard** prompt into a learnable **soft** prefix using reinforcement learning.  The project implements single step Proximal Policy Optimization (PPO) with prefix tuning on [Gemma](https://huggingface.co/google/gemma-3-1b-it) language models and a comparative reward model built on BERT.  All components are trained through configuration files and can be reused independently.

## Table of Contents

1. [Project Features](#project-features)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Inference example](#inference-example)
5. [Training](#training)
   - [Reward Model](#reward-model-training)
   - [Policy Model with PPO](#policy-model-training)
6. [Configuration](#configuration-files)
7. [Logging and Checkpoints](#logging-and-checkpoints)
8. [License](#license)

## Project Features

* **Prefix tuning policy model** – adds learnable embeddings in front of the base language model so that a hand written prompt can be optimized.
* **Comparative Reward Model** – BERT with a small prefix for learning to score responses by comparing the preferences between two answers.
* **Single‑Step PPO Trainer** – performs reinforcement learning on a policy model to improve soft prompts using rewards from the reward model.

## Repository Structure

```
hardPrompt2softPrompt/
├── RewardModelTraining.py          # entry point for reward model training
├── policyModelTraining.py          # entry point for PPO training
├── demo.ipynb                      # interactive demonstration notebook
├── hardPrompt2softPrompt/
│   ├── common/                     # utilities (config, logging, helpers)
│   ├── datasets/                   # dataset loaders
│   ├── models/
│   │   ├── policyModel/            # PrefixTuningPolicyModel implementation
│   │   ├── rewardModel/            # ComparativeRewardModel implementation
│   │   └── valueHead/              # Value head for PPO
│   └── trainer/                    # trainers for reward model and PPO
└── projects/                       # example YAML configs
```

## Installation

The project targets Python 3.10 and CUDA 12.1.  The commands below create a conda environment and install the required packages.

```bash
conda create --name hard2softPPO python=3.10
conda activate hard2softPPO

git clone https://github.com/yasaisen/hardPrompt2softPrompt.git
cd hardPrompt2softPrompt

# Additional dependencies
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51.3 matplotlib tensorboard

# Authenticate with Hugging Face to access Gemma checkpoints
huggingface-cli login
```

> **Note**: The repository does not include a `setup.py`.  Add the repository
> root to your `PYTHONPATH` or work within this directory when running the
> scripts.

## Inference example

After training you can load the policy model and generate a reply for a conversation.  The `from_pretrained` helper downloads the corresponding prefix embeddings if they are not available locally.

### Load a pre‑trained policy model
```python
from hardPrompt2softPrompt.models.policyModel.modeling_policyModel import PrefixTuningPolicyModel

model = PrefixTuningPolicyModel.from_pretrained(
    model_name='google/gemma-3-1b-it', 
)
```

### Create a chat template
```python
asked_question = "今天天氣如何"

conversation_history = [
    {
        'role': 'user', 
        'content': [{'type': 'text', 'text': '還行'}]
    }, 
    {
        'role': 'assistant', 
        'content': [{'type': 'text', 'text': '還行，但稍微有點悶熱吧？你覺得今天天氣怎麼樣呢？'}]
    }, 
    {
        'role': 'user', 
        'content': [{'type': 'text', 'text': '我覺得還好，不會到很熱'}]
    }, 
]
```

### Generate response
```python
conversation_history = model.generate_response( 
    asked_question=asked_question, 
    conversation_history=conversation_history, 
    temperature=0.7, 
)
response = conversation_history[-1]['content'][0]['text']
print(response)
```

## Training

### Reward model training

1. Prepare an Excel file containing the following columns:
   - `context_messages`: full conversation context as text.
   - `policy_response` and `reference_response`: two candidate replies.
   - `human_eval`: `0` if the policy response is better, `1` if the reference response is better.  Rows labelled `2` or `3` are ignored.
2. Create a YAML configuration.  An example is provided in `hardPrompt2softPrompt/projects/trainRM_2505142236.yaml`.
3. Run the training script:

```bash
python RewardModelTraining.py --cfg-path projects/trainRM_<timestamp>.yaml
```

The script logs metrics to `outputs/<timestamp>` and saves the best model checkpoint in the same directory.

### Policy model training

1. Prepare a JSON dataset in which each item contains the conversation context under the key `context`.  See `singleStepPPO_v1_Dataset` for the expected format.
2. Create a YAML configuration.  An example is provided in `hardPrompt2softPrompt/projects/trainPM_2508252035.yaml`.
3. Launch training:

```bash
python policyModelTraining.py --cfg-path projects/trainPM_<timestamp>.yaml
```

The trainer collects rollouts, updates the prefix embeddings and value head, records TensorBoard metrics and stores the best checkpoint under the configured `output_path`.

## Configuration files

Every training script relies on a YAML file with three top‑level sections:

- `model`: parameters for the policy model and/or reward model.
- `dataset`: paths and split ratios for the datasets.
- `task`: training hyperparameters and runtime options such as `device` and
  `output_path`.

The `ConfigHandler` automatically expands relative paths, creates an output
folder with a 

## Logging and Checkpoints

Every run creates a unique timestamped folder inside the configured
`output_path`.  The folder contains:

* `*_result.log` – human‑readable log of metrics for each training/validation
  step.
* `*.pth` – checkpoint files saved via `ConfigHandler.save_weight`, including
  model state dictionaries and metadata.
* TensorBoard logs when enabled in policy training.

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE)
file for details.

## Acknowledgements

The implementation makes heavy use of PyTorch and the Hugging Face Transformers library.


