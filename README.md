# hardPrompt2softPrompt
do PPO on gemma-it based lm with prefix tuning

## environment setup
```bash
conda create --name hard2softPPO python=3.10
conda activate hard2softPPO

git clone https://github.com/yasaisen/hardPrompt2softPrompt.git
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51.3 matplotlib

huggingface-cli login
```

## how to use
```python
from hardPrompt2softPrompt.models.policyModel.modeling_policyModel import PrefixTuningPolicyModel

model = PrefixTuningPolicyModel.from_pretrained(
    model_name='google/gemma-3-1b-it', 
)

messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": '你覺得 YouTube 頻道中，最吸引你的類型是哪一種呢？'},]
        }
]

messages_ids = model.chat_template_tokenizer(
    messages=messages
)

response = model.generate_response(
    messages_ids=messages_ids, 
    temperature=0.7,
)
print(response)
```
