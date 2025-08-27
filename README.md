# hardPrompt2softPrompt
do PPO on gemma-it based lm with prefix tuning

## environment setup
```bash
conda create --name hard2softPPO python=3.10
conda activate hard2softPPO

git clone https://github.com/yasaisen/hardPrompt2softPrompt.git
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51.3 matplotlib tensorboard

huggingface-cli login
```

## how to use
```python
from hardPrompt2softPrompt.models.policyModel.modeling_policyModel import PrefixTuningPolicyModel

model = PrefixTuningPolicyModel.from_pretrained(
    model_name='google/gemma-3-1b-it', 
)
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
conversation_history = model.generate_response( 
    asked_question=asked_question, 
    conversation_history=conversation_history, 
    temperature=0.7, 
)
response = conversation_history[-1]['content'][0]['text']
print(response)
```
