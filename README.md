# hardPrompt2softPrompt
do PPO on gemma-it based lm with prefix tuning

```bash
conda create --name hard2softPPO python=3.10
conda activate hard2softPPO

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51.3
```

```python
from hardPrompt2softPrompt.models.policyModel.modeling_policyModel import PrefixTuningPolicyModel

model = PrefixTuningPolicyModel(
    model_name='google/gemma-3-1b-it', 
    pretrain_path='./outputs/2505142356/2505142356_best_model.pth', 
)

messages_ids = model.chat_template_tokenizer(
    first_context='你覺得 YouTube 頻道中，最吸引你的類型是哪一種呢？'
)

response, _ = model.generate_response(
    messages_ids=messages_ids, 
    max_new_tokens=100, 
    use_prefix=True,
    temperature=1.0,
    prefix_checker=False
)
print(response)
```














