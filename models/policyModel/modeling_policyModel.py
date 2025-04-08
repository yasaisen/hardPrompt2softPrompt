"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2504021929
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

from ...common.utils import log_print, get_trainable_params, highlight, highlight_show


class PrefixTuningPolicyModel(nn.Module):
    def __init__(self, 
        model_name: str, 
        prefix_prompt: str=None, 
        pretrain_path=None, 
        gradient_checkpointing: bool=False,
        device: str="cuda", 
        torch_dtype=torch.float32
    ):
        super().__init__()
        self.state_name = 'PrefixTuningPolicyModel'
        self.device = device
        print()
        log_print(self.state_name, f"Building...")

        # 1) Load pretrained GPT
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            # attn_implementation="flash_attention_2", 
            torch_dtype=torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            log_print(self.state_name, f"pad_token=None")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model.config.gradient_checkpointing = gradient_checkpointing

        # 3) Prefix-tuning parameters
        if prefix_prompt is not None:
            log_print(self.state_name, f"prefix_prompt={prefix_prompt}")
            self.prefix_prompt = prefix_prompt
            self.prefix_ids = self.tokenizer.encode(self.prefix_prompt, add_special_tokens=False, return_tensors="pt").to(torch.long)
            self.prefix_length = int(self.prefix_ids.shape[1])
            self.max_length = self.base_model.config.max_position_embeddings - self.prefix_length
            self.hidden_size = self.base_model.config.hidden_size
            """
            4-(4+prefix_length)
            """
            word_embeds = self.base_model.model.embed_tokens(self.prefix_ids)
            self.prefix_embeddings = nn.Parameter(word_embeds.detach().clone().squeeze(), requires_grad=True)
            log_print(self.state_name, f"prefix_shape={self.prefix_embeddings.shape}")

        if pretrain_path is not None:
            log_print(self.state_name, f"pretrain_path={pretrain_path}")
            ckpt = torch.load(pretrain_path)
            ### TODO: model save
            self.prefix_embeddings = nn.Parameter(ckpt['prefix_embeddings_state_dict'])
            log_print(self.state_name, f"prefix_shape={self.prefix_embeddings.shape}")
        
        # self.prefix_embeddings = nn.Parameter(torch.randn(prefix_length, self.hidden_size, dtype=torch_dtype, device=self.device))
        # nn.init.normal_(self.prefix_embeddings, mean=0.0, std=0.5)

        log_print(self.state_name, f"max_length={self.max_length}")

        # 2) Freeze all GPT parameters
        for param in self.parameters():
            param.requires_grad = False
        self.prefix_embeddings.requires_grad = True

        log_print(self.state_name, f"requires_grad={sum(p.numel() for p in self.parameters() if p.requires_grad)}")
        log_print(self.state_name, f"prefix_embeddings={self.prefix_embeddings.numel() if self.prefix_embeddings.requires_grad else 0}")
        log_print(self.state_name, f"prefix_embeddings={self.prefix_embeddings.shape}")
        log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self)}")
        self.to(self.device) # 
        log_print(self.state_name, f"...Done\n")

    def generate_response(self, 
        messages_ids: torch.Tensor, 
        max_new_tokens: int = 50, 
        use_prefix: bool = True,
        temperature: float = 1.0
    ):
        self.eval()
        generated_tokens = []
        sum_logprob = 0.0
        probs_list = []
        
        for _ in range(max_new_tokens):
            input_ids = torch.cat([messages_ids, torch.tensor([generated_tokens], dtype=torch.long, device=self.device)], dim=1)
            with torch.no_grad():
                logits = self(
                    input_ids=input_ids, 
                    use_prefix=use_prefix,
                    stage='decode',
                )

            # greedy search
            next_token_logits = logits[0, -1, :] # torch.Size([1, 256000])
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            log_prob = torch.log(probs[next_token_id] + 1e-10)
            
            generated_tokens.append(next_token_id)
            sum_logprob += log_prob.item()
            probs_list.append(probs.unsqueeze(0))
            
            if next_token_id == self.tokenizer.eos_token_id:
                break

        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response, sum_logprob, probs_list

    def full_forward(self, 
        messages_ids: torch.Tensor,
        response_ids: torch.Tensor,
        use_prefix: bool,
        temperature: float = 1.0
    ):
        ### get full response logits from forward pass response to policy model ###
        combined_ids = torch.cat([messages_ids[:, :-3], response_ids[:, 7:-3]], dim=1)
        # highlight_show('[full_forward] input_ids(decoded)', self.tokenizer.decode(combined_ids.tolist()[0], skip_special_tokens=False))

        logits = self(
            input_ids=combined_ids, 
            use_prefix=use_prefix,
            stage='prefill',
        )
        response_logits = logits[:, -response_ids.shape[1]:] # [1, seq_len + 1, 256000]

        probs = F.softmax(response_logits / temperature, dim=-1)
        log_probs = F.log_softmax(response_logits / temperature, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        ### calculate new_log_prob ###
        token_log_probs = torch.gather( # [1, seq_len]
            log_probs,
            -1,
            response_ids.unsqueeze(-1)
        ).squeeze(-1)
        new_log_prob = token_log_probs.sum()

        return response_logits, new_log_prob, entropy

    def forward(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        use_prefix: bool = True,
        stage: str = '',
    ):
        # highlight_show('[forward] input_ids(decoded)', self.tokenizer.decode(input_ids.tolist()[0], skip_special_tokens=False))

        # template_start = torch.tensor([[     2,    106,   1645,    108]], dtype=torch.long).to(self.device) # user
        template_start = torch.tensor([[     2,    106,   9020,    108]], dtype=torch.long).to(self.device) # system
        template_end = torch.tensor([[   107,    108]], dtype=torch.long).to(self.device)
        template_start_len = int(template_start.shape[1])
        template_end_len = int(template_end.shape[1])

        # kill first [     2,    106,   (1645),    108,   (3940),    107,    108] tokens
        check_tensor = torch.tensor([[     2,    106,   1645,    108,   3940,    107,    108]], dtype=torch.long).to(self.device)
        if not torch.equal(input_ids[:, :7], check_tensor):
            raise ValueError(f"Input input_ids with not prefix [\n{self.tokenizer.decode(check_tensor.tolist()[0], skip_special_tokens=False)}\n] but [\n{self.tokenizer.decode(input_ids[:, :7].tolist()[0], skip_special_tokens=False)}\n]")
        
        cutted_input_ids = input_ids[:, 7:]
        batch_size, seq_len = cutted_input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, 
                template_start_len + self.prefix_length + template_end_len + seq_len, 
                dtype=torch.long, 
                device=cutted_input_ids.device
            )

        cutted_input_ids = cutted_input_ids.to(self.device)
        self.prefix_ids = self.prefix_ids.to(self.device)
        self.prefix_embeddings = self.prefix_embeddings.to(self.device)

        formaled_input_ids = torch.cat([template_start, self.prefix_ids, template_end], dim=1)
        # highlight_show('input_ids(decoded)', self.tokenizer.decode(torch.cat([formaled_input_ids, cutted_input_ids], dim=1).tolist()[0], skip_special_tokens=False))
        formaled_inputs_embeds = self.base_model.model.embed_tokens(formaled_input_ids)
        
        if use_prefix:
            formaled_inputs_embeds = torch.cat([
                formaled_inputs_embeds[:, :template_start_len], 
                self.prefix_embeddings.unsqueeze(0), 
                formaled_inputs_embeds[:, -template_end_len:]
            ], dim=1)

        inputs_embeds = torch.cat([
            formaled_inputs_embeds.expand(batch_size, -1, -1), 
            self.base_model.model.embed_tokens(cutted_input_ids)
        ], dim=1).to(cutted_input_ids.device)

        # log_print(self.state_name, f"[{highlight()}] [{stage}] {use_prefix} / {inputs_embeds.shape[1]}")
        transformer_outputs = self.base_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.base_model.lm_head(hidden_states)
        
        return logits

    def chat_template_tokenizer(self, 
            chat_dict, 
            is_response:bool = False,
        ):

        if is_response:
            chat_dict = [{'role': 'assistant', 'content': chat_dict}]
        
        chat_dict = [{'role': 'user', 'content': 'temp'}] + chat_dict
        prompt = self.tokenizer.apply_chat_template(chat_dict, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(torch.long).to(self.device)

        if input_ids.shape[1] > self.max_length:
            raise ValueError(f"Input length {input_ids.shape[1]} exceeds max_length {self.max_length}")

        return input_ids
    
    @classmethod
    def from_config(cls, cfg):
        root_path = cfg['task'].get("root_path")
        device = str(cfg['task'].get("device"))

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if cfg['model'].get('policy_model') is not None:
            policy_model_cfg = cfg['model']['policy_model']
            if policy_model_cfg.get("policy_model_path") == "":
                policy_model_path = None
            else:
                policy_model_path = os.path.join(root_path, policy_model_cfg.get("policy_model_path"))
            policy_model_name = str(policy_model_cfg.get("policy_model_name"))
            prefix_prompt = str(policy_model_cfg.get("prefix_prompt"))
            torch_dtype = dtype_map[str(policy_model_cfg.get("torch_dtype"))]

        model = cls(
            model_name=policy_model_name,
            prefix_prompt=prefix_prompt, 
            pretrain_path=policy_model_path, 
            gradient_checkpointing=False,
            torch_dtype=torch_dtype,
            device=device, 
        )
        return model
    











    