"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2503252044
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

from ...common.utils import log_print, get_trainable_params


class PrefixTuningPolicyModel(nn.Module):
    """
    Reuses the same GPT for both "policy" forward (with prefix + value head)
    and "reference" forward (without prefix, no value head).
    """
    def __init__(self, 
        model_name: str, 
        prefix_prompt: str=None, 
        pretrain_path=None, 
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

    # ------------------------------------------------------------------
    # Generation: uses in "prefix mode" to do token-by-token sampling.
    # ------------------------------------------------------------------
    def generate_response(
        self, 
        messages_ids, 
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
                logits = self(input_ids, use_prefix=use_prefix)

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

    def full_forward(
        self, 
        messages_ids, 
        response_ids, 
        use_prefix: bool,
        temperature: float = 1.0
    ):
        ### get full response logits from forward pass response to policy model ###

        combined_ids = torch.cat([messages_ids[:, 7:-3], response_ids[:, 7:-3]], dim=1)
        # print('================================================input_ids decode')
        # print(self.tokenizer.decode(combined_ids.tolist()[0], skip_special_tokens=False))
        # print('================================================input_ids decode')

        logits = self(combined_ids, use_prefix=use_prefix)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        use_prefix: bool = True
    ):
        """
        If use_prefix=True, we prepend the trainable prefix embeddings.
        For 'reference' forward pass, call with use_prefix=False.
        """
        # print('================================================input_ids decode')
        # print(self.tokenizer.decode(input_ids.tolist()[0], skip_special_tokens=False))
        # print('================================================input_ids decode')

        batch_size, seq_len = input_ids.shape
        input_ids = input_ids.to(self.device)
        self.prefix_ids = self.prefix_ids.to(self.device)

        # template_start = torch.tensor([[     2,    106,   1645,    108]], dtype=torch.long).to(self.device)
        template_start = torch.tensor([[     2,    106,   9020,    108]], dtype=torch.long).to(self.device)
        template_start_len = int(template_start.shape[1])

        template_end = torch.tensor([[   107,    108]], dtype=torch.long).to(self.device)
        template_end_len = int(template_end.shape[1])

        """
        kill first [     2,    106,   1645,    108,   3940,    107,    108] tokens
        """

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=input_ids.device)
        
        # If we do NOT use prefix, we just do a regular forward through GPT
        if not use_prefix:
            # direct GPT forward

             # torch.Size([1, 4]) torch.Size([1, 39]) torch.Size([1, 2]) -> torch.Size([1, 45])
            formaled_input_ids = torch.cat([template_start, self.prefix_ids, template_end], dim=1)
            # print('================================================input_ids decode')
            # print(self.tokenizer.decode(torch.cat([formaled_input_ids, input_ids[:, 7:]], dim=1).tolist()[0], skip_special_tokens=False))
            # print('================================================input_ids decode')

             # torch.Size([1, 45]) -> torch.Size([1, 45, 3072])
            formaled_inputs_embeds = self.base_model.model.embed_tokens(formaled_input_ids)

             # torch.Size([1, 45, 3072]) -> torch.Size([1, 45, 3072])
            formaled_inputs_embeds = formaled_inputs_embeds.expand(batch_size, -1, -1)

             # torch.Size([1, 87]) -> torch.Size([1, 87, 3072])
            word_embeds = self.base_model.model.embed_tokens(input_ids[:, 7:])

             # torch.Size([1, 45, 3072]) torch.Size([1, 87, 3072]) -> torch.Size([1, 132, 3072])
            inputs_embeds = torch.cat([formaled_inputs_embeds, word_embeds], dim=1).to(input_ids.device)


            prefix_mask = torch.ones(batch_size, template_start_len + self.prefix_length + template_end_len - 7, dtype=torch.long, device=input_ids.device)
            extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            # print(inputs_embeds.device)
            # print(extended_mask.device)

            transformer_outputs = self.base_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_mask,
                return_dict=True,
                output_hidden_states=True
            )
            hidden_states = transformer_outputs.last_hidden_state
            
        else:
            template_start_embeds = self.base_model.model.embed_tokens(template_start).squeeze(0)
            template_end_embeds = self.base_model.model.embed_tokens(template_end).squeeze(0)

             # torch.Size([4, 3072]) torch.Size([39, 3072]) torch.Size([2, 3072]) -> torch.Size([45, 3072])
            unformaled_inputs_embeds = torch.cat([template_start_embeds, self.prefix_embeddings, template_end_embeds], dim=0)

             # torch.Size([45, 3072]) -> torch.Size([1, 45, 3072])
            unformaled_inputs_embeds = unformaled_inputs_embeds.unsqueeze(0).expand(batch_size, -1, -1)

             # torch.Size([1, 87]) -> torch.Size([1, 87, 3072])
            word_embeds = self.base_model.model.embed_tokens(input_ids[:, 7:])

             # torch.Size([1, 45, 3072]) torch.Size([1, 87, 3072]) -> torch.Size([1, 132, 3072])
            inputs_embeds = torch.cat([unformaled_inputs_embeds, word_embeds], dim=1).to(input_ids.device)
            

            prefix_mask = torch.ones(batch_size, template_start_len + self.prefix_length + template_end_len - 7, dtype=torch.long, device=input_ids.device)
            extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
            # Forward pass
            transformer_outputs = self.base_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_mask,
                return_dict=True,
                output_hidden_states=True
            )
            hidden_states = transformer_outputs.last_hidden_state
        
        # GPT LM head => next-token logits
        logits = self.base_model.lm_head(hidden_states)
        
        return logits

    def truncate_from_beginning(self, chat, only_str=False):

        if only_str:
            chat = [{'role': 'assistant', 'content': chat}]
        
        chat = [{'role': 'user', 'content': 'temp'}] + chat
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(torch.long).to(self.device)

        if input_ids.shape[1] > self.max_length:
            raise ValueError(f"Input length {input_ids.shape[1]} exceeds max_length {self.max_length}")

        # """
        # Truncate input so it does not exceed self.max_length tokens.
        # """
        # input_ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)], #False)], 
        #                          dtype=torch.long, device=self.device)
        
        # if input_ids.shape[1] > self.max_length:
        #     # Keep the last (max_length - 1) tokens + the very first token
        #     start_idx = input_ids.shape[1] - self.max_length + 1
        #     truncated_input_ids = torch.cat([
        #         input_ids[:, :1],
        #         input_ids[:, start_idx:]
        #     ], dim=1)
        #     return truncated_input_ids
        # else:
        #     return input_ids

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
            torch_dtype=torch_dtype,
            device=device, 
        )
        return model
    











