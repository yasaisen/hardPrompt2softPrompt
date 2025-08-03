"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506111354
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict
import os

from ...common.utils import (
    detect_string_type, 
    log_print, 
    get_trainable_params, 
    highlight, 
    highlight_show
)


WEIGHT_MAPPING_DICT = {
    'google/gemma-3-1b-it': 'hard2softModel_v1.0_2505142356.pth', 
}


class PrefixTuningPolicyModel(nn.Module):
    def __init__(self, 
        model_name: str, 
        prefix_prompt: str = None, 
        pretrain_path: str = None, 
        gradient_checkpointing: bool = False,
        device: str = "cuda", 
        torch_dtype = torch.float32,
        silent: bool = True,
    ):
        super().__init__()
        self.state_name = 'PrefixTuningPolicyModel'
        self.device = device
        # print()
        log_print(self.state_name, f"Building...", silent)

        self.torch_dtype = torch_dtype
        self.model_name = model_name

        if self.model_name == 'google/gemma-3-1b-it':
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                # attn_implementation="flash_attention_2", 
                torch_dtype=self.torch_dtype
            )

        elif self.model_name == 'google/gemma-3-4b-it':
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                # attn_implementation="flash_attention_2", 
                torch_dtype=self.torch_dtype
            ).language_model

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            log_print(self.state_name, f"pad_token=None", silent)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model.config.gradient_checkpointing = gradient_checkpointing

        self.prefix_check_ids = [2, 105, 2364, 107, 7617, 108, 3041, 106, 107]
        self.prefix_check_tensor_len = len(self.prefix_check_ids)
        self.prefix_lock_idx = 4

        if prefix_prompt is not None:
            log_print(self.state_name, f"prefix_prompt={prefix_prompt}", silent)
            self.prefix_prompt = prefix_prompt
            self.prefix_ids = self.tokenizer.encode(self.prefix_prompt, add_special_tokens=False, return_tensors="pt").to(torch.long)
            self.prefix_length = int(self.prefix_ids.shape[1])
            # self.max_length = self.base_model.config.max_position_embeddings - self.prefix_length
            self.max_length = self.tokenizer.model_max_length - self.prefix_length
                
            self.hidden_size = self.base_model.config.hidden_size
            word_embeds = self.base_model.model.embed_tokens(self.prefix_ids)

            self.prefix_embeddings = nn.Parameter(word_embeds.detach().clone().squeeze(), requires_grad=True)
            log_print(self.state_name, f"prefix_shape={self.prefix_embeddings.shape}", silent)

        if pretrain_path is not None:
            log_print(self.state_name, f"pretrain_path={pretrain_path}", silent)
            ckpt = torch.load(pretrain_path, weights_only=True)
            self.prefix_embeddings = nn.Parameter(ckpt['prefix_embeddings_state_dict'])
            self.prefix_length = int(self.prefix_embeddings.shape[0])
            self.prefix_ids = torch.tensor([[108] * self.prefix_length], dtype=torch.long).to(self.device)
            # self.max_length = self.base_model.config.max_position_embeddings - self.prefix_length
            self.max_length = self.tokenizer.model_max_length - self.prefix_length
            log_print(self.state_name, f"prefix_shape={self.prefix_embeddings.shape}", silent)

        log_print(self.state_name, f"max_length={self.max_length}", silent)

        for param in self.parameters():
            param.requires_grad = False
        self.prefix_embeddings.requires_grad = True

        log_print(self.state_name, f"requires_grad={sum(p.numel() for p in self.parameters() if p.requires_grad)}", silent)
        log_print(self.state_name, f"prefix_embeddings={self.prefix_embeddings.numel() if self.prefix_embeddings.requires_grad else 0}", silent)
        log_print(self.state_name, f"prefix_embeddings={self.prefix_embeddings.shape}", silent)
        log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self)}", silent)
        self.to(self.device)
        log_print(self.state_name, f"...Done\n", silent)

    # def generate_response(self, 
    #     messages_ids: torch.Tensor, 
    #     max_new_tokens: int = 100, 
    #     use_prefix: bool = True,
    #     temperature: float = 1.0,
    #     prefix_checker: bool = False, 
    #     output_ids: bool = False, 
    # ):
    #     self.eval()
    #     generated_ids = []
        
    #     for _ in range(max_new_tokens):
    #         input_ids = torch.cat([messages_ids, torch.tensor([generated_ids], dtype=torch.long, device=self.device)], dim=1)
    #         with torch.no_grad():
    #             logits = self(
    #                 input_ids=input_ids, 
    #                 use_prefix=use_prefix,
    #                 output_hidden_states=False,
    #                 prefix_checker=prefix_checker,
    #                 stage='decode',
    #             )
    #         next_token_logits = logits[0, -1, :]
    #         log_probs = F.log_softmax(next_token_logits / temperature, dim=-1)
            
    #         # greedy search
    #         next_token_id = torch.argmax(log_probs, dim=-1).item()
    #         generated_ids.append(next_token_id)
            
    #         if next_token_id == self.tokenizer.eos_token_id:
    #             break
                
    #     response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    #     generated_ids = torch.tensor(generated_ids, dtype=torch.long, device=self.device).unsqueeze(0)

    #     response = response.split('/n')[0]
        
    #     if output_ids:
    #         return response, generated_ids
    #     else:
    #         return response

    def generate_response_with_batch(self, 
        input_ids: torch.Tensor,  # [bsz, seq_len]
        attention_mask: torch.Tensor = None,  # [bsz, seq_len]
        max_new_tokens: int = 50, 
        use_prefix: bool = True,
        temperature: float = 1.0
    ):
        batch_size = input_ids.shape[0] # [batch_size, seq_len_padded]
        
        if attention_mask is None:
            attention_mask = (input_ids != pad_token_id).long()  # 1=token，0=padding
        
        seq_lengths = attention_mask.sum(dim=1)  # [batch_size]
        
        all_generated = input_ids.clone()
        
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                logits = self(
                    input_ids=all_generated, 
                    attention_mask=attention_mask,
                    use_prefix=use_prefix,
                    output_hidden_states=False,
                    stage='decode',
                )
                logits = logits[:, (self.prefix_length - 1):, :]
            
            next_token_logits = []
            for i in range(batch_size):
                curr_length = int(seq_lengths[i].item())
                next_token_logits.append(logits[i, curr_length-1:curr_length])
            
            next_token_logits = torch.cat(next_token_logits, dim=0)
            
            next_tokens = next_token_logits.argmax(dim=-1)  # [batch_size]
            
            for i in range(batch_size):
                if unfinished_sequences[i] == 1:
                    curr_length = int(seq_lengths[i].item())

                    if curr_length >= all_generated.shape[1]:
                        new_pad = torch.zeros((all_generated.size(0), 1), device=all_generated.device, dtype=all_generated.dtype)
                        all_generated = torch.cat([all_generated, new_pad], dim=1)
                        attention_mask = torch.cat([attention_mask, new_pad], dim=1)
                    all_generated[i, curr_length] = next_tokens[i]
                    
                    seq_lengths[i] += 1
                    attention_mask[i, curr_length] = 1
                    
                    if next_tokens[i] == self.tokenizer.eos_token_id:
                        unfinished_sequences[i] = 0

                    now_str = self.tokenizer.decode(next_tokens[i], skip_special_tokens=True)
                    if detect_string_type(now_str):
                        
                        # log_print('check', f"[{highlight()}] [{next_tokens[i]}] [{now_str}]")
                        unfinished_sequences[i] = 0

            if unfinished_sequences.sum() == 0:
                break

        batch_outputs = self.tokenizer.batch_decode(all_generated, skip_special_tokens=True)
        
        return batch_outputs, all_generated

    def full_forward(self, 
        messages_ids: torch.Tensor,
        response_ids: torch.Tensor,
        use_prefix: bool,
        temperature: float = 1.0
    ):
        combined_ids = torch.cat([messages_ids, response_ids], dim=1)
        # highlight_show('[full_forward] input_ids(decoded)', self.tokenizer.decode(combined_ids.tolist()[0], skip_special_tokens=False))

        logits, hidden_states = self(
            input_ids=combined_ids, 
            use_prefix=use_prefix,
            output_hidden_states=True,
        )
        response_logits = logits[:, -response_ids.shape[1]:]
        
        logp_gen = F.log_softmax(response_logits / temperature, dim=-1)
        old_logp = logp_gen.gather(
            -1, response_ids.unsqueeze(-1)
        ).squeeze(-1).detach()                  # [B, L_g]
        seq_old_logp = old_logp.sum(dim=1)      # [B]

        probs = torch.exp(logp_gen)
        entropy = -(probs * logp_gen).sum(dim=-1).mean()

        return response_logits, hidden_states, seq_old_logp, entropy

    def forward(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        use_prefix: bool = True,
        output_hidden_states: bool = False,
        prefix_checker: bool = True,
        stage: str = '',
    ):
        # highlight_show('[forward] input_ids(decoded)', self.tokenizer.decode(input_ids.tolist()[0], skip_special_tokens=False))

        template_start = torch.tensor([self.prefix_check_ids[:self.prefix_lock_idx]], dtype=torch.long).to(self.device) # system
        template_end = torch.tensor([self.prefix_check_ids[self.prefix_lock_idx + 1:]], dtype=torch.long).to(self.device)
        template_start_len = int(template_start.shape[1])
        template_end_len = int(template_end.shape[1])

        # log_print('forward', f"[{highlight('input_ids')}] {input_ids.shape} {input_ids}")

        check_tensor = torch.tensor(self.prefix_check_ids, dtype=torch.long).to(self.device)
        for bsz_idx in range(input_ids.shape[0]):
            if not torch.equal(input_ids[bsz_idx, :self.prefix_check_tensor_len], check_tensor):
                # raise ValueError(f"Input input_ids {bsz_idx} with not prefix [\n{self.tokenizer.decode(check_tensor.tolist()[0], skip_special_tokens=False)}\n] but [\n{self.tokenizer.decode(input_ids[bsz_idx, :self.prefix_check_tensor_len].tolist(), skip_special_tokens=False)}\n]")
                raise ValueError(f"Input input_ids {bsz_idx} with not prefix [\n{check_tensor}\n] but [\n{input_ids[bsz_idx, :self.prefix_check_tensor_len]}\n]")

        cutted_input_ids = input_ids[:, self.prefix_check_tensor_len:]
        batch_size, seq_len = cutted_input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, 
                template_start_len + self.prefix_length + template_end_len + seq_len, 
                dtype=torch.long, 
                device=cutted_input_ids.device
            )
        else:
            attn_pad = torch.ones((attention_mask.size(0), self.prefix_length - 1), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attn_pad, attention_mask], dim=1)

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
        
        if output_hidden_states:
            return logits, hidden_states
        else:
            return logits

    def chat_template_tokenizer(self, 
            chat_dicts: List[List[Dict[str, str]]], 
            max_length: int, 
        ):
        max_length = max_length - self.prefix_length
        prompt_text_list = []
        for chat_dict in chat_dicts:
            messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "temp"},]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "start"},]
                    }
                ] + chat_dict["messages"]
            ]
            
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )

            input_ids = self.tokenizer(
                prompt_text,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )['input_ids']
            if len(input_ids[0]) > max_length:
                raise ValueError(f"Input length {len(input_ids[0])} exceeds max_length {self.max_length}")

            # log_print('chat_template_tokenizer', f"[{highlight('prompt_text')}] {type(prompt_text[0])} {len(prompt_text[0])}")
            prompt_text_list += [prompt_text[0][5:]]

        temp_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'right'
        batched_inputs = self.tokenizer(
            prompt_text_list,
            padding=True, # "max_length",
            truncation=True,
            max_length=max_length, 
            return_tensors="pt"
        )
        self.tokenizer.padding_side = temp_padding_side
        
        batched_inputs = {
            k: (v.to(torch.bfloat16) if k != "input_ids" else v).to(self.device)
            for k, v in batched_inputs.items()
        }
        
        batched_input_ids = batched_inputs["input_ids"]
        batched_attention_mask = batched_inputs["attention_mask"]
        # log_print('chat_template_tokenizer', f"[{highlight('batched_input_ids')}] {batched_input_ids.shape}")
        # log_print('chat_template_tokenizer', f"[{highlight('batched_attention_mask')}] {batched_attention_mask.shape}")
        
        # log_print('chat_template_tokenizer', f"[{highlight('end')}]")
        return batched_input_ids, batched_attention_mask
    
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

    @classmethod
    def from_pretrained(cls, 
        model_name: str, 
        ckpt_path: str = None, 
    ):
        if ckpt_path is None or not os.path.isfile(ckpt_path):
            print('downloading from huggingface...')
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id   = 'yasaisen/hardPrompt2softPrompt',
                filename  = WEIGHT_MAPPING_DICT[model_name],
                # revision  = revision,
                # cache_dir = cache_dir, 
                # resume_download = True, 
            )
        else:
            print('loading from local...')

        model = cls(
            model_name=model_name,
            pretrain_path=ckpt_path, 
        )
        return model










    