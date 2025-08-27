"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2508031527
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
    highlight_show, 
    grad_checker
)


WEIGHT_MAPPING_DICT = {
    'google/gemma-3-1b-it': 'hard2softModel_v1.0_2505142356.pth', 
}


class PrefixTuningPolicyModel(nn.Module):
    def __init__(self, 
        model_name: str, 

        prefix_prompt: str = None, 
        checkpoint_path: str = None, 
        model_max_length: int = None, 
        min_response_length: int = None, 
        gradient_checkpointing: bool = False,
        device: str = "cuda", 
        torch_dtype = torch.float32,
        silent: bool = False,
    ):
        super().__init__()
        self.state_name = 'PrefixTuningPolicyModel'
        self.device = device
        # print()
        log_print(self.state_name, f"Building...", silent)

        self.init_model_and_tokenizer( 
            model_name=model_name, 
            torch_dtype=torch_dtype, 
            gradient_checkpointing=gradient_checkpointing, 
            model_max_length=model_max_length, 
        )
        self.min_response_length = min_response_length
        
        self.init_prefix_template()

        if prefix_prompt is not None:
            self.init_prefixPrompt2embedding( 
                prefix_prompt=prefix_prompt, 
                silent=silent, 
            )
        if checkpoint_path is not None:
            self.init_checkPoint2embedding( 
                checkpoint_path=checkpoint_path, 
                silent=silent, 
            )

        log_print(self.state_name, f"prefix_shape={self.prefix_embeddings.shape}", silent)
        log_print(self.state_name, f"max_length={self.max_input_length}", silent)

        for param in self.parameters():
            param.requires_grad = False
        self.prefix_embeddings.requires_grad = True

        log_print(self.state_name, f"requires_grad={sum(p.numel() for p in self.parameters() if p.requires_grad)}", silent)
        log_print(self.state_name, f"prefix_embeddings={self.prefix_embeddings.numel() if self.prefix_embeddings.requires_grad else 0}", silent)
        log_print(self.state_name, f"prefix_embeddings={self.prefix_embeddings.shape}", silent)
        log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self)}", silent)
        self.to(self.device)
        log_print(self.state_name, f"...Done\n", silent)

    def init_model_and_tokenizer(self, 
        model_name: str, 
        torch_dtype = torch.float32,
        gradient_checkpointing: bool = False, 
        model_max_length: int = None, 
    ):
        """
        set: [`base_model`, `tokenizer`, `model_max_length`]
        """
        self.model_name = model_name
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            # attn_implementation="flash_attention_2", 
            torch_dtype=torch_dtype
        )
        if self.model_name == 'google/gemma-3-4b-it':
            self.base_model = self.base_model.language_model
        self.base_model.config.gradient_checkpointing = gradient_checkpointing
        self.hidden_size = self.base_model.config.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_max_length = model_max_length
        if self.model_max_length is None:
            self.model_max_length = self.tokenizer.model_max_length
            # self.model_max_length = self.base_model.config.max_position_embeddings

    def init_prefixPrompt2embedding(self, 
        prefix_prompt: str, 
        silent: bool = True, 
    ):
        """
        set: [`prefix_embeddings`, `prefix_length`, `max_input_length`, `prefix_ids`]
        """
        log_print(self.state_name, f"prefix_prompt={prefix_prompt}", silent)
        self.prefix_prompt = prefix_prompt

        self.prefix_ids = self.tokenizer.encode(
            self.prefix_prompt, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).to(torch.long)
        self.prefix_embeddings = nn.Parameter(
            self.base_model.model.embed_tokens(
                self.prefix_ids
            ).detach().clone().squeeze(), 
            requires_grad=True, 
        )

        self.prefix_length = int(self.prefix_ids.shape[1])
        self.max_input_length = self.model_max_length - self.prefix_length

    def init_checkPoint2embedding(self, 
        checkpoint_path: str, 
        silent: bool = True, 
    ):
        """
        set: [`prefix_embeddings`, `prefix_length`, `max_input_length`, `prefix_ids`]
        """
        log_print(self.state_name, f"checkpoint_path={checkpoint_path}", silent)
        ckpt = torch.load(checkpoint_path, weights_only=True)

        self.prefix_ids = ckpt['prefix_ids']
        self.prefix_embeddings = nn.Parameter(
            ckpt['prefix_embeddings_state_dict'], 
            requires_grad=True, 
        )

        self.prefix_length = int(self.prefix_ids.shape[1])
        self.max_input_length = self.model_max_length - self.prefix_length

    def init_prefix_template(self,
        prefix_check_ids: List = [2, 105, 2364, 107, 7617, 108, 3041, 106, 107], 
        prefix_lock_idx: int = 4, 
    ):
        """
        set: [`prefix_template_messages`, `prefix_check_ids`, `template_start_ids`, `template_end_ids`, `template_start_ids_len`, `template_end_ids_len`]
        """
        self.prefix_template_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "temp"},]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "start"},]
            }
        ]
        # prompt_text = self.tokenizer.apply_chat_template(
        #     [self.prefix_template_messages],
        #     add_generation_prompt=True,
        #     tokenize=False
        # )
        # self.prefix_check_ids = self.tokenizer(
        #     [prompt_text[0][5:]], # remove <bos> that will be given next time
        #     padding=True,
        #     truncation=True,
        #     return_tensors="pt"
        # )['input_ids']
        
        self.prefix_check_ids = self.tokenizer.apply_chat_template(
            [self.prefix_template_messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )['input_ids'].tolist()[0] # [2, 105, 2364, 107, 7617, 108, 3041, 106, 107, 105, 4368, 107]
        # log_print('tokenizer', f"[{highlight('input_ids')}] {self.prefix_check_ids}")

        # self.prefix_check_ids = prefix_check_ids
        self.prefix_check_tensor_len = len(self.prefix_check_ids)
        self.prefix_lock_idx = prefix_lock_idx

        self.template_start_ids = torch.tensor([self.prefix_check_ids[:self.prefix_lock_idx]], dtype=torch.long).to(self.device) # system
        self.template_start_ids_len = int(self.template_start_ids.shape[1])

        self.template_end_ids = torch.tensor([self.prefix_check_ids[self.prefix_lock_idx + 1:]], dtype=torch.long).to(self.device)
        self.template_end_ids_len = int(self.template_end_ids.shape[1])

    def prefix_checker(self, 
        input_ids: torch.Tensor, 
    ):
        check_tensor = torch.tensor(self.prefix_check_ids, dtype=torch.long).to(self.device)
        for bsz_idx in range(input_ids.shape[0]):
            if not torch.equal(input_ids[bsz_idx, :self.prefix_check_tensor_len], check_tensor):
                # raise ValueError(f"Input input_ids {bsz_idx} with not prefix [\n{self.tokenizer.decode(check_tensor.tolist()[0], skip_special_tokens=False)}\n] but [\n{self.tokenizer.decode(input_ids[bsz_idx, :self.prefix_check_tensor_len].tolist(), skip_special_tokens=False)}\n]")
                raise ValueError(f"Input input_ids {bsz_idx} with not prefix [\n{check_tensor}\n] but [\n{input_ids[bsz_idx, :self.prefix_check_tensor_len]}\n]")

    def tokenize_maxlencut_checker(self, 
        prompt_text: str, 
    ):
        input_ids = self.tokenizer(
            prompt_text,
            padding=True,
            return_tensors="pt"
        )['input_ids'][0]
        # log_print('tokenize_maxlencut_checker', f"[{highlight('len(input_ids)')}] {len(input_ids)}")
        if len(input_ids) > (self.max_input_length - self.min_response_length):
            raise ValueError(f"Input length {len(input_ids)} exceeds max_input_length {(self.max_input_length - self.min_response_length)}")

    @torch.no_grad()
    def generate_response_with_batch(self,
        input_ids: torch.Tensor, # [bsz, floating_prompt_len]
        attention_mask: torch.Tensor = None, # [bsz, floating_prompt_len]
        use_prefix: bool = True,
        temperature: float = 1.0
    ):
        bsz, floating_prompt_len = input_ids.shape  # [bsz, floating_prompt_len]
        max_new_tokens = self.max_input_length - floating_prompt_len

        pad_token_id = self.tokenizer.pad_token_id

        if attention_mask is None:
            attention_mask = (input_ids != pad_token_id).long()

        if attention_mask.shape[1] < self.max_input_length:
            pad_mask = torch.zeros(
                (bsz, self.max_input_length - attention_mask.shape[1]),
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([attention_mask, pad_mask], dim=1) # [bsz, max_input_length]
        seq_lengths = attention_mask.sum(dim=1)

        all_generated = torch.full( # [bsz, max_input_length]
            (bsz, self.max_input_length), 
            pad_token_id,
            dtype=input_ids.dtype, 
            device=input_ids.device
        )
        all_generated[:, :floating_prompt_len] = input_ids

        only_b_response = torch.full( # [bsz, max_input_length]
            (bsz, self.max_input_length),
            pad_token_id,
            dtype=input_ids.dtype, 
            device='cpu'
        )
        
        unfinished_sequences = torch.ones(bsz, dtype=torch.long, device=input_ids.device)
        for step in range(max_new_tokens):
            current_max_len = int(seq_lengths.max().item())

            with torch.no_grad():
                logits = self(
                    input_ids=all_generated[:, :current_max_len],
                    attention_mask=attention_mask[:, :current_max_len],
                    use_prefix=use_prefix,
                    output_hidden_states=False,
                )
                logits = logits[:, (self.prefix_length - 1):, :]

            next_token_logits = []
            for i in range(bsz):
                curr_length = int(seq_lengths[i].item())
                next_token_logits.append(logits[i, curr_length - 1:curr_length])

            next_token_logits = torch.cat(next_token_logits, dim=0)

            next_tokens = next_token_logits.argmax(dim=-1)  # [bsz]

            for i in range(bsz):
                if unfinished_sequences[i] == 1:
                    curr_length = int(seq_lengths[i].item())

                    # Ensure we do not write beyond the preallocated buffer
                    if curr_length >= self.max_input_length:
                        continue
                    
                    all_generated[i, curr_length] = next_tokens[i]
                    only_b_response[i, curr_length] = next_tokens[i]

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

        final_max_len = int(seq_lengths.max().item())

        del all_generated
        b_response_text = self.tokenizer.batch_decode(
            only_b_response, skip_special_tokens=True
        )
        batched_inputs = self.tokenizer(
            b_response_text,
            padding=True,
            return_tensors="pt"
        )
        batched_inputs = {
            k: (v.to(torch.bfloat16) if k != "input_ids" else v).to(self.device)
            for k, v in batched_inputs.items()
        }
        b_response_ids = batched_inputs["input_ids"]
        # b_response_text: [bsz, (response_text)]
        # b_response_ids: [bsz, max_input_length - floating_prompt_len]
        return b_response_text, b_response_ids

    def full_forward(self, 
        messages_ids: torch.Tensor, # [bsz, floating_prompt_len]
        response_ids: torch.Tensor, # [bsz, max_input_length - floating_prompt_len]
        use_prefix: bool,
        temperature: float = 1.0
    ):
        combined_ids = torch.cat([messages_ids, response_ids], dim=1)
        logits, hidden_states = self(
            input_ids=combined_ids, 
            use_prefix=use_prefix,
            output_hidden_states=True,
        )
        last_prompt_hidden_state = hidden_states[:, messages_ids.shape[1] - 1 + self.prefix_length - 1, :].detach() # [bsz, hidden_size]

        response_logits = logits[:, -response_ids.shape[1]:]
        logp_gen = F.log_softmax(response_logits / temperature, dim=-1)
        old_logp = logp_gen.gather(
            -1, response_ids.unsqueeze(-1)
        ).squeeze(-1)
        seq_old_logp = old_logp.sum(dim=1)

        probs = torch.exp(logp_gen)
        entropy = -(probs * logp_gen).sum(dim=-1).mean()

        # last_prompt_hidden_state: [bsz, hidden_size]
        # seq_old_logp: [bsz]
        # entropy: [bsz]
        # response_logits: [bsz, max_new_tokens, vocab_size]
        return last_prompt_hidden_state, seq_old_logp, entropy, response_logits

    def forward(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        use_prefix: bool = True,
        output_hidden_states: bool = False,
    ):
        # highlight_show('[forward] input_ids(decoded)', self.tokenizer.decode(input_ids.tolist()[0], skip_special_tokens=False))

        self.prefix_checker(input_ids=input_ids)
        cutted_input_ids = input_ids[:, self.prefix_check_tensor_len:]
        batch_size, seq_len = cutted_input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, 
                self.template_start_ids_len + self.prefix_length + self.template_end_ids_len + seq_len, 
                dtype=torch.long, 
                device=cutted_input_ids.device
            )
        else:
            attn_pad = torch.ones((attention_mask.size(0), self.prefix_length - 1), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attn_pad, attention_mask], dim=1)

        cutted_input_ids = cutted_input_ids.to(self.device)
        self.prefix_ids = self.prefix_ids.to(self.device)
        self.prefix_embeddings = self.prefix_embeddings.to(self.device)

        formaled_input_ids = torch.cat([self.template_start_ids, self.prefix_ids, self.template_end_ids], dim=1)
        # highlight_show('input_ids(decoded)', self.tokenizer.decode(torch.cat([formaled_input_ids, cutted_input_ids], dim=1).tolist()[0], skip_special_tokens=False))
        formaled_inputs_embeds = self.base_model.model.embed_tokens(formaled_input_ids)
        
        if use_prefix:
            formaled_inputs_embeds = torch.cat([
                formaled_inputs_embeds[:, :self.template_start_ids_len], 
                self.prefix_embeddings.unsqueeze(0), 
                formaled_inputs_embeds[:, -self.template_end_ids_len:]
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
        b_messages, # [bsz, (chat_format)]
    ):
        prompt_text_list = [] # [bsz, (formated_chat_str)]
        for context_messages in b_messages:
            messages = [self.prefix_template_messages + context_messages] # List[List[Dict]]
            
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            prompt_text = [prompt_text[0][5:]] # remove <bos> that will be given next time
            self.tokenize_maxlencut_checker(prompt_text=prompt_text)

            prompt_text_list += prompt_text

        temp_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'right'
        batched_inputs = self.tokenizer(
            prompt_text_list, # [bsz, (formated_chat_str)]
            padding=True,
            # truncation=True,
            # max_length=self.max_input_length, 
            return_tensors="pt"
        )
        self.tokenizer.padding_side = temp_padding_side
        
        batched_inputs = {
            k: (v.to(torch.bfloat16) if k != "input_ids" else v).to(self.device)
            for k, v in batched_inputs.items()
        }
        
        batched_input_ids = batched_inputs["input_ids"]
        batched_attention_mask = batched_inputs["attention_mask"]

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
            model_max_length = int(policy_model_cfg.get("max_length"))
            min_response_length = int(policy_model_cfg.get("min_response_length"))
            gradient_checkpointing = bool(policy_model_cfg.get("grad_checkpointing"))
            torch_dtype = dtype_map[str(policy_model_cfg.get("torch_dtype"))]

        model = cls(
            model_name=policy_model_name,
            prefix_prompt=prefix_prompt, 
            checkpoint_path=policy_model_path, 
            model_max_length=model_max_length, 
            min_response_length=min_response_length, 
            gradient_checkpointing=gradient_checkpointing,
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
                repo_id = 'yasaisen/hardPrompt2softPrompt',
                filename = WEIGHT_MAPPING_DICT[model_name],
                # revision  = revision,
                # cache_dir = cache_dir, 
                # resume_download = True, 
            )
        else:
            print('loading from local...')

        model = cls(
            model_name=model_name,
            checkpoint_path=ckpt_path, 
        )
        return model










    