"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2508251733
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict
from tqdm import tqdm
from statistics import mean

from ...common.utils import log_print, highlight_show, highlight, grad_checker
from ...models.rewardModel.modeling_rewardModel import ComparativeRewardModel
from ...models.policyModel.modeling_policyModel import PrefixTuningPolicyModel
from ...models.valueHead.modeling_valueHead import ValueHead

from dataclasses import dataclass

def off_loader(
    sample_dict: Dict, 
    into_log: bool = False, 
    device: str = 'cpu', 
):
    for key in list(sample_dict.keys()):
        if isinstance(sample_dict[key], torch.Tensor):
            sample_dict[key] = sample_dict[key].to(device)
        if into_log:
            if isinstance(sample_dict[key], torch.Tensor):
                sample_dict[key] = sample_dict[key].clone().detach().cpu().mean().item()
    torch.cuda.empty_cache()
    return sample_dict

@dataclass
class optConfig:
    learning_rate: float
    weight_decay: float
    max_lr: float
    num_epoch: int
    steps_per_epoch: int
    pct_start: float
    anneal_strategy: str
    
    def get_optimizer_scheduler(self, 
        params, 
    ):
        optimizer = optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            epochs=self.num_epoch,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy
        )
        return optimizer, scheduler

class SingleStepPPOTrainer:
    def __init__(self,
        policy_model: PrefixTuningPolicyModel,
        reward_model: ComparativeRewardModel,
        value_head: ValueHead, 
        policy_opt_config: optConfig, 
        value_opt_config: optConfig, 
        device: str = "cuda",
        
        temperature: float = 1.0,
        use_kl: bool = False, 
        clip_epsilon: float = 0.2, 
        max_kl: float = 0.2, 
        valueL_coef: float = 0.3, 
        klL_coef: float = 0.1, 
        entropyL_coef: float = 0.01, 
        max_grad_norm: float = 1.0, 
    ):
        self.state_name = 'SingleStepPPOTrainer'
        self.device = device
        print()
        log_print(self.state_name, f"Building...")

        self.temperature = temperature
        self.use_kl = use_kl
        self.clip_epsilon = clip_epsilon
        self.max_kl = max_kl
        self.valueL_coef = valueL_coef
        self.klL_coef = klL_coef
        self.entropyL_coef = entropyL_coef

        self.max_grad_norm = max_grad_norm

        self.policy = policy_model.to(self.device)
        self.reward_model = reward_model.to(self.device)
        self.value_head = value_head.to(self.device)

        self.optimizer_policy, self.scheduler_policy = policy_opt_config.get_optimizer_scheduler(
            params=[self.policy.prefix_embeddings]
        )
        self.optimizer_value, self.scheduler_value = value_opt_config.get_optimizer_scheduler(
            params=self.value_head.parameters()
        )
        log_print(self.state_name, f"...Done\n")

    def batch_processor(self,
        b_samples # [bsz, {'messages': (chat_format), 'context': str}]
    ):
        b_messages = []
        b_contexts = []
        for sample in b_samples:
            b_messages += [sample['messages']]
            b_contexts += [sample['context']]

        return b_messages, b_contexts

    @torch.no_grad()
    def get_response(self,
        b_messages: List[List[Dict]], # [bsz, (chat_format)]
        use_prefix: bool, 
        temperature: float, 
    ):
        b_messages_ids, b_message_attnmask = self.policy.chat_template_tokenizer(
            b_messages=b_messages, # [bsz, (chat_format)]
        )
        b_response_text, b_response_ids = self.policy.generate_response_with_batch(
            input_ids=b_messages_ids, # [bsz, max_token_len]
            attention_mask=b_message_attnmask, # [bsz, max_token_len]
            use_prefix=use_prefix,
            temperature=temperature,
        )
        # b_messages_ids: [bsz, max_token_len]
        # b_response_text: [bsz, (response_text)]
        # b_response_ids: [bsz, max_new_tokens]
        return b_messages_ids, b_response_text, b_response_ids
    
    def get_full_forward(self,
        b_messages_ids: torch.Tensor, # [bsz, max_token_len]
        b_response_ids: torch.Tensor, # [bsz, max_new_tokens]
        use_prefix: bool,
        no_grad: bool, 
        temperature: float, 
    ):
        if (not no_grad) and use_prefix:
            self.policy.train()
            b_last_prompt_hidden_state, b_seq_logp, b_entropy, b_response_logits = self.policy.full_forward(
                messages_ids=b_messages_ids, # [bsz, max_token_len]
                response_ids=b_response_ids, # [bsz, max_new_tokens]
                use_prefix=True,
                temperature=temperature,
            )
        else:
            self.policy.eval()
            with torch.no_grad():
                b_last_prompt_hidden_state, b_seq_logp, b_entropy, b_response_logits = self.policy.full_forward(
                    messages_ids=b_messages_ids, # [bsz, max_token_len]
                    response_ids=b_response_ids, # [bsz, max_new_tokens]
                    use_prefix=use_prefix,
                    temperature=temperature,
                )
                b_seq_logp = b_seq_logp.detach()
                b_entropy = b_entropy.detach()
        # b_last_prompt_hidden_state: [bsz, hidden_size]
        # b_seq_logp: [bsz]
        # b_entropy: [bsz]
        # b_response_logits: [bsz, max_new_tokens, vocab_size]
        return b_last_prompt_hidden_state.detach(), b_seq_logp, b_entropy, b_response_logits
        
    def get_seq_values(self,
        b_last_prompt_hidden_state: torch.Tensor, # [bsz, hidden_size]
        no_grad: bool, 
    ):
        if not no_grad:
            self.value_head.train()
            seq_values  = self.value_head(b_last_prompt_hidden_state.detach())
        else:
            self.value_head.eval()
            with torch.no_grad():
                seq_values  = self.value_head(b_last_prompt_hidden_state.detach()).detach()
        # seq_values: [bsz]
        return seq_values

    @torch.no_grad()
    def compute_batch_reward(self, 
        b_messages: List[str], # [bsz, (chat_format)]
        b_response_text: List[str], # [bsz, (response_text)]
    ) -> List[float]:
        
        reward_list = []
        for context, response in zip(b_messages, b_response_text):
            context_ids = self.reward_model.truncate_from_beginning(str(context))
            response_ids = self.reward_model.truncate_from_beginning(response)

            context_mask = torch.ones_like(context_ids)
            response_mask = torch.ones_like(response_ids)

            self.reward_model.eval()
            with torch.no_grad():
                reward = self.reward_model.get_reward(
                    context_ids,
                    context_mask,
                    response_ids,
                    response_mask
                ).detach()
            reward_list += [reward]
        
        rewards = torch.tensor(reward_list).to(self.device)
        # rewards: [bsz]
        return rewards

    def compute_stepwise_kl(self, 
        policy_logits: torch.Tensor, 
        reference_logits: torch.Tensor, 
    ) -> torch.Tensor:
        policy_logits = policy_logits.to(torch.float32)
        reference_logits = reference_logits.to(torch.float32)

        policy_probs = F.softmax(policy_logits, dim=-1)
        reference_probs = F.softmax(reference_logits, dim=-1)
        log_policy_probs = F.log_softmax(policy_logits, dim=-1)
        
        # KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
        kl_divergence = policy_probs * (
            log_policy_probs - torch.log(reference_probs + 1e-10)
        )
        kl_divergence = kl_divergence.sum(dim=-1)
        return kl_divergence

    def compute_seq_avg_kl(self, 
        policy_logits: torch.Tensor, 
        reference_logits: torch.Tensor, 
    ):
        total_kl = torch.tensor(0.0, device=self.device)
        new_token_len = int(policy_logits.shape[1])
        for step_idx in range(new_token_len):
            step_kl = self.compute_stepwise_kl(
                policy_logits=policy_logits, 
                reference_logits=reference_logits,
            )
            total_kl += step_kl.mean()
        avg_kl = total_kl / new_token_len

        return avg_kl

    @torch.no_grad()
    def collect_rollouts(self, 
        b_dataset: List[List[Dict[str, str]]],  # [num_batches, bsz, chat_format]
        valid: bool = False, 
        calc_diff: bool = True, 
    ) -> List[Dict]:
        log_print('collect_rollouts', f"Processing {len(b_dataset)} batches...")
        
        rollout_list = []
        valid_rollout_list = []
        for b_samples in tqdm(b_dataset):
            rollout = {}
            b_messages, b_contexts = self.batch_processor(
                b_samples=b_samples
            )
            b_messages_ids, b_response_text, b_response_ids = self.get_response(
                b_messages=b_messages, # [bsz, (chat_format)]
                use_prefix=True, 
                temperature=self.temperature, 
            )
            rollout['context_text'] = b_contexts # [bsz, str(chat_format)]
            rollout['messages_ids'] = b_messages_ids # [bsz, max_token_len]
            rollout['messages_text'] = b_messages # [bsz, (chat_format)]
            rollout['response_ids'] = b_response_ids # [bsz, max_new_tokens]
            rollout['response_text'] = b_response_text # [bsz, (response_text)]

            ###############################################

            b_last_prompt_hidden_state, b_seq_logp, _, _ = self.get_full_forward(
                b_messages_ids=b_messages_ids, # [bsz, (chat_format)]
                b_response_ids=b_response_ids, # [bsz, max_new_tokens]
                use_prefix=True, 
                no_grad=True, 
                temperature=self.temperature, 
            )
            # b_seq_logp: [bsz]
            rollout['seq_old_logp'] = b_seq_logp.detach()

            ###############################################

            b_seq_values = self.get_seq_values(
                b_last_prompt_hidden_state=b_last_prompt_hidden_state, # [bsz, hidden_size]
                no_grad=True, 
            )
            # b_seq_values: [bsz]
            rollout['seq_old_values'] = b_seq_values

            ###############################################

            b_rewards = self.compute_batch_reward(
                b_messages=b_contexts, # [bsz, str(chat_format)]
                b_response_text=b_response_text, # [bsz, (response_text)]
            )
            b_rewards_n = (b_rewards - b_rewards.mean()) / b_rewards.std().clamp_min(1e-8)
            # b_rewards: [bsz]
            rollout['rewards'] = b_rewards.detach()
            rollout['rewards_n'] = b_rewards_n.detach()

            b_advantages = b_rewards - b_seq_values # [bsz]
            # b_advantages_n = (b_advantages - b_advantages.mean()) / b_advantages.std().clamp_min(1e-8)
            # b_advantages: [bsz]
            rollout['advantages'] = b_advantages.detach()

            ###############################################

            if self.use_kl:
                _, _, _, b_ref_response_logits = self.get_full_forward(
                    b_messages_ids=b_messages_ids, # [bsz, (chat_format)]
                    b_response_ids=b_response_ids, # [bsz, max_new_tokens]
                    use_prefix=False, 
                    no_grad=True, 
                    temperature=self.temperature, 
                )
                # b_ref_response_logits: [bsz, max_new_tokens, vocab_size]
                rollout['b_ref_response_logits'] = b_ref_response_logits.detach()

            ###############################################

            if calc_diff or valid:
                _, b_response_text_ref, _ = self.get_response(
                    b_messages=b_messages, # [bsz, (chat_format)]
                    use_prefix=False, 
                    temperature=self.temperature, 
                )
                rollout['response_text_ref'] = b_response_text_ref # [bsz, (response_text)]

                b_rewards_ref = self.compute_batch_reward(
                    b_messages=b_contexts, # [bsz, str(chat_format)]
                    b_response_text=b_response_text_ref, # [bsz, (response_text)]
                )
                # b_rewards: [bsz]
                rollout['rewards_ref'] = b_rewards_ref.detach()
                rollout['reward_diff'] = (rollout['rewards'] - rollout['rewards_ref']).mean()

            if valid:
                valid_rollout = {}
                valid_rollout['messages_text'] = rollout['messages_text']
                valid_rollout['pol_response_text'] = rollout['response_text']
                valid_rollout['pol_rewards'] = rollout['rewards']
                valid_rollout['ref_response_text'] = rollout['response_text_ref']
                valid_rollout['ref_rewards'] = rollout['rewards_ref']
                valid_rollout['reward_diff'] = rollout['reward_diff']

                print("="*30)
                for key in list(valid_rollout.keys()):
                    if 'text' not in key:
                        log_print("collect_rollouts", f"{highlight(key)} {valid_rollout[key]}")
                print("="*30, '\n')

                valid_rollout = off_loader(
                    sample_dict=valid_rollout, 
                    into_log=True, 
                    device='cpu', 
                )
                valid_rollout_list += [valid_rollout]

            rollout = off_loader(
                sample_dict=rollout, 
                device='cpu', 
            )
            rollout_list += [rollout]

        if valid:
            return valid_rollout_list

        return rollout_list

    def ppo_loss(self, 
        rollout_list: List[Dict], 
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:

        losses = {
            'policy_loss': torch.zeros((), device=self.device), 
            'value_loss': torch.zeros((), device=self.device),
            'entropy_loss': torch.zeros((), device=self.device),
        }
        if self.use_kl:
            losses['kl_loss'] = torch.zeros((), device=self.device)

        metric_list = []
        for sample in rollout_list:
            sample = off_loader(
                sample_dict=sample, 
                device=self.device, 
            )
            metrics = {}
            metrics['reward_diff'] = sample['reward_diff']
            metrics['old_logp'] = sample['seq_old_logp']
            metrics['old_values'] = sample['seq_old_values']
            metrics['old_rewards'] = sample['rewards_n']
            metrics['advantages'] = sample['advantages']

            b_last_prompt_hidden_state, b_seq_logp, b_entropy, b_response_logits = self.get_full_forward(
                b_messages_ids=sample['messages_ids'], # [bsz, (chat_format)]
                b_response_ids=sample['response_ids'], # [bsz, max_new_tokens]
                use_prefix=True, 
                no_grad=False, 
                temperature=self.temperature, 
            )
            # b_seq_logp: [bsz]
            # b_entropy: [bsz]
            # b_response_logits: [bsz, max_new_tokens, vocab_size]
            metrics['seq_new_logp'] = b_seq_logp
            metrics['entropy'] = b_entropy

            b_seq_values = self.get_seq_values(
                b_last_prompt_hidden_state=b_last_prompt_hidden_state, # [bsz, hidden_size]
                no_grad=False, 
            )
            # b_seq_values: [bsz]
            metrics['seq_new_values'] = b_seq_values

            ###############################################

            # PPO policy loss
            ratio = torch.exp(b_seq_logp - sample['seq_old_logp']) # [bsz]
            metrics['ratio'] = ratio
            surr1 = ratio * sample['advantages'] # [bsz]
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * sample['advantages']
            policy_loss = -torch.min(surr1, surr2).mean() # (scalar)
            losses['policy_loss'] = losses['policy_loss'] + policy_loss

            # Value loss
            value_loss = F.mse_loss(b_seq_values, sample['rewards']) # (scalar)
            losses['value_loss'] = losses['value_loss'] + value_loss

            # KL loss
            if self.use_kl:
                avg_kl = self.compute_seq_avg_kl( 
                    policy_logits=b_response_logits, 
                    reference_logits=sample['b_ref_response_logits'], 
                )
                kl_loss = torch.max(
                    avg_kl - self.max_kl, 
                    torch.tensor(0.0, device=self.device)
                ) # (scalar)
                losses['kl_loss'] = losses['kl_loss'] + kl_loss

            # Entropy loss
            entropy_loss = -b_entropy.mean() # (scalar)
            losses['entropy_loss'] = losses['entropy_loss'] + entropy_loss

            print("="*30)
            for idx, key in enumerate(list(metrics.keys())):
                log_print("ppo_loss", f"{highlight(key)} {metrics[key]}")
                if isinstance(metrics[key], torch.Tensor) and metrics[key].requires_grad and idx <= 3:
                    raise f"[{key}] meowmeowmeowmeowmeowmeow"
                if isinstance(metrics[key], torch.Tensor) and (not metrics[key].requires_grad) and idx > 3:
                    raise f"[{key}] meowmeowmeowmeowmeowmeow"
            print("="*30, '\n')

            metrics = off_loader(
                sample_dict=metrics, 
                into_log=True, 
                device='cpu', 
            )
            metric_list += [metrics]

        avg_metrics_dict = {k: mean(d[k] for d in metric_list) for k in metric_list[0].keys()}

        for key in list(losses.keys()):
            losses[key] = losses[key] / len(rollout_list)
            if not losses[key].requires_grad:
                raise f"[{key}] meowmeowmeowmeowmeowmeow"

        losses['total_loss'] = (
            losses['policy_loss'] + 
            # losses['kl_loss'] + 
            # losses['value_loss'] + 
            losses['entropy_loss']
        )
        # losses['total_loss'] = losses['policy_loss']
        
        return losses, avg_metrics_dict

    def backward(self, 
        losses: Dict
    ):
        self.optimizer_policy.zero_grad()
        self.optimizer_value.zero_grad()
        losses['total_loss'].backward()
        losses['value_loss'].backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in self.policy.parameters() if p.requires_grad),
            self.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.value_head.parameters(),
            self.max_grad_norm
        )
        self.optimizer_policy.step()
        self.optimizer_value.step()
        self.scheduler_policy.step()
        self.scheduler_value.step()

    @classmethod
    def from_config(cls, 
        cfg, 
        policy_model: PrefixTuningPolicyModel,
        reward_model: ComparativeRewardModel,
        steps_per_epoch: int,
    ):
        device = str(cfg['task'].get("device"))

        if cfg.get("task") is not None:
            trainer_cfg = cfg['task']
            device = str(trainer_cfg.get("device"))

            temperature = float(trainer_cfg.get("temperature"))
            use_kl = bool(trainer_cfg.get("use_kl"))
            clip_epsilon = float(trainer_cfg.get("clip_epsilon"))
            max_kl = float(trainer_cfg.get("max_kl"))
            valueL_coef = float(trainer_cfg.get("valueL_coef"))
            klL_coef = float(trainer_cfg.get("klL_coef"))
            entropyL_coef = float(trainer_cfg.get("entropyL_coef"))
            max_grad_norm = float(trainer_cfg.get("max_grad_norm"))

            learning_rate = float(trainer_cfg.get("learning_rate"))
            weight_decay = float(trainer_cfg.get("weight_decay"))
            max_lr = float(trainer_cfg.get("max_lr"))
            num_epoch = int(trainer_cfg.get("num_epoch"))
            pct_start = float(trainer_cfg.get("pct_start"))
            anneal_strategy = str(trainer_cfg.get("anneal_strategy"))

        value_head = ValueHead(policy_model.hidden_size)

        policy_opt_config = optConfig(
            learning_rate=learning_rate, 
            weight_decay=weight_decay, 
            max_lr=max_lr, 
            num_epoch=num_epoch, 
            steps_per_epoch=steps_per_epoch, 
            pct_start=pct_start, 
            anneal_strategy=anneal_strategy, 
        )

        value_opt_config = optConfig(
            learning_rate=learning_rate, 
            weight_decay=weight_decay, 
            max_lr=max_lr, 
            num_epoch=num_epoch, 
            steps_per_epoch=steps_per_epoch, 
            pct_start=pct_start, 
            anneal_strategy=anneal_strategy, 
        )

        trainer = cls(
            policy_model=policy_model,
            reward_model=reward_model,
            value_head=value_head, 
            policy_opt_config=policy_opt_config, 
            value_opt_config=value_opt_config, 
            device=device,

            temperature=temperature,
            use_kl=use_kl,
            clip_epsilon=clip_epsilon, 
            max_kl=max_kl, 
            valueL_coef=valueL_coef, 
            klL_coef=klL_coef, 
            entropyL_coef=entropyL_coef, 
            max_grad_norm=max_grad_norm, 
        )
        return trainer
    











