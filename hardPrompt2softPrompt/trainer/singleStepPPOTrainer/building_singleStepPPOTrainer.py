"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2508031527
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict

from ...common.utils import log_print, highlight_show, highlight
from ...models.rewardModel.modeling_rewardModel import ComparativeRewardModel
from ...models.policyModel.modeling_policyModel import PrefixTuningPolicyModel
from ...models.valueHead.modeling_valueHead import ValueHead


class SingleStepPPOTrainer:
    def __init__(self,
        policy_model: PrefixTuningPolicyModel,
        reward_model: ComparativeRewardModel,
        device: str = "cuda",
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.1, 
        max_kl: float = 0.2, 
        value_clip = 0.2,
        vf_coef = 0.3,
        max_grad_norm: float = 1.0,
        max_token_len: int = 50,
        temperature: float = 1.0,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4, 
        max_lr: float = 1e-3, 
        num_epoch: int = 30, 
        steps_per_epoch: int = 30, 
        pct_start: float = 0.2, 
        anneal_strategy: str = 'cos', 
    ):
        self.state_name = 'SingleStepPPOTrainer'
        self.device = device
        print()
        log_print(self.state_name, f"Building...")

        self.policy = policy_model.to(self.device)
        self.reward_model = reward_model.to(self.device)

        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.max_kl = max_kl
        self.value_clip = value_clip
        self.vf_coef = vf_coef

        self.max_grad_norm = max_grad_norm
        self.max_token_len = max_token_len
        self.temperature = temperature

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_lr = max_lr
        self.num_epoch = num_epoch
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        
        self.optimizer_policy = optim.AdamW(
            [self.policy.prefix_embeddings],
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler_policy = optim.lr_scheduler.OneCycleLR(
            self.optimizer_policy,
            max_lr=self.max_lr,
            epochs=self.num_epoch,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy
        )

        hidden_size = self.policy.hidden_size
        self.value_head = ValueHead(hidden_size).to(self.device)

        self.optimizer_value = optim.AdamW(
            self.value_head.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler_value = optim.lr_scheduler.OneCycleLR(
            self.optimizer_value,
            max_lr=self.max_lr,
            epochs=self.num_epoch,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy
        )

        self.training_stats = {
            'steps': 0,
            'total_policy_reward': 0,
            'total_reference_reward': 0,
            'total_step_kl': 0,
            'avg_policy_reward': 0,
            'avg_reference_reward': 0,
            'avg_step_kl': 0
        }
        log_print(self.state_name, f"...Done\n")

    def get_response(self,
        messages_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        use_prefix: bool,
        print_response: bool = False,
    ):
        if print_response:
            highlight_show('got_context', self.policy.tokenizer.decode(messages_ids[:, self.policy.prefix_check_tensor_len:].tolist()[0], skip_special_tokens=False))

        policy_response, policy_generated_ids = self.policy.generate_response_with_batch(
            input_ids=messages_ids,
            attention_mask=attention_mask,
            use_prefix=use_prefix,
            temperature=self.temperature
        )
        # log_print(self.state_name, f"[{highlight()}] max_token_len: {messages_token_len} / {max_new_tokens}")

        if print_response:
            highlight_show('policy_response', policy_response)

        return policy_response, policy_generated_ids
    
    def get_full_forward(self,
        messages_ids: torch.Tensor, 
        response_ids: torch.Tensor, 
        use_prefix: bool, 
        valid: bool, 
    ):
        if not valid:
            self.policy.train()
            response_logits, hidden_states, seq_old_logp, entropy = self.policy.full_forward(
                messages_ids=messages_ids, 
                response_ids=response_ids,
                use_prefix=use_prefix,
                temperature=self.temperature,
            )
        else:
            self.policy.eval()
            with torch.no_grad():
                response_logits, hidden_states, seq_old_logp, entropy = self.policy.full_forward(
                    messages_ids=messages_ids, 
                    response_ids=response_ids,
                    use_prefix=use_prefix,
                    temperature=self.temperature,
                )

        return response_logits, hidden_states, seq_old_logp, entropy
        
    def get_seq_values(self,
        hidden_states: torch.Tensor, 
        valid: bool, 
    ):
        hidden_for_value = hidden_states.detach()
        if not valid:
            self.value_head.train()
            policy_values  = self.value_head(hidden_for_value)
            seq_old_values = policy_values[:, -1]  # [B]
        else:
            self.value_head.eval()
            with torch.no_grad():
                policy_values  = self.value_head(hidden_for_value)
                seq_old_values = policy_values[:, -1].detach()  # [B]

        return seq_old_values

    def compute_stepwise_kl(self, 
        policy_logits: torch.Tensor, 
        reference_logits: torch.Tensor
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

    def compute_reward(self, 
        context: str, 
        response: str
    ) -> float:
        context_ids = self.reward_model.truncate_from_beginning(context)
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
            )
        # return reward.item()
        return reward.detach()

    def compute_batch_reward(self, 
        batch_context, 
        batch_response, 
    ) -> List[float]:
        
        reward_list = []
        for context, response in zip(batch_context, batch_response):
            reward_list += [
                self.compute_reward(
                    context=str(context), 
                    response=response
                )
            ]
        return torch.tensor(reward_list).to(self.device)

    def sample_init(self,
        context: str,
        messages: List[Dict[str, str]], 
        valid: bool = False, 
        output_response: bool = False, 
    ):
        batched_messages_ids, batched_messages_attnmask = self.policy.chat_template_tokenizer(
            chat_dicts=messages, 
            max_length=self.max_token_len,
        )
        policy_response, policy_response_ids = self.get_response(
            messages_ids=batched_messages_ids, 
            attention_mask=batched_messages_attnmask,
            use_prefix=True, 
            print_response=False, 
        )
        policy_response_logits, policy_hidden_states, policy_seq_old_logp, entropy = self.get_full_forward(
            messages_ids=batched_messages_ids, 
            response_ids=policy_response_ids, 
            use_prefix=True, 
            valid=valid, 
        )
        reference_response_logits, _, reference_seq_new_logp, _ = self.get_full_forward(
            messages_ids=batched_messages_ids, 
            response_ids=policy_response_ids,
            use_prefix=False,
            valid=valid, 
        )
        policy_rewards = self.compute_batch_reward(
            batch_context=context,
            batch_response=policy_response,
        )
        policy_rewards = (policy_rewards - policy_rewards.mean()) / (policy_rewards.std()+1e-8)      # [B]

        seq_old_values = self.get_seq_values(
            hidden_states=policy_hidden_states,
            valid=True
        )

        raw_advantages = policy_rewards - seq_old_values  # [B]
        advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)
        returns = raw_advantages + seq_old_values

        reference_response, _ = self.get_response(
            messages_ids=batched_messages_ids, 
            attention_mask=batched_messages_attnmask, 
            use_prefix=False, 
            print_response=False, 
        )
        reference_rewards = self.compute_batch_reward(
            batch_context=context, 
            batch_response=reference_response
        )
        reference_rewards = (reference_rewards - reference_rewards.mean()) / (reference_rewards.std()+1e-8)      # [B]

        print()
        log_print('sample_init', f"[{highlight('policy_rewards')}] {policy_rewards}")
        log_print('sample_init', f"[{highlight('reference_rewards')}] {reference_rewards}")
        diff_reward = policy_rewards - reference_rewards
        diff_reward = diff_reward.mean()
        log_print('sample_init', f"[{highlight('diff_reward')}] {diff_reward}")
        print()

        # raise 'meow'
        if not valid:
            sample_results = {
                'messages_ids': batched_messages_ids,
                'policy_response_ids': policy_response_ids,

                'policy_seq_old_logp': policy_seq_old_logp,
                'reference_seq_new_logp': reference_seq_new_logp,
                'reference_response_logits': reference_response_logits,
                'advantages': advantages,
                'returns': returns,
                'seq_old_values': seq_old_values,
                'policy_rewards': policy_rewards,
                'reference_rewards': reference_rewards,
            }
        else:
            sample_results = None       
        metrics = {
            'state': 'valid' if valid else 'warmUp',

            'policy_seq_old_logp': policy_seq_old_logp, # .item(),
            'reference_seq_new_logp': reference_seq_new_logp, # .item(),
            'advantages': advantages, # .item(),
            'returns': returns, # .item(),
            'seq_old_values': seq_old_values, # .item(),
            'policy_rewards': policy_rewards, # .item(),
            'reference_rewards': reference_rewards, # .item(),
            'diff_reward': diff_reward.item(), 

            'context_messages': str(messages),
            'policy_response': str(policy_response),
            'reference_response': str(reference_response),
        }

        for key in metrics:
            if key not in ['context_messages', 'policy_response', 'reference_response']:
                log_print('sample_init', f"[{highlight(key)}] {metrics[key]}")
                if isinstance(metrics[key], torch.Tensor):
                    if torch.isnan(metrics[key]).any() or torch.isinf(metrics[key]).any():
                        raise f"{key} / {metrics[key]}"
        print()

        return sample_results, metrics

    def compute_policy_loss(self,
        sample_results: Dict[str, torch.Tensor], 
        valid: bool = False,
        output_response: bool = False, 
    ):

        policy_response_logits, policy_hidden_states, policy_seq_new_logp, seq_entropy = self.get_full_forward(
            messages_ids=sample_results['messages_ids'], 
            response_ids=sample_results['policy_response_ids'], 
            use_prefix=True, 
            valid=valid, 
        )

        entropy_loss = - self.entropy_coef * seq_entropy


        log_ratio = policy_seq_new_logp - sample_results['policy_seq_old_logp']
        seq_ratio = torch.exp(log_ratio)  # [B]
        approx_kl = ((seq_ratio - 1) - log_ratio).mean()
        clip_frac = ((seq_ratio - 1.0).abs() > self.clip_epsilon).float().mean()

        pg_surr1 = seq_ratio * sample_results['advantages']
        pg_surr2 = torch.clamp(
            seq_ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon
        ) * sample_results['advantages']
        pg_loss = -torch.min(pg_surr1, pg_surr2).mean()

        # policy_hidden_states = policy_hidden_states[-1][:, -1]  # [B, D]
        seq_new_values = self.get_seq_values(
            hidden_states=policy_hidden_states, 
            valid=valid
        )

        seq_values_clipped = sample_results['seq_old_values'] + torch.clamp(
            seq_new_values - sample_results['seq_old_values'],
            -self.value_clip,
            self.value_clip
        )
        vf_surr1 = (seq_new_values - sample_results['returns']) ** 2
        vf_surr2 = (seq_values_clipped - sample_results['returns']) ** 2
        vf_loss = self.vf_coef * torch.max(vf_surr1, vf_surr2).mean()

        seq_kl = policy_seq_new_logp - sample_results['reference_seq_new_logp']  # [B]
        kl_loss = seq_kl.mean()
        kl_loss = self.kl_coef * kl_loss

        total_kl = torch.tensor(0.0, device=self.device)
        for step_idx in range(sample_results['policy_response_ids'].shape[1]):
            step_kl = self.compute_stepwise_kl(
                policy_logits=policy_response_logits[:, step_idx], 
                reference_logits=sample_results['reference_response_logits'][:, step_idx],
            )
            total_kl += step_kl.mean()
        avg_kl = total_kl / int(sample_results['policy_response_ids'].shape[1])
        kl_loss = self.kl_coef * torch.max(
            avg_kl - self.max_kl, 
            torch.tensor(0.0, device=self.device)
        )

        total_loss = pg_loss + vf_loss + kl_loss + entropy_loss

        metrics = {
            'state': 'valid' if valid else 'train',

            'policy_seq_old_logp': sample_results['policy_seq_old_logp'].tolist(),
            'policy_seq_new_logp': policy_seq_new_logp.tolist(),
            'seq_ratio': seq_ratio.tolist(),
            'approx_kl': approx_kl.item(),
            'clip_frac': clip_frac.item(),
            'advantages': sample_results['advantages'].tolist(),
            'returns': sample_results['returns'].tolist(),
            'pg_surr1': pg_surr1.tolist(),
            'pg_surr2': pg_surr2.tolist(),
            'pg_loss': pg_loss.item(),

            'seq_old_values': sample_results['seq_old_values'].tolist(),
            'seq_new_values': seq_new_values.tolist(),
            'seq_values_clipped': seq_values_clipped.tolist(),
            'policy_rewards': sample_results['policy_rewards'].tolist(),
            'vf_surr1': vf_surr1.tolist(),
            'vf_surr2': vf_surr2.tolist(),
            'vf_loss': vf_loss.item(),

            'kl_loss': kl_loss.item(),
            'entropy': seq_entropy.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
        }

        for key in metrics:
            log_print('compute_policy_loss', f"[{highlight(key)}] {metrics[key]}")
            if isinstance(metrics[key], torch.Tensor):
                if torch.isnan(metrics[key]).any() or torch.isinf(metrics[key]).any():
                    raise f"{key} / {metrics[key]}"
        print()

        return total_loss, metrics
    
    def update_metrics(self,
    ):
        raise NotImplementedError

    @classmethod
    def from_config(cls, 
        cfg, 
        policy_model: PrefixTuningPolicyModel,
        reward_model: ComparativeRewardModel,
        steps_per_epoch: int,
    ):
        device = str(cfg['task'].get("device"))

        trainer_cfg = cfg['task']
        clip_epsilon = float(trainer_cfg.get("clip_epsilon"))
        entropy_coef = float(trainer_cfg.get("entropy_coef"))
        kl_coef = float(trainer_cfg.get("kl_coef"))
        max_grad_norm = float(trainer_cfg.get("max_grad_norm"))
        max_kl = float(trainer_cfg.get("max_kl"))
        value_clip = float(trainer_cfg.get("value_clip"))
        vf_coef = float(trainer_cfg.get("vf_coef"))
        max_token_len = int(trainer_cfg.get("max_token_len"))
        temperature = float(trainer_cfg.get("temperature"))

        learning_rate = float(trainer_cfg.get("learning_rate"))
        weight_decay = float(trainer_cfg.get("weight_decay"))
        max_lr = float(trainer_cfg.get("max_lr"))
        num_epoch = int(trainer_cfg.get("num_epoch"))
        pct_start = float(trainer_cfg.get("pct_start"))
        anneal_strategy = str(trainer_cfg.get("anneal_strategy"))

        trainer = cls(
            policy_model=policy_model,
            reward_model=reward_model,

            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            kl_coef=kl_coef,
            max_grad_norm=max_grad_norm,
            max_kl=max_kl,
            value_clip=value_clip,
            vf_coef=vf_coef,
            max_token_len=max_token_len,
            temperature=temperature,

            learning_rate=learning_rate,
            weight_decay=weight_decay, 
            max_lr=max_lr, 
            num_epoch=num_epoch, 
            steps_per_epoch=steps_per_epoch, 
            pct_start=pct_start, 
            anneal_strategy=anneal_strategy, 

            device=device,
        )
        return trainer
    











