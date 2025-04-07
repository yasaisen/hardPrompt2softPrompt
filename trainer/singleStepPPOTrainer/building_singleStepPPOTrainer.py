"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2504030040
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict

from ...common.utils import log_print, highlight_show, highlight
from ...models.rewardModel.modeling_rewardModel import ComparativeRewardModel
from ...models.policyModel.modeling_policyModel import PrefixTuningPolicyModel


class SingleStepPPOTrainer:
    def __init__(self,
        policy_model: PrefixTuningPolicyModel,
        reward_model: ComparativeRewardModel,
        device: str = "cuda",
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.1, 
        max_grad_norm: float = 1.0,
        max_kl: float = 0.2, 
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
        
        self.optimizer = optim.AdamW(
            [self.policy.prefix_embeddings],
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
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
        output_probs: bool = False,
        print_response: bool = False,
    ):
        messages_token_len = int(messages_ids.shape[1]) - 7 + 6 + int(self.policy.prefix_ids.shape[1])
        max_new_tokens = max(self.max_token_len - messages_token_len, 0)
        if print_response:
            highlight_show('got_context', self.policy.tokenizer.decode(messages_ids[:, 7:].tolist()[0], skip_special_tokens=False))

        reference_response, reference_log_prob, reference_probs = self.policy.generate_response(
            messages_ids,
            max_new_tokens=max_new_tokens,
            use_prefix=False,
            temperature=self.temperature
        )
        # log_print(self.state_name, f"[{highlight()}] max_token_len: {messages_token_len} / {max_new_tokens}")

        policy_response, policy_log_prob, policy_probs = self.policy.generate_response(
            messages_ids,
            max_new_tokens=max_new_tokens,
            use_prefix=True,
            temperature=self.temperature
        )
        # log_print(self.state_name, f"[{highlight()}] max_token_len: {messages_token_len} / {max_new_tokens}")

        if print_response:
            highlight_show('reference_response', reference_response)
            highlight_show('policy_response', policy_response)
            
        if output_probs:
            return policy_response, policy_log_prob, policy_probs, reference_response, reference_log_prob, reference_probs, max_new_tokens

        return policy_response, policy_log_prob, reference_response, reference_log_prob, max_new_tokens

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
        return reward.item()

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

    def compute_policy_loss(self,
        context: str,
        messages: List[Dict[str, str]], 
        valid: bool = False, 
        output_response: bool = False, 
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        messages_ids = self.policy.chat_template_tokenizer(
            chat_dict=messages, 
            is_response=False
        )
        policy_response, policy_old_log_prob, reference_response, _, max_new_tokens = self.get_response(
            messages_ids=messages_ids, 
            output_probs=False,
            print_response=False,
        )

        policy_reward = self.compute_reward(
            context=context, 
            response=policy_response
        )
        reference_reward = self.compute_reward(
            context=context, 
            response=reference_response
        )

        response_ids = self.policy.chat_template_tokenizer(
            chat_dict=policy_response, 
            is_response=True
        )

        self.policy.eval()
        with torch.no_grad():
            reference_response_logits, _, _ = self.policy.full_forward(
                messages_ids=messages_ids, 
                response_ids=response_ids, 
                use_prefix=False,
                temperature=self.temperature,
            )
        if valid:
            self.policy.eval()
            with torch.no_grad():
                policy_response_logits, policy_new_log_prob, entropy = self.policy.full_forward(
                    messages_ids=messages_ids, 
                    response_ids=response_ids, 
                    use_prefix=True,
                    temperature=self.temperature,
                )
        else:
            self.policy.train()
            policy_response_logits, policy_new_log_prob, entropy = self.policy.full_forward(
                messages_ids=messages_ids, 
                response_ids=response_ids, 
                use_prefix=True,
                temperature=self.temperature,
            )

        total_kl = torch.tensor(0.0, device=self.device)
        for step_idx in range(response_ids.shape[1]):
            step_kl = self.compute_stepwise_kl(
                policy_logits=policy_response_logits[:, step_idx], 
                reference_logits=reference_response_logits[:, step_idx],
            )
            total_kl += step_kl.mean()
        avg_kl = total_kl / int(response_ids.shape[1])

        kl_loss = self.kl_coef * torch.max(
            avg_kl - self.max_kl,
            torch.tensor(0.0, device=self.device)
        )

        ### calculate PPO ###
        ratio = torch.exp(policy_new_log_prob - policy_old_log_prob)
        reward_tensor = torch.tensor(policy_reward, device=self.device)

        surr1 = ratio * reward_tensor
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * reward_tensor
        policy_loss = -torch.min(surr1, surr2)

        entropy_loss = -self.entropy_coef * entropy
        
        total_loss = policy_loss + entropy_loss + kl_loss

        self.training_stats['steps'] += 1
        metrics = {
            'step': self.training_stats['steps'],
            'max_new_tokens': max_new_tokens,

            'policy_reward': policy_reward,
            'reference_reward': reference_reward,

            'old_log_prob': policy_old_log_prob,
            'new_log_prob': policy_new_log_prob.item(),
            'ratio': ratio.item(),
            'step_avg_kl': avg_kl.item(),

            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
        }
        self.training_stats['total_policy_reward'] += policy_reward
        self.training_stats['total_reference_reward'] += reference_reward
        self.training_stats['total_step_kl'] += avg_kl.item()

        self.training_stats['avg_policy_reward'] = self.training_stats['total_policy_reward'] / self.training_stats['steps']
        self.training_stats['avg_reference_reward'] = self.training_stats['total_reference_reward'] / self.training_stats['steps']
        self.training_stats['avg_step_kl'] = self.training_stats['total_step_kl'] / self.training_stats['steps']
        metrics['avg_policy_reward'] = self.training_stats['avg_policy_reward']
        metrics['avg_reference_reward'] = self.training_stats['avg_reference_reward']
        metrics['avg_step_kl'] = self.training_stats['avg_step_kl']

        if output_response:
            metrics['context_messages'] = self.policy.tokenizer.decode(messages_ids[:, 7:].tolist()[0], skip_special_tokens=False)
            metrics['policy_response'] = policy_response
            metrics['reference_response'] = reference_response

        return total_loss, metrics

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
        max_token_len = int(trainer_cfg.get("max_token_len"))
        temperature = float(trainer_cfg.get("temperature"))

        learning_rate = float(trainer_cfg.get("learning_rate"))
        weight_decay = float(trainer_cfg.get("weight_decay"))
        max_lr = float(trainer_cfg.get("max_lr"))
        num_epoch = int(trainer_cfg.get("num_epoch"))
        pct_start = float(trainer_cfg.get("pct_start"))
        anneal_strategy = str(trainer_cfg.get("anneal_strategy"))

        model = cls(
            policy_model=policy_model,
            reward_model=reward_model,

            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            kl_coef=kl_coef,
            max_grad_norm=max_grad_norm,
            max_kl=max_kl,
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
        return model
    











