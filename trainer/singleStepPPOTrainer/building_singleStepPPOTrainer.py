"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2505061628
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

        self.value_clip = 0.2
        self.vf_coef = 0.3

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

        hidden_size = self.policy.base_model.config.hidden_size
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
        use_prefix: bool,
        print_response: bool = False,
    ):
        messages_token_len = int(messages_ids.shape[1]) - 1 + int(self.policy.prefix_ids.shape[1]) # -[temp]
        max_new_tokens = max(self.max_token_len - messages_token_len, 0)
        if print_response:
            highlight_show('got_context', self.policy.tokenizer.decode(messages_ids[:, self.policy.prefix_check_tensor_len:].tolist()[0], skip_special_tokens=False))

        policy_response, policy_generated_ids = self.policy.generate_response(
            messages_ids,
            max_new_tokens=max_new_tokens,
            use_prefix=use_prefix,
            temperature=self.temperature
        )
        # log_print(self.state_name, f"[{highlight()}] max_token_len: {messages_token_len} / {max_new_tokens}")

        if print_response:
            highlight_show('policy_response', policy_response)

        return policy_response, policy_generated_ids, max_new_tokens
    
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
        return reward

    def sample_init(self,
        context: str,
        messages: List[Dict[str, str]], 
        valid: bool = False, 
        output_response: bool = False, 
    ):
        messages_ids = self.policy.chat_template_tokenizer(
            chat_dict=messages, 
        )

        policy_response, policy_response_ids, max_new_tokens = self.get_response(
            messages_ids=messages_ids, 
            use_prefix=True,
            print_response=True,
        )
        policy_response_logits, policy_hidden_states, policy_seq_old_logp, entropy = self.get_full_forward(
            messages_ids=messages_ids, 
            response_ids=policy_response_ids, 
            use_prefix=True, 
            valid=valid, 
        )

        policy_rewards = self.compute_reward(
            context=context, 
            response=policy_response
        )
        log_print('sample_init', f"[{highlight()}] [policy_rewards] {policy_rewards} / {type(policy_rewards)}")
        # policy_rewards = (policy_rewards - policy_rewards.mean()) / (policy_rewards.std()+1e-8)      # [B]
        # log_print('sample_init', f"[{highlight()}] [policy_rewards] {policy_rewards} / {type(policy_rewards)}")

        policy_values  = self.value_head(policy_hidden_states)
        seq_old_values = policy_values[:, -1].detach()  # [B]
        log_print('sample_init', f"[{highlight()}] [seq_old_values] {seq_old_values} / {type(seq_old_values)}")

        advantages = policy_rewards - seq_old_values  # [B]
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        log_print('sample_init', f"[{highlight()}] [advantages] {advantages} / {type(advantages)}")
        

        reference_response, _, _ = self.get_response(
            messages_ids=messages_ids, 
            use_prefix=False,
            print_response=True,
        )
        reference_rewards = self.compute_reward(
            context=context, 
            response=reference_response
        )
        # reference_rewards = (reference_rewards - reference_rewards.mean()) / (reference_rewards.std()+1e-8)      # [B]
        log_print('sample_init', f"[{highlight()}] [reference_rewards] {reference_rewards} / {type(reference_rewards)}")
        print()

        if valid:
            sample_results = {
                'policy_seq_old_logp': policy_seq_old_logp,
                'advantages': advantages,
                'seq_old_values': seq_old_values,
                'policy_rewards': policy_rewards,
                'reference_rewards': reference_rewards,

                'policy_response': policy_response,
                'reference_response': reference_response,
            }
        else:
            sample_results = {
                'messages_ids': messages_ids,
                'policy_response_ids': policy_response_ids,

                'policy_seq_old_logp': policy_seq_old_logp,
                'advantages': advantages,
                'seq_old_values': seq_old_values,
                'policy_rewards': policy_rewards,
                'reference_rewards': reference_rewards,
            }
        # return full_ids, full_mask, L_g, gen, seq_old_logp, advantages, seq_values, rewards
        return sample_results

    def compute_policy_loss(self,
        sample_results: Dict[str, torch.Tensor], 
        valid: bool = False,
        output_response: bool = False, 
    ):
        # TODO to device 
        # policy_old_log_prob = policy_old_log_prob.to(self.device)
        # add value_clip

        policy_response_logits, policy_hidden_states, policy_seq_new_logp, seq_entropy = self.get_full_forward(
            messages_ids=sample_results['messages_ids'], 
            response_ids=sample_results['policy_response_ids'], 
            use_prefix=True, 
            valid=valid, 
        )
        entropy_loss = - self.entropy_coef * seq_entropy
        log_print('compute_policy_loss', f"[{highlight()}] [entropy_loss] {entropy_loss} / {type(entropy_loss)}")

        seq_ratio = torch.exp(policy_seq_new_logp - sample_results['policy_seq_old_logp'])  # [B]

        surr1 = seq_ratio * sample_results['advantages']
        surr2 = torch.clamp(
            seq_ratio, 
            1.0 - self.clip_epsilon, 
            1.0 + self.clip_epsilon
        ) * sample_results['advantages']
        pg_loss = -torch.min(surr1, surr2).mean()
        log_print('compute_policy_loss', f"[{highlight()}] [pg_loss] {pg_loss} / {type(pg_loss)}")


        # policy_hidden_states = policy_hidden_states[-1][:, -1]  # [B, D]
        seq_new_values = self.value_head.net(policy_hidden_states)  # [B, 1]
        seq_new_values = seq_new_values.squeeze(-1)  # [B]

        seq_values_clipped = sample_results['seq_old_values'] + torch.clamp(
            seq_new_values - sample_results['seq_old_values'], 
            -self.value_clip, 
            self.value_clip
        )
        vf_loss1 = (seq_new_values - sample_results['policy_rewards']) ** 2
        vf_loss2 = (seq_values_clipped - sample_results['policy_rewards']) ** 2
        vf_loss = self.vf_coef * torch.max(vf_loss1, vf_loss2).mean()
        log_print('compute_policy_loss', f"[{highlight()}] [vf_loss] {vf_loss} / {type(vf_loss)}")


        reference_response_logits, _, reference_seq_new_logp, _ = self.get_full_forward(
            messages_ids=sample_results['messages_ids'], 
            response_ids=sample_results['policy_response_ids'],
            use_prefix=False,
            valid=valid, 
        )
        
        seq_kl = policy_seq_new_logp - reference_seq_new_logp  # [B]
        kl_loss = seq_kl.mean()
        kl_loss = self.kl_coef * kl_loss
        log_print('compute_policy_loss', f"[{highlight()}] [kl_loss] {kl_loss} / {type(kl_loss)}")

        total_loss = pg_loss + vf_loss + kl_loss + entropy_loss
        log_print('compute_policy_loss', f"[{highlight()}] [total_loss] {total_loss} / {type(total_loss)}")
        print()

        metrics = {
            'pg_loss': pg_loss.item(),
            'vf_loss': vf_loss.item(),
            'kl_loss': kl_loss.item(),
            'entropy_loss': seq_entropy.item(),
            'total_loss': total_loss.item(),
            'policy_seq_old_logp': sample_results['policy_seq_old_logp'],
            'advantages': sample_results['advantages'],
            'seq_old_values': sample_results['seq_old_values'],
            'policy_rewards': sample_results['policy_rewards']
        }
        # metrics = self.update_metrics(
        #     max_new_tokens=max_new_tokens,
        #     policy_reward=policy_reward,
        #     reference_reward=reference_reward,
        #     policy_old_log_prob=policy_old_log_prob.item(),
        #     policy_new_log_prob=policy_new_log_prob.item(),
        #     ratio=ratio.item(),
        #     avg_kl=avg_kl.item(),
        #     policy_loss=policy_loss.item(),
        #     kl_loss=kl_loss.item(),
        #     entropy_loss=entropy_loss.item(),
        #     total_loss=total_loss.item(),
        #     messages_ids=messages_ids,
        #     policy_response=policy_response,
        #     reference_response=reference_response,
        #     output_response=output_response,
        # )

        return total_loss, metrics
    
    def update_metrics(self,
        max_new_tokens: int,
        policy_reward: float,
        reference_reward: float,
        policy_old_log_prob: float,
        policy_new_log_prob: float,
        ratio: float,
        avg_kl: float,
        policy_loss: float,
        kl_loss: float,
        entropy_loss: float,
        total_loss: float,
        messages_ids: torch.Tensor,
        policy_response: str,
        reference_response: str,
        output_response: bool = False,
    ):
        self.training_stats['steps'] += 1
        metrics = {
            'step': self.training_stats['steps'],
            'lr': self.scheduler.get_last_lr()[0],
            'max_new_tokens': max_new_tokens,

            'policy_reward': policy_reward,
            'reference_reward': reference_reward,

            'old_log_prob': policy_old_log_prob,
            'new_log_prob': policy_new_log_prob,
            'ratio': ratio,
            'step_avg_kl': avg_kl,

            'policy_loss': policy_loss,
            'kl_loss': kl_loss,
            'entropy_loss': entropy_loss,
            'total_loss': total_loss,
        }
        self.training_stats['total_policy_reward'] += policy_reward
        self.training_stats['total_reference_reward'] += reference_reward
        self.training_stats['total_step_kl'] += avg_kl

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

        return metrics

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
    











