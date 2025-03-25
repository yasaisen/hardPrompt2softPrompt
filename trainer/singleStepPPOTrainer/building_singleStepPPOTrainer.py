"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2503252044
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict
import os

from ...common.utils import log_print, get_trainable_params
from ...models.rewardModel.modeling_rewardModel import ComparativeRewardModel
from ...models.policyModel.modeling_policyModel import PrefixTuningPolicyModel


class SingleStepPPOTrainer:
    def __init__(
        self,
        policy_model: PrefixTuningPolicyModel,
        reward_model: ComparativeRewardModel,
        device: str = "cuda",
        # PPO超參數
        learning_rate: float = 1e-5,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.1,  # KL散度係數
        max_grad_norm: float = 1.0,
        max_kl: float = 0.2,  # KL散度的最大閾值
    ):
        self.state_name = 'SingleStepPPOTrainer'
        self.device = device
        print()
        log_print(self.state_name, f"Building...")

        self.policy = policy_model.to(self.device)
        self.reward_model = reward_model.to(self.device)
        
        # 只優化policy model中的prefix參數
        self.optimizer = optim.Adam([self.policy.prefix_embeddings], lr=learning_rate)
        
        # 保存超參數
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.max_kl = max_kl
        self.max_grad_norm = max_grad_norm
        
        # 追蹤訓練統計
        self.training_stats = {
            'steps': 0,
            'total_reward': 0,
            'avg_reward': 0,
            'avg_kl': 0,
            'total_kl': 0
        }
        log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self)}")
        self.to(self.device) # 
        log_print(self.state_name, f"...Done\n")

    def get_response(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 50,
        temperature: float = 1.0
    ) -> Tuple[str, float]:
        messages_ids = self.policy.truncate_from_beginning(messages)
        print('================================================got context')
        print(self.policy.tokenizer.decode(messages_ids[:, 7:].tolist()[0], skip_special_tokens=False))
        print('================================================got context')

        # reference_response, reference_log_prob, reference_probs = self.policy.generate_response(
        #     messages_ids,
        #     max_new_tokens=max_new_tokens,
        #     use_prefix=False,
        #     temperature=temperature
        # )

        policy_response, policy_log_prob, policy_probs = self.policy.generate_response(
            messages_ids,
            max_new_tokens=max_new_tokens,
            use_prefix=True,
            temperature=temperature
        )

        # generate_ids = self.policy.base_model.generate(
        #     messages_ids, 
        #     max_new_tokens=max_new_tokens,
        #     temperature=temperature
        # )
        # generate_response = self.policy.tokenizer.decode(generate_ids.tolist()[0][len(messages_ids.tolist()[0]):], skip_special_tokens=False)
        reference_response, reference_log_prob, reference_probs = None, None, None

        # print('================================================generate full')
        # print(generate_response)
        # print('================================================reference_response')
        # print(reference_response)
        # print(reference_log_prob)
        print('================================================policy_response')
        print(policy_response)
        print(policy_log_prob)
        print('================================================end')

        return policy_response, policy_log_prob, policy_probs, reference_response, reference_log_prob, reference_probs

    def compute_reward(self, context: str, response: str) -> float:
        context_ids = self.reward_model.truncate_from_beginning(context)
        response_ids = self.reward_model.truncate_from_beginning(response)
        
        context_mask = torch.ones_like(context_ids)
        response_mask = torch.ones_like(response_ids)
        
        with torch.no_grad():
            reward = self.reward_model.get_reward(
                context_ids,
                context_mask,
                response_ids,
                response_mask
            )
        
        return reward.item()

    def get_base_model_probs(self, context_ids: torch.Tensor) -> List[torch.Tensor]:
        """獲取base model在每個時間步的概率分布"""
        with torch.no_grad():
            outputs = self.policy.base_model.generate(
                context_ids,
                max_new_tokens=50,  # 可以根據需要調整
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.policy.tokenizer.pad_token_id,
                eos_token_id=self.policy.tokenizer.eos_token_id,
            )
            
            # 將scores轉換為概率分布
            base_probs = []
            for score in outputs.scores:
                probs = F.softmax(score, dim=-1)
                base_probs.append(probs)
                
        return base_probs, outputs.sequences

    def compute_stepwise_kl(
        self,
        policy_logits: torch.Tensor,
        base_probs: torch.Tensor,
    ) -> torch.Tensor:
        """計算每個時間步的KL散度"""
        policy_probs = F.softmax(policy_logits, dim=-1)
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        
        kl_div = (policy_probs * (policy_log_probs - torch.log(base_probs + 1e-10))).sum(-1)
        return kl_div

    def compute_policy_loss(
        self,
        messages,
        response,
        base_probs_,
        old_log_prob: float,
        reward: float
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:

        messages_ids = self.policy.truncate_from_beginning(messages)
        response_ids = self.policy.truncate_from_beginning(response, only_str=True)
        response_logits, new_log_prob, entropy = self.policy.full_forward(messages_ids, response_ids)


        ##############################################################
        # 獲取base model的概率分布
        base_probs, _ = self.get_base_model_probs(messages_ids)
        # print('@@@@@@@@@@@@@@@@', base_probs)
        # print('@@@@@@@@@@@@@@@@', base_probs_)
        ##############################################################
        
        print('================================================')
        embeddings = self.policy.base_model.lm_head(self.policy.prefix_embeddings)
        print(embeddings.shape)
        next_token_id = torch.argmax(embeddings, dim=-1).to(torch.long)
        response = self.policy.tokenizer.decode(next_token_id, skip_special_tokens=True)
        print(response)
        # point_loss = torch.tensor(float(input('input(-3 - +3):')) / -6, device=self.device)
        # print(point_loss)
        print('================================================')

        # 計算與base model的KL散度
        total_kl = torch.tensor(0.0, device=self.device)
        for step_idx, base_prob in enumerate(base_probs):
            if step_idx < response_logits.size(1):  # 確保不超過response長度
                step_kl = self.compute_stepwise_kl(
                    response_logits[:, step_idx], 
                    base_prob
                )
                total_kl += step_kl.mean()
        avg_kl = total_kl / len(base_probs)
        
        # 如果KL散度太大，增加懲罰
        kl_loss = self.kl_coef * torch.max(
            avg_kl - self.max_kl,
            torch.tensor(0.0, device=self.device)
        )


        ### calculate PPO ###
        ratio = torch.exp(new_log_prob - old_log_prob)
        reward_tensor = torch.tensor(reward, device=self.device)

        surr1 = ratio * reward_tensor
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * reward_tensor
        policy_loss = -torch.min(surr1, surr2)

        # 計算熵損失
        entropy_loss = -self.entropy_coef * entropy
        
        # 總損失（加入KL損失）
        total_loss = policy_loss + entropy_loss + kl_loss # + point_loss
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'kl_loss': kl_loss.item(),
            'avg_kl': avg_kl.item(),
            'total_loss': total_loss.item(),
            'ratio': ratio.item(),
            'reward': reward
        }
        
        return total_loss, new_log_prob, metrics

    def step(
        self,
        context: str,
        messages: List[Dict[str, str]],
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """執行單個PPO步驟"""
        # 1. 生成回應
        policy_response, old_log_prob, policy_probs, reference_response, reference_log_prob, reference_probs = self.get_response(messages=messages, temperature=temperature)

        # 2. 計算獎勵
        reward = self.compute_reward(context, policy_response)
        print('policy_reward', reward)
        # print('reference_reward', self.compute_reward(context, reference_response))
        
        # 3. 計算損失
        loss, new_log_prob, metrics = self.compute_policy_loss(
            messages, 
            policy_response, 
            reference_probs, 
            old_log_prob, 
            reward
        )
        self.policy.train()
        
        # 4. 優化步驟
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.prefix_embeddings, self.max_grad_norm)
        self.optimizer.step()
        
        # 5. 更新統計
        self.training_stats['steps'] += 1
        self.training_stats['total_reward'] += reward
        self.training_stats['avg_reward'] = self.training_stats['total_reward'] / self.training_stats['steps']
        self.training_stats['total_kl'] += metrics['avg_kl']
        self.training_stats['avg_kl'] = self.training_stats['total_kl'] / self.training_stats['steps']
        
        # 6. 返回此步驟的完整信息
        step_info = {
            # 'response': policy_response,
            'reward': reward,
            'old_log_prob': old_log_prob,
            'new_log_prob': new_log_prob.item(),
            **metrics,
            **self.training_stats
        }
        
        return step_info
    
    @classmethod
    def from_config(cls, 
        cfg, 
        policy_model: PrefixTuningPolicyModel,
        reward_model: ComparativeRewardModel,
    ):
        device = str(cfg['task'].get("device"))

        trainer_cfg = cfg['task']
        learning_rate = float(trainer_cfg.get("learning_rate"))
        clip_epsilon = float(trainer_cfg.get("clip_epsilon"))
        entropy_coef = float(trainer_cfg.get("entropy_coef"))
        kl_coef = float(trainer_cfg.get("kl_coef"))
        max_grad_norm = float(trainer_cfg.get("max_grad_norm"))
        max_kl = float(trainer_cfg.get("max_kl"))

        model = cls(
            policy_model=policy_model,
            reward_model=reward_model,
            learning_rate=learning_rate,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            kl_coef=kl_coef,
            max_grad_norm=max_grad_norm,
            max_kl=max_kl,
            device=device,
        )
        return model










