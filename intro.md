# HardPrompt2SoftPrompt

**Preference-Optimized Hard-to-Soft Prompt Adaptation with Prefix Tuning and Single-Step PPO**

**Keywords:** RLHF, Prefix Tuning, Soft Prompt, Single-Step PPO, Reward Model, Preference Learning

## Overview

In task-specific applications, a hand-crafted system prompt can often provide a reasonable behavioral baseline for a large language model, but its responses may still deviate from the preferences of real users or domain experts.

This project explores a parameter-efficient alternative to full-model fine-tuning: **convert an existing hard prompt into a trainable soft prefix and optimize only that prefix using preference feedback**.

The target application was a conversational data-collection setting for assessments involving individuals potentially affected by Alzheimer's disease. The language model was designed to imitate a human tester by maintaining the conversation, asking appropriate follow-up questions, and encouraging the participant to provide more speech data. Disease analysis or diagnosis itself was outside the scope of this project.

The central idea is:

> **Hard Prompt → Prompt Embeddings → Learnable Soft Prefix → Preference Optimization**

Instead of randomly initializing soft-prompt parameters, the original human-written prompt is used as a semantically meaningful initialization.

---

## Method

### 1. Preference Data and Reward Model

The original data consisted of Traditional Chinese conversations between real testers and participants during assessment sessions.

Conversation contexts were randomly truncated, and `google/gemma-3-1b-it` generated two candidate responses for each context. A real tester manually compared the two candidates using four labels:

* A is better than B
* B is better than A
* Both are poor
* Both are good

The last two categories were discarded because they did not provide a directional preference signal.

This process produced **500 Context–Better–Worse pairwise samples**.

The Reward Model was based on `bert-base-chinese`. The BERT backbone was frozen, while a **20-token learnable prefix**, context/response projection layers, and an MLP reward head were trained.

For a context (c) and response (y), the model predicts a scalar reward (r(c,y)). Training encourages:

$$r(c,y_{\text{better}}) > r(c,y_{\text{worse}})$$

using the reward difference with `BCEWithLogitsLoss`.

The dataset was randomly split into **80% training and 20% validation**.

---

### 2. Hard Prompt to Soft Prompt

The policy model used `google/gemma-3-1b-it`.

Instead of initializing prefix parameters randomly, the original tester system prompt was tokenized and its token embeddings were directly copied into trainable prefix embeddings.

The entire Gemma model was frozen, and **only the soft-prefix embeddings were updated**.

Conceptually:

```text
Human-written hard prompt
        ↓
Token embeddings
        ↓
Learnable soft prefix
        ↓
Preference optimization
```

This preserves the semantics of an already functional prompt while allowing continuous optimization in embedding space.

---

### 3. Single-Step PPO

Policy optimization used **245 historical human assessment conversations**. Conversations were randomly truncated to create training contexts and split using a **98%/2% train-validation ratio**.

In this project, **an entire generated response is treated as one action / decision step**, rather than treating individual generated tokens as separate reinforcement-learning steps. This is referred to as **Single-Step PPO**.

For each context:

1. The soft-prefix policy generates a complete response.
2. The Reward Model assigns a scalar reward.
3. A Value Head estimates the expected reward.
4. The reward and value estimate are used to compute the advantage.
5. A PPO clipped policy-gradient objective updates the soft prefix.

The optimization also included a value loss and entropy regularization.

A reference policy used the same frozen Gemma model with the **original hard prompt**, providing a baseline against which the optimized soft-prefix policy could be compared.

A KL-divergence constraint between the policy and reference distributions was also designed as a stability mechanism. However, due to implementation and training-stability issues, **KL regularization was disabled in the reported experiment**.

---

## Results

### Reward Model

On the single 80/20 train-validation split, the Reward Model achieved:

**90.00% validation pairwise ranking accuracy**

This indicates that the learned reward function could reasonably distinguish tester-preferred responses from less-preferred alternatives within the collected dataset.

---

### Soft-Prompt Policy

The PPO policy-gradient, value, and entropy objectives remained trainable throughout optimization, although the loss curves showed substantial oscillation. PPO loss itself was therefore **not interpreted as evidence of behavioral convergence**.

The held-out policy validation subset contained only **5 conversations**.

Among these five samples:

* the optimized soft-prefix policy received a higher Reward Model score than the original hard-prompt reference in **4 cases**;
* the reference policy received a higher score in **1 case**.

During training, the policy–reference reward difference was also positive for most recorded steps, suggesting that PPO successfully moved the soft prefix toward behaviors favored by the learned Reward Model.

---

## Discussion

The main observation of this project is that a human-written hard prompt can serve not only as an inference-time instruction, but also as a **semantic initialization for continuous prompt optimization**.

Rather than optimizing billions of LLM parameters, the method searches within a very small parameter space consisting only of the prompt embeddings. This makes the approach particularly suitable for tasks where prompting already provides a strong baseline and only moderate behavioral adaptation is required.

However, the results should be interpreted as preliminary.

First, the Reward Model was trained on only 500 pairwise comparisons, and the reported 90.00% accuracy was obtained from a single random validation split without an independent test set or repeated runs.

Second, the PPO validation subset contained only five samples, which is far too small for reliable statistical conclusions.

Most importantly, the **same Reward Model was used both as the PPO training signal and as the metric for comparing the optimized policy with the reference policy**. Therefore, the observed reward improvement demonstrates that the soft prompt successfully optimized the learned surrogate preference objective, but does not independently prove that human testers would prefer the resulting responses.

Potential reward over-optimization also cannot be excluded.

A stronger evaluation would require independent human preference testing, preferably with blinded comparison between responses generated by the original hard prompt and the optimized soft prompt.

---

## Conclusion

This project demonstrates a parameter-efficient pipeline for:

**Hard Prompt → Soft Prompt → Preference Optimization**

A human-written prompt is converted directly into trainable embeddings and optimized with a learned Reward Model and response-level Single-Step PPO, while the underlying language model remains frozen.

The Reward Model achieved **90.00% validation ranking accuracy**, and the optimized policy generally obtained higher surrogate rewards than the original hard-prompt reference in the small validation experiment.

The results provide preliminary evidence that **semantically initialized soft prompts can be optimized toward learned human preferences without modifying the base LLM**, while also highlighting the need for larger-scale and independent human evaluation.
