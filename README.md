# Cyberbullying Detection — Project 4 (RL)

## Project summary
This repository extends the Project 3 *fast group-level cyberbullying detection* pipeline with a **reinforcement learning (RL) decision layer** for alerting.

The goal is to make **real-time alert decisions** on large, high-volume message streams using *cheap, streaming features* rather than expensive per-message LLM moderation.

In this notebook the system:
- Simulates a live stream (`civil_comments`) and injects synthetic “raids” (bursty harassment).
- Runs a pretrained **dual-head Set Transformer** over a rolling context window to produce two set-level risk signals.
- Converts those signals into a low-dimensional RL state.
- Trains a PPO policy to choose **when to alert** (`0/1`) under a simple reward that trades off false positives vs missed toxic events.

## Background / problem statement
Social platforms often need to detect **group-level / crowded cyberbullying**: “pile-ons”, raids, or coordinated harassment where many semantically similar harmful messages arrive in a short span.

Why common approaches are limited:
- **Per-message inference is expensive at scale** (latency + cost).
- **LLM moderation is typically even slower/more expensive**, making it hard to run inline on high-volume streams.
- **Basic RAG is similarity-based**: it can retrieve related examples, but it does not directly produce a *set-level* risk score capturing **density** and **concentration**.

This project focuses on a **fast discriminative monitoring loop** and an RL policy that learns alerting behavior.

## Team members
- Tomer Sagi

## Repository contents
- `Cyberbullying_Inference_+_RL_NO_TIME_AWARE.ipynb`
  - Loads a **pretrained set model** (dual-head Set Transformer) from Google Drive.
  - Simulates live streams using `civil_comments` + injected raids.
  - Runs streaming inference to create an `out_df` of features.
  - Defines a Gymnasium environment (`AlertEnv`) and trains a PPO policy.

---

## Dependencies on Project 3 artifacts
This notebook loads a pretrained model from a Google Drive folder:
- `BASE_DIR = /content/drive/MyDrive/grunitech-project3-cyberbullying`
- Expected checkpoint path:
  - `models/cluster_model/settransformer_knn_pretrained.pt`

So, to run Project 4 end-to-end you typically need to first run the Project 3 notebooks that generate these artifacts.

---

## Datasets

### Inference simulation dataset
- Hugging Face Datasets: `civil_comments` (https://huggingface.co/datasets/civil_comments)

Used for:
- `text`: message content.
- `toxicity`: **proxy ground truth** for evaluating alert behavior.

The notebook also:
- assigns synthetic `conv_id` to emulate multiple concurrent threads.
- generates synthetic timestamps `ts` to emulate message arrival times.

### Raid injection (burst simulation)
The notebook defines `inject_raid(...)` to simulate “crowd harassment” bursts:
- samples a high-toxicity template message (based on a threshold)
- injects a burst of near-duplicate messages (controlled by `raid_size` and `dt`)
- assigns them to a dedicated conversation id (e.g., `conv_id=999`)

This is used to test whether the system detects **dense and semantically consistent** bursts without running expensive per-message LLM reasoning.

---

## Model & methods (high level)

### 1) Stream encoding (MiniLM)
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Each incoming message is encoded into an embedding vector.

### 2) Set-level scoring (pretrained dual-head Set Transformer)
- The set model operates on a *context set* (rolling buffer) of embeddings.
- It outputs two signals:
  - `mean_hat`: mean/intensity estimate
  - `conc_hat`: concentration/burstiness estimate

### 3) Feature construction (monitoring loop)
The notebook computes per-step features such as:
- rolling statistics (`rolling_mean`, `rolling_conc`)
- context density (`ctx_density`)
- fraction of toxic messages (`ctx_toxic_frac`)

An `out_df` is produced that contains state + explainability info.

---

## RL: alert policy learning

### Environment
The notebook defines a Gymnasium environment `AlertEnv` with:
- **Observation**: `[rolling_mean, rolling_conc, ctx_density, ctx_toxic_frac]`
- **Action**:
  - `0` = no alert
  - `1` = alert

**Episode definition (as implemented)**
- One episode is a full pass over the constructed stream dataframe (`out_df`).
- Episode length is therefore approximately `len(out_df)` (in the notebook run: `ep_len_mean ≈ 5.04e+03`).
- At each step `t`, the agent observes the 4D state derived from the monitoring loop and chooses whether to alert.

### Reward (as implemented)
Reward is shaped using the *proxy* toxicity:
- Alert on high-toxicity messages (`toxicity > 0.7`): positive reward
- Alert on low-toxicity messages: negative reward
- Do nothing on high-toxicity messages: small penalty

Concretely:

| Agent action | Condition (proxy) | Reward |
|---:|---|---:|
| alert (`1`) | `toxicity > 0.7` | `+1.0` |
| alert (`1`) | `toxicity <= 0.7` | `-0.5` |
| no alert (`0`) | `toxicity > 0.7` | `-0.2` |
| no alert (`0`) | `toxicity <= 0.7` | `0.0` |

### Algorithm
- Stable-Baselines3 `PPO` with an `MlpPolicy`
- Example hyperparameters used in the notebook:
  - `learning_rate=3e-4`
  - `n_steps=256`
  - `batch_size=64`
  - `total_timesteps=10_000`

### Sample from the learning loop (PPO rollouts)
Stable-Baselines3 reports episode statistics during training. In the notebook run:

| PPO iteration | total_timesteps | ep_len_mean | ep_rew_mean |
|---:|---:|---:|---:|
| 20 | 5120 | ~5040 | -216 |
| 40 | 10240 | ~5040 | -118 |

This shows that the agent’s average episodic reward (under the shaped proxy reward) improved during training.

---

## Evaluation: RL policy vs naive threshold baseline
The notebook compares PPO against a simple threshold baseline:

- **Naive baseline**: alert if `rolling_mean > 0.6` OR `rolling_conc > 0.5`.
- **RL policy**: action from `model.predict(obs, deterministic=True)` on the same 4D observation.

Both are scored with the same `evaluate_policy(...)` reward function (the shaped proxy reward described above).

**Final reward comparison (as printed in the notebook)**

| Policy | Total reward |
|---|---:|
| Naive threshold | `-19.8` |
| PPO (RL) | `-19.8` |

**Conclusion**
In this specific run, the trained PPO policy did **not** outperform the naive threshold baseline on the final evaluation metric (both achieve the same total reward).

This is still a useful result because it highlights a key modeling point for this setup:
- With a **single deterministic stream** and a **per-step proxy reward**, PPO can learn (rollout reward improves) yet still fail to beat a hand-tuned rule on the final aggregate score.
- Improving over the baseline likely requires changes such as: multiple randomized episodes (different raid locations / seeds), better reward shaping (sequence-level costs for alert fatigue), richer action space (warn/timeout/escalate), and/or explicit evaluation splits.

---

## Setup instructions

### Recommended: Google Colab
This project is written for Colab and Google Drive.

1. Upload the notebook to Colab
2. Mount Google Drive
3. Ensure `BASE_DIR` points to a Drive folder containing the pretrained set model
4. Run the notebook top-to-bottom

### Local run (optional)
This repo includes `requirements.txt`, but you’ll still need:
- access to the pretrained checkpoint file (or modify the notebook to load it locally)

---

## Limitations
- The RL reward uses `civil_comments.toxicity` as a **proxy ground truth**; it is not a perfect match to cyberbullying.
- The environment is simplified to a binary alert decision.
- The notebook is “no time aware” in the sense that the state is primarily derived from rolling buffers rather than explicit windowed temporal models.

---

## Future work
- **Per-thread aggregation**: build thread-aware context sets instead of mixing multiple `conv_id`s in a single buffer.
- **Conversation-level RL**: expand action space (warn/timeout/slowmode/escalate) and optimize long-term outcomes.
- **Time-window modeling**: explicit window features (e.g., 30s/5m/1h) for true density estimates.
- **Target classification**: identify who/what is targeted and the type/severity of harm.
