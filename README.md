# DenoiseRL-Agent: Recovering from Noisy Action Prefixes

<p align="center">
  <a href="https://github.com/ALEX-nlp/DenoiseRL">
    <img src="https://img.shields.io/badge/Main%20project-DenoiseRL-2f6f4e" alt="DenoiseRL">
  </a>
  <a href="./recipe/denoise_v2">
    <img src="https://img.shields.io/badge/Implementation-recipe%2Fdenoise__v2-2b64d8" alt="recipe/denoise_v2">
  </a>
</p>

**DenoiseRL-Agent** extends the DenoiseRL recovery objective from mathematical reasoning to multi-step agent-environment interaction. A weaker agent supplies failed action trajectories, the policy is placed at a noisy intermediate state, and reinforcement learning trains it to recover and complete the task successfully.

The maintained implementation is **[`recipe/denoise_v2`](./recipe/denoise_v2)**. Use this recipe for training and evaluation.

WebShop and ScienceWorld adapters, matched GRPO baselines, and recommended
8×80GB launch profiles are documented in
[`recipe/denoise_v2/task_suite`](./recipe/denoise_v2/task_suite/README.md).

## Method

For an ALFWorld gamefile, a weak policy first produces a failed interaction trajectory. DenoiseRL-Agent truncates that trajectory into an action prefix `z`, restores the environment state reached by the prefix, and asks the trainable policy to continue:

```text
y ~ pi_theta(. | q, z),     reward = environment_success([z, y])
```

The agent version follows the same recovery principle as [DenoiseRL](https://github.com/ALEX-nlp/DenoiseRL), with environment-specific curriculum control:

- **Step-level noise.** Prefix length is measured in environment interaction steps rather than reasoning lines.
- **Recovery-only groups.** Each GRPO group contains 16 recovery rollouts and no additional clean rollout slots.
- **Task-wise adaptive difficulty.** Each of the six ALFWorld task families maintains its own prefix ratio `rho`.
- **Online control.** Recovery accuracy updates the ratio as:

  ```text
  rho[type] <- clip(rho[type] + alpha * (accuracy[type] - target_accuracy))
  ```

- **Fresh gamefiles.** Training traverses the filtered gamefile pool without replacement within each pool epoch and replaces the active batch after every optimizer step.
- **Checkpointable curriculum.** Pool order, cursor, task-wise ratios, and curriculum random states are saved with the training checkpoint.

Easy task families receive longer failed prefixes; difficult families receive shorter ones. At `rho = 0`, the group starts from a clean initial state and skips unnecessary weak-policy prefix generation.

## Results

Experiments use Qwen2.5-7B-Instruct as the policy and Qwen2.5-1.5B-Instruct as the weak model.

| Method | ALFWorld seen | ALFWorld unseen |
| --- | ---: | ---: |
| Base model | 13.6 | 12.7 |
| GRPO | 80.7 | 79.9 |
| **DenoiseRL-GRPO** | **96.3** | **88.1** |

DenoiseRL-Agent improves over GRPO by 15.6 percentage points on seen environments and 8.2 points on unseen environments.

## Repository layout

```text
DenoiseRL-agent/
├── agent_system/                       # agent loop and ALFWorld environment
├── recipe/denoise_v2/
│   ├── README.md                       # implementation details
│   ├── config/denoise_v2_trainer.yaml  # trainer and curriculum configuration
│   ├── collector.py                    # DenoiseRL trajectory collection
│   ├── gamefile_curriculum.py          # gamefile pool and task-wise rho state
│   ├── main_online_denoise.py          # training entry point
│   ├── online_ray_trainer.py           # online denoising trainer
│   ├── trajectory_prefix.py            # action-prefix construction
│   ├── run_denoise_grpo_train.sh       # main training launcher
│   └── run_denoise_grpo_eval.sh        # exhaustive evaluation launcher
├── tests/recipe/                       # DenoiseRL-Agent tests
└── verl/                               # RL training runtime
```

## Installation

Create an isolated environment and install the repository:

```bash
conda create -n denoiserl-agent python=3.12 -y
conda activate denoiserl-agent

pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

Install ALFWorld dependencies:

```bash
pip install gymnasium==0.29.1
pip install stable-baselines3==2.6.0
pip install alfworld
```

Hardware-sensitive packages should be installed against the CUDA and driver stack of the target cluster.

## Prepare ALFWorld data

On a machine with network access, download, extract, and prepare the local trainer data:

```bash
bash recipe/denoise_v2/setup_data.sh --download --offline
```

This prepares:

```text
recipe/alfworld_denoise/local_data/
├── alfworld/
│   ├── json_2.1.1/{train,valid_seen,valid_unseen}
│   └── logic/{alfred.pddl,alfred.twl2}
└── verl_agent/text/{train,test}.parquet
```

For an offline cluster, copy the prepared `local_data/` directory to the same location and run the setup script without `--download` to verify or regenerate the local parquet files.

The effective training pool contains 3,553 supported, solvable gamefiles across six task families after filtering unsupported trials.

## Training

Launch the maintained DenoiseRL-Agent recipe:

```bash
bash recipe/denoise_v2/run_denoise_grpo_train.sh
```

Typical overrides:

```bash
MODEL_ROOT=/path/to/models \
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
DENOISE_MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct \
N_GPUS_PER_NODE=8 \
V2_INITIAL_RHO=0.0 \
V2_MAX_RHO=0.5 \
V2_TARGET_ACCURACY=0.75 \
V2_ALPHA=0.2 \
V2_SHUFFLE_SEED=0 \
bash recipe/denoise_v2/run_denoise_grpo_train.sh
```

Important settings:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `TRAIN_BATCH_SIZE` | `16` | active gamefiles per optimizer step |
| `SUB_ROLLOUT_K` | `16` | recovery rollouts per gamefile |
| `V2_INITIAL_RHO` | `0.0` | initial action-prefix ratio |
| `V2_MIN_RHO` | `0.0` | minimum prefix ratio |
| `V2_MAX_RHO` | `0.5` | maximum prefix ratio |
| `V2_TARGET_ACCURACY` | `0.75` | target recovery accuracy |
| `V2_ALPHA` | `0.2` | curriculum update size |
| `V2_SHUFFLE_SEED` | `0` | gamefile-pool shuffle seed |
| `MAX_STEPS` | `50` | maximum environment steps |
| `HISTORY_LENGTH` | `2` | recent interaction history retained in the prompt |

The main Python entry point is:

```bash
python -m recipe.denoise_v2.main_online_denoise
```

Use the shell launcher for normal runs because it supplies the complete Hydra configuration and enforces the recovery-only rollout layout.

## Evaluation

Evaluate either an experiment checkpoint root or a specific `global_step_*` directory:

```bash
CKPT_DIR=/path/to/checkpoints/experiment \
  bash recipe/denoise_v2/run_denoise_grpo_eval.sh
```

The default `EVAL_SPLIT=both` evaluates all supported gamefiles from both `valid_seen` and `valid_unseen`:

```bash
EVAL_SPLIT=both \
CKPT_DIR=/path/to/checkpoints/experiment \
  bash recipe/denoise_v2/run_denoise_grpo_eval.sh
```

Other valid values are `seen` and `unseen`.

Evaluation starts each task from its clean initial state; the training-time weak policy and curriculum state are not needed for inference. The evaluator pins every reset to a concrete gamefile, checks for missing or duplicate games, reports coverage metrics for each split, and writes the gamefile path with the collapsed solver trajectory to the validation JSONL output.

## Outputs and resuming

Training checkpoints include `denoise_v2_curriculum.json`, which restores:

- the shuffled gamefile order and pool cursor;
- task-wise prefix ratios;
- curriculum random states; and
- active curriculum progress.

Rollout and validation JSONL dumps are enabled by default under:

```text
recipe/denoise_v2/dumps/<experiment>/{rollout,validation}/
```

Set `ENABLE_DUMP=0` to disable these dumps. Checkpoint and experiment locations can be overridden through the environment variables in `recipe/denoise_v2/params.sh`.

## Further details

See [`recipe/denoise_v2/README.md`](./recipe/denoise_v2/README.md) for the exact gamefile traversal policy, curriculum behavior, and exhaustive-evaluation guarantees.
