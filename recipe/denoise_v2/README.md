# ALFWorld DenoiseRL v2

WebShop and ScienceWorld now have matched GRPO baseline and DenoiseRL v2
recipes. See [task-suite setup, recommended parameters, training and evaluation](task_suite/README.md).

This recipe is the agent counterpart of the mathematical `recipe/denoise_v2` algorithm. It traverses a shuffled pool of concrete ALFWorld gamefiles without replacement inside each pool epoch and maintains one dynamic noise state per ALFWorld task type.

## Complete seen/unseen evaluation

Run exhaustive evaluation from either an experiment checkpoint root or one
specific `global_step_*` directory:

```bash
CKPT_DIR=checkpoints/verl_agent_alfworld_denoise_v2/<experiment> \
  bash recipe/denoise_v2/run_denoise_grpo_eval.sh
```

The default `EVAL_SPLIT=both` evaluates every supported, solvable gamefile in
`valid_seen` and `valid_unseen` exactly once when `VAL_N=1`. Every reset is
pinned to a concrete gamefile. Validation fails on a missing, duplicate, or
unexpected gamefile, reports `gamefile_count`, `gamefile_episode_count`,
`gamefile_unique_count`, and `gamefile_coverage` under each split, and writes the
gamefile path with one collapsed solver trajectory into every validation JSONL
row. This is a clean policy evaluation from each task's initial state; the
training-time denoiser and curriculum state are not needed for inference.

## What a gamefile represents

The bundled train directory contains 2,435 task-configuration directories and 4,652 raw `game.tw-pddl` trials. The current `AlfredTWEnv` filters out 1,086 trials marked unsolvable and 13 movable/sliced trials, leaving an effective pool of 3,553 gamefiles across 6 supported high-level task families. A gamefile identifies one fully instantiated task: task family, target object/receptacle, scene, concrete object placement, and initial state.

## Pool and curriculum

- The complete, already-filtered AlfredTWEnv training gamefile list is shuffled with `env.denoise.v2.shuffle_seed` at the beginning of each pool epoch.
- Each optimizer step consumes the next `data.train_batch_size` gamefiles without replacement.
- All 16 rollouts in a GRPO group are pinned to the same active gamefile and use the same rho.
- The 16 successes are first averaged per gamefile, then those gamefile accuracies are averaged by task type for the current step.
- Each observed task type updates its shared rho with `rho[type] <- clip(rho[type] + alpha * (accuracy[type] - target_accuracy))`.
- After every optimizer step, the complete active batch is replaced. A gamefile is never repeated inside the same pool epoch.
- If the pool epoch ends in a partial batch, the batch is completed from the next shuffled epoch while avoiding duplicates inside the batch and immediate reuse from the preceding batch when the pool size permits it.
- Pool order/cursor, task-type rho states, and both curriculum RNG states are checkpointed in `denoise_v2_curriculum.json`.

At `rho=0`, the group starts clean and the unnecessary denoiser shadow rollout is skipped.

## Run

The recipe reuses `recipe/alfworld_denoise/local_data` by default. Prepare it if needed:

```bash
bash recipe/denoise_v2/setup_data.sh --download --offline
```

Then launch:

```bash
bash recipe/denoise_v2/run_denoise_grpo_train.sh
```

Useful overrides:

```bash
V2_INITIAL_RHO=0.0 \
V2_MAX_RHO=0.5 \
V2_TARGET_ACCURACY=0.75 \
V2_ALPHA=0.2 \
V2_SHUFFLE_SEED=0 \
bash recipe/denoise_v2/run_denoise_grpo_train.sh
```

The main entry point is `recipe.denoise_v2.main_online_denoise`; `run_denoise_grpo_train_base.sh` contains the shared Hydra launch arguments used by the wrapper.
