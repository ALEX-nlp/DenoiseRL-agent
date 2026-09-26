"""Generate matched, overridable launch configurations without shell interpolation."""

import argparse
import json
import netrc
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_ROOT = Path("/inspire/hdd/global_user/xucaijun-253108120121/Model")


def has_wandb_credentials(env):
    """Check common credential sources locally, without login/network or logging keys.

    This checks presence, not server-side validity. Custom SDK authentication
    sources can still opt into online mode explicitly via WANDB_MODE.
    """
    if env.get("WANDB_API_KEY", "").strip() or env.get("WANDB_IDENTITY_TOKEN_FILE", "").strip():
        return True
    host = urlparse(env.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname
    if not host:
        return False
    paths = ([Path(env["NETRC"]).expanduser()] if env.get("NETRC")
             else [Path.home() / ".netrc", Path.home() / "_netrc"])
    for path in paths:
        try:
            if not path.is_file():
                continue
            auth = netrc.netrc(str(path)).authenticators(host)
        except (OSError, netrc.NetrcParseError):
            # Do not print parser errors: malformed netrc lines can contain keys.
            return False
        return bool(auth and auth[2])
    return False


def configure_wandb_mode(env, benchmark):
    """Respect explicit modes; unconfigured batch jobs must remain runnable."""
    if "WANDB_MODE" in env:
        return
    if benchmark == "scienceworld" and has_wandb_credentials(env):
        env["WANDB_MODE"] = "online"
    else:
        env["WANDB_MODE"] = "offline"
        if benchmark == "scienceworld":
            print("[launch] No W&B credentials found in the environment or netrc; "
                  "saving metrics offline. Live web charts require logging in on the job's runtime "
                  "or providing WANDB_API_KEY through the job environment. "
                  "Offline runs can be uploaded later with wandb sync.", flush=True)


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.name == "actor":
        path = path.parent
    tracker = path / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        step = tracker.read_text().strip()
        if not step.isdigit():
            raise ValueError(f"Invalid checkpoint tracker: {tracker}")
        path = path / f"global_step_{step}"
    if not path.name.startswith("global_step_") or not (path / "actor").is_dir():
        raise ValueError(f"Expected a checkpoint root, global_step directory or actor directory: {path}")
    return str(path)


def recommended(benchmark):
    # Match ALFWorld's task batch; rho control is inherited from denoise_v2_base.
    return {"webshop": {"batch": 16, "rollouts": 16, "steps": 15, "history": 2, "prompt": 4096},
            "scienceworld": {"batch": 16, "rollouts": 8, "steps": 50, "history": 4, "prompt": 8192}}[benchmark]


def build_overrides(args):
    profile = recommended(args.benchmark)
    denoise = args.method == "denoise" and args.mode == "train"
    training = args.mode == "train"
    scienceworld = args.benchmark == "scienceworld"
    eval_split = args.eval_split or ("test" if scienceworld and not training else "dev")
    if scienceworld and training and eval_split != "dev":
        raise ValueError("Use dev for training monitoring; test is reserved for the final evaluation")
    data_dir = Path(args.data_dir or ROOT / "recipe/denoise_v2/local_data" / args.benchmark).expanduser().resolve()
    model_root = os.getenv("MODEL_ROOT")
    def model_path(env_key, default):
        path = os.getenv(env_key)
        if path is None:
            return str(Path(model_root or DEFAULT_MODEL_ROOT) / default)
        return str(Path(model_root) / path) if model_root and not Path(path).is_absolute() else path
    experiment = os.getenv("EXPERIMENT_NAME", f"{args.benchmark}_{args.method}_7b_seed{args.seed}" + ("_swiftsage_n8_t50" if scienceworld else ""))
    values = {
        "data.train_files": str(data_dir / "train.parquet"),
        "data.val_files": str(data_dir / f"{eval_split}.parquet"),
        "data.train_batch_size": int(os.getenv("TRAIN_BATCH_SIZE", profile["batch"])),
        "data.val_batch_size": int(os.getenv("VAL_BATCH_SIZE", 16 if args.benchmark == "webshop" else 8)),
        "data.max_prompt_length": profile["prompt"], "data.max_response_length": 256,
        "actor_rollout_ref.model.path": model_path("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"),
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 128,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 4,
        "actor_rollout_ref.actor.use_kl_loss": training,
        "actor_rollout_ref.actor.kl_loss_coef": 0.01,
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.actor.use_invalid_action_penalty": True,
        "actor_rollout_ref.actor.invalid_action_penalty_coef": 0.1,
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 4,
        "actor_rollout_ref.rollout.n": 1,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.do_sample": True,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 2,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.5,
        "actor_rollout_ref.rollout.max_model_len": profile["prompt"] + 256,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 16384,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 4,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.0,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0,
        "env.env_name": f"{args.benchmark}/TaskSuite",
        "env.task_suite.benchmark": args.benchmark,
        "env.task_suite.manifest_path": str(data_dir / "tasks.json"),
        "env.task_suite.eval_split": eval_split,
        "env.task_suite.reward_mode": "score",
        "env.seed": args.seed, "env.max_steps": profile["steps"], "env.history_length": profile["history"],
        "env.rollout.n": profile["rollouts"],
        "env.resources_per_worker.num_cpus": 0.25,
        "env.denoise.enable": denoise,
        "env.denoise.main_rollout_n": 0 if denoise else profile["rollouts"],
        "env.denoise.sub_rollout_k": profile["rollouts"] if denoise else 0,
        "env.denoise.v2.enabled": training,
        "env.denoise.v2.shuffle_seed": args.seed,
        "env.denoise.online.model_path": model_path("DENOISE_MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct") if denoise else None,
        "env.denoise.online.response_length": 256,
        "env.denoise.online.denoiser_gpu_memory_utilization": 0.2,
        "env.denoise.online.solver_gpu_memory_utilization": 0.5,
        "algorithm.adv_estimator": "grpo", "algorithm.filter_groups.enable": False,
        "algorithm.use_kl_in_reward": False,
        "trainer.n_gpus_per_node": int(os.getenv("N_GPUS_PER_NODE", 8)),
        "trainer.nnodes": 1, "trainer.critic_warmup": 0,
        "trainer.total_epochs": 500, "trainer.total_training_steps": 500,
        "trainer.test_freq": 25, "trainer.save_freq": 25,
        "trainer.val_before_train": True, "trainer.val_only": not training,
        "trainer.project_name": f"denoise_v2_{args.benchmark}", "trainer.experiment_name": experiment,
        "trainer.logger": ["console", "wandb"] if training else ["console"],
        "trainer.default_local_dir": str(ROOT / "checkpoints" / f"denoise_v2_{args.benchmark}" / experiment),
        "trainer.rollout_data_dir": str(ROOT / "recipe/denoise_v2/dumps" / experiment / "rollout"),
        "trainer.validation_data_dir": str(ROOT / "recipe/denoise_v2/dumps" / experiment / "validation"),
    }
    if scienceworld:
        protocol = getattr(args, "eval_protocol", "swiftsage")
        values.update({
            "env.task_suite.scienceworld_simplifications": "easy",
            "env.task_suite.scienceworld_score_mode": "last_nonnegative",
            "env.task_suite.eval_protocol": "dev_monitor" if training else protocol,
            "env.task_suite.eval_per_type_limit": 3 if training else ((3 if eval_split == "dev" else 10) if protocol == "swiftsage" else None),
            "env.task_suite.eval_expected_tasks": 270 if not training and eval_split == "test" and protocol == "swiftsage" else None,
            "env.task_suite.eval_max_steps": profile["steps"] if training else 600,
            "env.task_suite.eval_env_step_limit": profile["steps"] if training else 300,
            "env.task_suite.eval_stop_on_stagnation": not training,
            "trainer.val_before_train": not training,
        })
        if not training:
            values["trainer.experiment_name"] = experiment + f"_{eval_split}_{protocol}"
            values["trainer.validation_data_dir"] = str(Path(values["trainer.validation_data_dir"]) / f"eval_{protocol}")
    if not denoise:
        # The clean baseline retains task-pool traversal but never adds noise.
        values.update({
            "env.denoise.v2.initial_rho": 0.0,
            "env.denoise.v2.min_rho": 0.0,
            "env.denoise.v2.max_rho": 0.0,
            "env.denoise.v2.alpha": 0.0,
        })
    if not training:
        if args.checkpoint:
            values.update({"trainer.resume_mode": "resume_path", "trainer.resume_from_path": resolve_checkpoint(args.checkpoint)})
        elif not args.base_model:
            raise ValueError("Evaluation requires --checkpoint/CKPT_DIR or explicit --base-model")
        else:
            values["trainer.resume_mode"] = "disable"
    return [f"{key}={json.dumps(value)}" for key, value in values.items()]


def final_evaluation_overrides(completion, protocol):
    """Reuse the trained model/config, but isolate the held-out evaluation."""
    checkpoint = resolve_checkpoint(completion["checkpoint"])
    values = {
        "data.val_files": str(Path(completion["manifest_path"]).parent / "test.parquet"),
        "env.task_suite.eval_split": "test",
        "env.task_suite.eval_protocol": protocol,
        "env.task_suite.eval_per_type_limit": 10 if protocol == "swiftsage" else None,
        "env.task_suite.eval_expected_tasks": 270 if protocol == "swiftsage" else None,
        "env.task_suite.eval_max_steps": 600,
        "env.task_suite.eval_env_step_limit": 300,
        "env.task_suite.eval_stop_on_stagnation": True,
        "env.task_suite.scienceworld_simplifications": "easy",
        "env.task_suite.scienceworld_score_mode": "last_nonnegative",
        "env.task_suite.reward_mode": "score",
        "env.denoise.enable": False, "env.denoise.v2.enabled": False,
        "env.denoise.online.model_path": None,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.0,
        "trainer.val_only": True, "trainer.val_before_train": True,
        "trainer.resume_mode": "resume_path", "trainer.resume_from_path": checkpoint,
        "trainer.completion_path": None,
        "trainer.experiment_name": completion["experiment_name"] + "_test_" + protocol,
        "trainer.validation_data_dir": str(Path(completion["validation_data_dir"]) / ("final_" + protocol)),
    }
    return [f"{key}={json.dumps(value)}" for key, value in values.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["webshop", "scienceworld", "sciworld"], required=True)
    parser.add_argument("--method", choices=["baseline", "denoise"], default="denoise")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--data-dir", default=os.getenv("TASK_DATA_DIR"))
    parser.add_argument("--eval-split", choices=["dev", "test"], default=None)
    parser.add_argument("--eval-protocol", choices=["swiftsage", "full"], default="swiftsage",
                        help="ScienceWorld: paper's first 10 test variations per type, or all native test variations")
    parser.add_argument("--skip-final-eval", action="store_true", help="Skip automatic ScienceWorld evaluation after training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=os.getenv("CKPT_DIR"))
    parser.add_argument("--base-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    if args.benchmark == "sciworld":
        args.benchmark = "scienceworld"
    if any("=" not in item for item in overrides):
        parser.error("Extra arguments must be Hydra key=value overrides")
    command = [sys.executable, "-m", "recipe.denoise_v2.main_online_denoise", "--config-name", "task_suite_trainer"]
    command += build_overrides(args) + overrides
    if args.dry_run:
        print(json.dumps(command, indent=2))
        if args.benchmark == "scienceworld" and args.mode == "train" and not args.skip_final_eval:
            print(f"After successful training: save the final checkpoint and evaluate test ({args.eval_protocol}).", file=sys.stderr)
        return
    env = dict(os.environ)
    configure_wandb_mode(env, args.benchmark)
    print(f"[launch] WANDB_MODE={env['WANDB_MODE']}", flush=True)
    if args.benchmark != "scienceworld" or args.mode != "train" or args.skip_final_eval:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return
    # A unique success marker prevents evaluating a stale checkpoint after a
    # failed/empty/resumed run. It is written only after the final save succeeds.
    with tempfile.TemporaryDirectory(prefix="scienceworld-completion-") as directory:
        completion_path = Path(directory) / "completed.json"
        train_command = command + [f"trainer.completion_path={json.dumps(str(completion_path))}"]
        subprocess.run(train_command, cwd=ROOT, env=env, check=True)
        if not completion_path.is_file():
            raise RuntimeError("Training did not produce a final checkpoint marker; refusing to evaluate a stale checkpoint")
        completion = json.loads(completion_path.read_text())
        final_command = command + final_evaluation_overrides(completion, args.eval_protocol)
        print(f"[launch] Final test evaluation: {completion['checkpoint']} ({args.eval_protocol})", flush=True)
        # An explicitly set W&B run ID must not merge train and test runs.
        eval_env = {key: value for key, value in env.items() if key not in {"WANDB_RUN_ID", "WANDB_RESUME"}}
        subprocess.run(final_command, cwd=ROOT, env=eval_env, check=True)


if __name__ == "__main__":
    main()
