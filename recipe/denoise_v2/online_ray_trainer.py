import copy
import json
from pathlib import Path

from omegaconf import OmegaConf, open_dict

from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.workers.rollout.async_server import AsyncLLMServerManager


DENOISE_ROLLOUT = "denoise_rollout"
SOLVER_ROLLOUT = "actor_rollout"


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    return cfg.get(key, default)


def _is_null(value):
    return value is None or str(value).strip().lower() in {"", "none", "null"}


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off", "none", "null"}


class OnlineDenoisePPOTrainer(RayPPOTrainer):
    """PPO trainer that shares a fixed denoiser rollout model with the solver."""

    def _online_cfg(self):
        denoise_cfg = self.config.env.get("denoise", {})
        return denoise_cfg.get("online", {})

    def _build_denoise_actor_rollout_cfg(self):
        online_cfg = self._online_cfg()
        model_path = _cfg_get(online_cfg, "model_path", None)
        if model_path is None or str(model_path).strip() == "":
            raise ValueError("env.denoise.online.model_path must be set for online DenoiseRL.")

        denoise_cfg = copy.deepcopy(self.config.actor_rollout_ref)
        with open_dict(denoise_cfg):
            denoise_cfg.model.path = str(model_path)
            denoise_cfg.rollout.n = 1
            # This worker is inference-only and never receives actor updates. Its
            # FSDP weights need to initialize vLLM once, then subsequent denoiser
            # steps can wake the sleeping engine without copying the same model.
            denoise_cfg.rollout.sync_weights_every_generation = False

            overrides = {
                "name": "rollout_name",
                "tensor_model_parallel_size": "tensor_model_parallel_size",
                "max_model_len": "max_model_len",
                "max_num_batched_tokens": "max_num_batched_tokens",
                "max_num_seqs": "max_num_seqs",
                "prompt_length": "prompt_length",
                "response_length": "response_length",
                "temperature": "temperature",
                "top_p": "top_p",
                "top_k": "top_k",
                "do_sample": "do_sample",
                "enforce_eager": "enforce_eager",
                "free_cache_engine": "free_cache_engine",
                "enable_chunked_prefill": "enable_chunked_prefill",
            }
            for rollout_key, online_key in overrides.items():
                value = _cfg_get(online_cfg, online_key, None)
                if value is not None:
                    setattr(denoise_cfg.rollout, rollout_key, value)

            # The denoiser is inference-only. It is started with role="rollout",
            # so these actor knobs are only defensive if a backend inspects them.
            denoise_cfg.actor.fsdp_config.optimizer_offload = True
            denoise_cfg.actor.use_kl_loss = False

        return denoise_cfg

    def _apply_colocated_gpu_utilization(self, solver_cfg, denoise_cfg):
        online_cfg = self._online_cfg()
        denoiser_util = _cfg_get(online_cfg, "denoiser_gpu_memory_utilization", None)
        solver_util = _cfg_get(online_cfg, "solver_gpu_memory_utilization", None)

        if not _is_null(denoiser_util) or not _is_null(solver_util):
            if not _is_null(denoiser_util):
                with open_dict(denoise_cfg):
                    denoise_cfg.rollout.gpu_memory_utilization = float(denoiser_util)
            if not _is_null(solver_util):
                with open_dict(solver_cfg):
                    solver_cfg.rollout.gpu_memory_utilization = float(solver_util)
            mode = "explicit"
        else:
            shared_util = _cfg_get(online_cfg, "shared_gpu_memory_utilization", None)
            if not _is_null(shared_util):
                base_util = float(shared_util)
                cap = float(_cfg_get(online_cfg, "colocate_gpu_util_cap", 0.9))
                with open_dict(denoise_cfg):
                    denoise_cfg.rollout.gpu_memory_utilization = base_util
                with open_dict(solver_cfg):
                    solver_cfg.rollout.gpu_memory_utilization = min(2.0 * base_util, cap)
                mode = "legacy-shared"
            else:
                legacy_util = _cfg_get(online_cfg, "gpu_memory_utilization", None)
                if _is_null(legacy_util):
                    return
                with open_dict(denoise_cfg):
                    denoise_cfg.rollout.gpu_memory_utilization = float(legacy_util)
                mode = "legacy-denoiser-only"

        print(
            "[online-denoise] vLLM gpu_memory_utilization "
            f"({mode}): denoiser={denoise_cfg.rollout.gpu_memory_utilization:.3f}, "
            f"solver={solver_cfg.rollout.gpu_memory_utilization:.3f}."
        )

    def _separate_denoise_process(self) -> bool:
        return _as_bool(_cfg_get(self._online_cfg(), "separate_denoise_process", True))

    def _v2_enabled(self) -> bool:
        return bool(getattr(self.traj_collector, "v2_enabled", False))

    def _v2_state_path(self, *, loading: bool = False) -> Path:
        if loading and self.config.trainer.resume_mode == "resume_path":
            checkpoint_dir = Path(self.config.trainer.resume_from_path)
        else:
            checkpoint_dir = (
                Path(self.config.trainer.default_local_dir)
                / f"global_step_{self.global_steps}"
            )
        return checkpoint_dir / "denoise_v2_curriculum.json"

    def _save_checkpoint(self):
        super()._save_checkpoint()
        if not self._v2_enabled():
            return
        state_path = self._v2_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(self.traj_collector.v2_state_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _load_checkpoint(self):
        result = super()._load_checkpoint()
        if not self._v2_enabled() or self.global_steps <= 0:
            return result
        state_path = self._v2_state_path(loading=True)
        if not state_path.exists():
            raise FileNotFoundError(
                f"Missing DenoiseRL v2 state for resumed step {self.global_steps}: {state_path}"
            )
        self.traj_collector.load_v2_state_dict(
            json.loads(state_path.read_text(encoding="utf-8"))
        )
        print(f"[denoise v2] restored task-type curriculum state from {state_path}.")
        return result

    def _add_optional_worker_classes(self, pool_to_cls, solver_cfg):
        if self.use_critic:
            critic_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            pool_to_cls.setdefault(critic_pool, {})["critic"] = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic],
                config=self.config.critic,
            )

        if self.use_reference_policy:
            ref_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            pool_to_cls.setdefault(ref_pool, {})["ref"] = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=solver_cfg,
                role="ref",
            )

        if self.use_rm:
            rm_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            pool_to_cls.setdefault(rm_pool, {})["rm"] = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )

    def _spawn_worker_groups(self, pool_class_groups, wg_kwargs):
        all_wg = {}
        for pool, class_dict in pool_class_groups:
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))
        return all_wg

    def init_workers(self):
        if not self.config.env.denoise.enable:
            # Clean baseline/evaluation: no weak model or tokenizer is loaded.
            return super().init_workers()
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if not self.hybrid_engine:
            raise NotImplementedError("Online DenoiseRL requires hybrid_engine actor-rollout workers.")

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_cls = self.role_worker_mapping[Role.ActorRollout]
        solver_cfg = copy.deepcopy(self.config.actor_rollout_ref)
        denoise_cfg = self._build_denoise_actor_rollout_cfg()
        self._apply_colocated_gpu_utilization(solver_cfg, denoise_cfg)

        separate_denoise_process = self._separate_denoise_process()
        if separate_denoise_process:
            # Keep the denoiser on the same placement group/GPU as the solver, but
            # in a separate Ray process. vLLM sleep mode only allows one sleep-mode
            # engine instance per process.
            resource_pool.max_colocate_count = max(resource_pool.max_colocate_count, 2)

        self.resource_pool_to_cls[resource_pool][SOLVER_ROLLOUT] = RayClassWithInitArgs(
            cls=actor_cls,
            config=solver_cfg,
            role="actor_rollout",
        )
        self._add_optional_worker_classes(self.resource_pool_to_cls, solver_cfg)

        pool_class_groups = []
        if separate_denoise_process:
            pool_class_groups.append(
                (
                    resource_pool,
                    {
                        DENOISE_ROLLOUT: RayClassWithInitArgs(
                            cls=actor_cls,
                            config=denoise_cfg,
                            role="rollout",
                        )
                    },
                )
            )
        else:
            self.resource_pool_to_cls[resource_pool][DENOISE_ROLLOUT] = RayClassWithInitArgs(
                cls=actor_cls,
                config=denoise_cfg,
                role="rollout",
            )

        pool_class_groups.extend(self.resource_pool_to_cls.items())

        wg_kwargs = {"device_name": self.device_name}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        all_wg = self._spawn_worker_groups(pool_class_groups, wg_kwargs)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        self.denoise_rollout_wg = all_wg[DENOISE_ROLLOUT]
        self.actor_rollout_wg = all_wg[SOLVER_ROLLOUT]
        self.denoise_rollout_wg.init_model()
        self.actor_rollout_wg.init_model()

        if hasattr(self.traj_collector, "set_denoise_rollout_wg"):
            self.traj_collector.set_denoise_rollout_wg(self.denoise_rollout_wg)

        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )
