import os

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from recipe.denoise_v2.collector import DenoiseTrajectoryCollector
from recipe.denoise_v2.online_ray_trainer import OnlineDenoisePPOTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


@hydra.main(config_path="config", config_name="denoise_v2_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from agent_system.environments import make_envs
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.utils.import_utils import load_extern_type

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        denoise_cfg = config.env.get("denoise", {})
        online_cfg = denoise_cfg.get("online", {})
        denoise_enabled = bool(denoise_cfg.get("enable", False))
        if str(denoise_cfg.get("mode", "")).lower() != "online":
            raise ValueError("main_online_denoise requires env.denoise.mode=online.")
        denoise_model_path = online_cfg.get("model_path", None)
        if denoise_enabled and (denoise_model_path is None or str(denoise_model_path).strip() == ""):
            raise ValueError("Set env.denoise.online.model_path (or DENOISE_MODEL_PATH in the launch script).")

        solver_local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        denoise_local_path = copy_to_local(
            denoise_model_path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        ) if denoise_enabled else None
        with open_dict(config):
            config.actor_rollout_ref.model.path = solver_local_path
            config.env.denoise.online.model_path = denoise_local_path

        envs, val_envs = make_envs(config)

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(solver_local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(solver_local_path, trust_remote_code=trust_remote_code, use_fast=True)
        denoise_tokenizer = hf_tokenizer(denoise_local_path, trust_remote_code=trust_remote_code) if denoise_enabled else None

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError("Online DenoiseRL currently supports the FSDP/FSDP2 backend only.")

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        if config.algorithm.adv_estimator == "gae":
            role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
            mapping[Role.Critic] = global_pool_id

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager

            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError(f"Unsupported reward manager for online denoise: {reward_manager_name}")
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl+env, keep actor_rollout_ref.rollout.n=1 and use env.rollout.n for GRPO groups."
        )

        traj_collector_cls = DenoiseTrajectoryCollector
        traj_collector_cfg = config.get("traj_collector", {}).get("custom_cls", {})
        if traj_collector_cfg.get("path", None) is not None:
            traj_collector_cls = load_extern_type(traj_collector_cfg.path, traj_collector_cfg.name)
        traj_collector = traj_collector_cls(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            denoise_tokenizer=denoise_tokenizer,
            denoise_processor=None,
        )
        if hasattr(traj_collector, "configure_v2"):
            task_envs = getattr(envs, "envs", None)
            is_task_suite = hasattr(task_envs, "task_ids")
            if is_task_suite:
                traj_collector.v2_environment_fingerprint = task_envs.task_pool_fingerprint
            task_ids = getattr(task_envs, "task_ids" if is_task_suite else "game_files", ())
            task_types = getattr(
                task_envs,
                "task_types" if is_task_suite else "game_file_task_types",
                {},
            )
            traj_collector.configure_v2(
                task_ids,
                task_types,
                reset_key="task_id" if is_task_suite else "gamefile",
            )

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = OnlineDenoisePPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
