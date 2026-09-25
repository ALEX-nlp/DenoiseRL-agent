"""Text prompts and replay history shared by the two task-suite backends."""

import re
import numpy as np

from agent_system.environments.base import EnvironmentManagerBase
from agent_system.memory import SimpleMemory
from .runtime import RayTaskEnvs, load_manifest


def project_actions(text_actions):
    actions, valids = [], []
    for text in text_actions:
        matches = re.findall(r"<action>\s*(.*?)\s*</action>", text, flags=re.S | re.I)
        valid = len(matches) == 1 and bool(matches[0]) and "\n" not in matches[0]
        actions.append(matches[0] if valid else "__invalid_action__")
        valids.append(valid)
    return actions, valids


class TaskEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, config):
        super().__init__(envs, project_actions, config)
        self.memory = SimpleMemory()

    def _observations(self):
        return {"text": self.build_mixed_text_obs_after_prefix(), "image": None,
                "anchor": list(self.pre_text_obs)}

    @staticmethod
    def _verify_shared_states(obs, infos, prefixes):
        states = {}
        for observation, info, prefix in zip(obs, infos, prefixes):
            key = (info["task_id"], tuple(prefix))
            state = (observation, info["task_description"], float(info["task_score"]),
                     tuple(sorted(info["admissible_actions"])))
            if key in states and states[key] != state:
                raise RuntimeError("Identical task/prefix produced different states; check simulator version and data")
            states[key] = state

    def reset(self, kwargs):
        obs, infos = self.envs.reset(kwargs)
        self._verify_shared_states(obs, infos, [[] for _ in obs])
        self.pre_text_obs, self.infos = list(obs), list(infos)
        self.memory.reset(len(obs))
        self.memory.keys = ["text_obs", "action"]
        self.finished = [False] * len(obs)
        return self._observations(), infos

    def step_selected(self, indices, text_actions):
        actions, valids = project_actions(text_actions)
        obs, rewards, dones, infos = self.envs.step_selected(indices, actions)
        for j, i in enumerate(indices):
            if not self.finished[i]:
                self.memory._data[i].append({"text_obs": self.pre_text_obs[i], "action": actions[j]})
            self.pre_text_obs[i], self.infos[i] = obs[j], infos[j]
            infos[j]["is_action_valid"] = bool(valids[j] and infos[j].get("is_action_valid", True))
            self.finished[i] = bool(dones[j])
        return self._observations(), np.asarray(rewards, dtype=np.float32), np.asarray(dones), infos

    def step(self, text_actions):
        return self.step_selected(list(range(len(self.pre_text_obs))), text_actions)

    def reset_selected_with_prefixes(self, indices, prefix_actions):
        if len(indices) != len(prefix_actions):
            raise ValueError("Prefix count differs from selected environment count")
        items = [{"task_id": self.infos[i]["task_id"], "prefix_actions": list(actions)}
                 for i, actions in zip(indices, prefix_actions)]
        obs, infos = self.envs.reset_selected(indices, items)
        self._verify_shared_states(obs, infos, prefix_actions)
        for j, i in enumerate(indices):
            self.pre_text_obs[i], self.infos[i] = obs[j], infos[j]
            self.memory._data[i] = list(infos[j]["prefix_history"])
            self.finished[i] = False
        return self._observations(), infos

    def build_mixed_text_obs_after_prefix(self, prefix_lens=None):
        history_length = int(self.config.env.history_length)
        prompts = []
        for i, (obs, info) in enumerate(zip(self.pre_text_obs, self.infos)):
            history = self.memory._data[i][-history_length:] if history_length > 0 else []
            history_text = "\n".join(f"Observation: {h['text_obs']}\nAction: {h['action']}" for h in history)
            state = "\n".join(s for s in (obs, info.get("look", ""), info.get("inventory", "")) if s)
            actions = "\n".join(info["admissible_actions"])
            prompts.append(
                f"You are an agent in {self.config.env.task_suite.benchmark}.\n"
                f"Task: {info['task_description']}\n"
                f"Actions already taken: {len(self.memory._data[i])}\n"
                f"Recent history:\n{history_text}\nCurrent observation:\n{state}\n"
                f"Available actions (replace placeholders where present):\n{actions}\n"
                "Briefly reason inside <think>...</think>, then output exactly one action inside <action>...</action>."
            )
        return prompts

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for row, info in zip(reversed(total_batch_list[batch_idx]), reversed(total_infos[batch_idx])):
            if row["active_masks"]:
                success["success_rate"].append(float(info["won"]))
                task_type = self.envs.task_types[info["task_id"]]
                slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", task_type)
                success[f"task_type/{slug}/success_rate"].append(float(info["won"]))
                return


def make_task_envs(config):
    from omegaconf import OmegaConf
    suite = config.env.task_suite
    manifest = load_manifest(suite.manifest_path, suite.benchmark)
    resources = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)
    def create(split, capacity):
        vector = RayTaskEnvs(manifest, split, capacity, resources, config.env.max_steps, suite.reward_mode)
        return TaskEnvironmentManager(vector, config)
    # Val-only still needs the training manager interface but no unused workers.
    train_capacity = 0 if config.trainer.get("val_only", False) else config.data.train_batch_size * config.env.rollout.n
    train = create("train", train_capacity)
    val = create(suite.eval_split, config.data.val_batch_size * config.actor_rollout_ref.rollout.val_kwargs.n)
    return train, {suite.eval_split: val}
