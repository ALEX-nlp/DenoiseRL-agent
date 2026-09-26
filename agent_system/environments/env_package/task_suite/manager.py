"""Text prompts and replay history shared by the two task-suite backends."""

import re
import numpy as np

from agent_system.environments.base import EnvironmentManagerBase
from agent_system.memory import SimpleMemory
from .runtime import RayTaskEnvs, load_manifest
from agent_system.scienceworld_protocol import render_prompt


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
        result = {"text": self.build_mixed_text_obs_after_prefix(), "image": None,
                  "anchor": list(self.pre_text_obs)}
        if self.config.env.task_suite.benchmark == "scienceworld":
            result["scienceworld_prompt_parts"] = self._scienceworld_prompt_parts()
        return result

    def _scienceworld_prompt_parts(self):
        length = int(self.config.env.history_length)
        parts = []
        for i, (obs, info) in enumerate(zip(self.pre_text_obs, self.infos)):
            history = self.memory._data[i][-length:] if length > 0 else []
            # Native templates are bounded in number. Exhaustive object-action
            # combinations can run to thousands of entries and hide the task.
            templates = info.get("action_templates")
            if not templates:
                raise ValueError("ScienceWorld backend must supply action_templates")
            state = list(dict.fromkeys(s for s in (obs, info.get("look", ""), info.get("inventory", "")) if s))
            parts.append({"task": info["task_description"], "steps": len(self.memory._data[i]),
                          "templates": "\n".join(templates), "state": "\n".join(state),
                          "history": [f"Observation: {h['text_obs']}\nAction: {h['action']}\n" for h in history]})
        return parts

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
        if self.config.env.task_suite.benchmark == "scienceworld":
            return [render_prompt(parts) for parts in self._scienceworld_prompt_parts()]
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
    def create(split, capacity, validation=False):
        vector = RayTaskEnvs(
            manifest, split, capacity, resources,
            (suite.get("eval_max_steps") or config.env.max_steps) if validation else config.env.max_steps,
            suite.reward_mode,
            per_type_limit=suite.get("eval_per_type_limit") if validation else None,
            expected_tasks=suite.get("eval_expected_tasks") if validation else None,
            backend_overrides={key: value for key, value in {
                "simplifications": suite.get("scienceworld_simplifications"),
                "score_mode": suite.get("scienceworld_score_mode"),
                "prompt_version": 2,
                "env_step_limit": suite.get("eval_env_step_limit") if validation else None,
                "stop_on_stagnation": suite.get("eval_stop_on_stagnation", False) if validation else False,
            }.items() if value is not None} if suite.benchmark == "scienceworld" else None,
        )
        if validation:
            vector.eval_protocol = suite.get("eval_protocol", "full")
        return TaskEnvironmentManager(vector, config)
    # Val-only still needs the training manager interface but no unused workers.
    train_capacity = 0 if config.trainer.get("val_only", False) else config.data.train_batch_size * config.env.rollout.n
    train = create("train", train_capacity)
    val = create(suite.eval_split, config.data.val_batch_size * config.actor_rollout_ref.rollout.val_kwargs.n, validation=True)
    return train, {suite.eval_split: val}
