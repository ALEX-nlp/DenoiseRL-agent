"""Task pinning, selected stepping and deterministic prefix replay."""

from copy import deepcopy
import json
import math
from pathlib import Path

from .backends import fingerprint, make_backend, WebShopBackend
from agent_system.scienceworld_protocol import select_tasks


def load_manifest(path, benchmark):
    manifest = json.loads(Path(path).read_text())
    if manifest["benchmark"] != benchmark or manifest["version"] != 1:
        raise ValueError("Task manifest benchmark/version mismatch; rerun prepare_tasks.")
    seen = set()
    for split in ("train", "dev", "test"):
        rows = manifest["splits"][split]
        if not rows:
            raise ValueError(f"Empty task split: {split}")
        for row in rows:
            task_id = row["task_id"]
            if not isinstance(task_id, str) or not row["task_type"] or task_id in seen:
                raise ValueError(f"Duplicate or invalid task identity across splits: {task_id!r}")
            seen.add(task_id)
    return manifest


class TaskWorker:
    def __init__(self, benchmark, options, max_steps, reward_mode, expected_fingerprint=None, backend=None):
        if reward_mode not in {"score", "success"} or max_steps < 1:
            raise ValueError("Invalid reward_mode or max_steps")
        self.backend = backend or make_backend(benchmark, {**options, "max_steps": max_steps})
        if expected_fingerprint and benchmark == "webshop":
            if self.backend.catalog_fingerprint != expected_fingerprint:
                self.backend.close()
                raise ValueError("WebShop goal catalog differs from manifest (data or catalog seed changed).")
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.task_id = None
        self.done = True

    def reset(self, task_id, prefix_actions=()):
        self.task_id = str(task_id)
        self.obs, self.info = self.backend.reset(self.task_id)
        if self.info["task_id"] != self.task_id:
            raise RuntimeError("Simulator reset returned a different task")
        self.steps, self.done = 0, False
        self.info["truncated"] = False
        history = []
        for action in prefix_actions:
            history.append({"text_obs": self.obs, "action": action})
            self.step(action)
            # Never silently train on an already terminal prefix.
            if self.done:
                raise ValueError(f"Replay reached a terminal state for {self.task_id}; prefix is not recoverable")
        info = deepcopy(self.info)
        info["prefix_history"] = history
        return self.obs, info

    def step(self, action):
        if self.done:
            return self.obs, 0.0, True, deepcopy(self.info)
        self.obs, terminal, self.info = self.backend.step(action)
        if self.info["task_id"] != self.task_id:
            raise RuntimeError("Simulator changed task identity during rollout")
        self.steps += 1
        self.info["truncated"] = bool(self.info.get("truncated", False) or (not terminal and self.steps >= self.max_steps))
        self.done = terminal or self.steps >= self.max_steps
        score = float(self.info["task_score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid normalized task score: {score}")
        # Pay the final score once. Intermediate/prefix progress is not summed
        # again, and terminal failures with partial progress retain a signal.
        reward = (score if self.reward_mode == "score" else float(self.info["won"])) if self.done else 0.0
        self.info["environment_steps"] = self.steps
        return self.obs, reward, self.done, deepcopy(self.info)

    def close(self):
        self.backend.close()


class TaskWorkerGroup:
    """Share immutable WebShop products/index while keeping browser sessions separate."""

    def __init__(self, benchmark, options, max_steps, reward_mode, expected_fingerprint, slots):
        self.slots = []
        server = None
        for slot in range(slots):
            backend = None
            if benchmark == "webshop":
                backend = WebShopBackend(options, server=server, session_prefix=f"slot{slot}_")
                if slot == 0:
                    server = backend.env.server
                else:
                    backend.catalog_fingerprint = self.slots[0].backend.catalog_fingerprint
            self.slots.append(TaskWorker(benchmark, options, max_steps, reward_mode,
                                         expected_fingerprint, backend=backend))

    def call_many(self, method, requests):
        return [getattr(self.slots[slot], method)(*args) for slot, args in requests]

    def close(self):
        for slot in self.slots:
            slot.close()


class RayTaskEnvs:
    def __init__(self, manifest, split, num_processes, resources, max_steps, reward_mode,
                 per_type_limit=None, backend_overrides=None, expected_tasks=None):
        import ray
        self.ray = ray
        self.max_steps = int(max_steps)
        self.backend_options = {**manifest["backend_options"], **(backend_overrides or {})}
        identity = {"manifest": manifest, "max_steps": max_steps, "reward_mode": reward_mode}
        if backend_overrides:
            identity["backend_overrides"] = backend_overrides
        self.task_pool_fingerprint = fingerprint(identity)
        rows = manifest["splits"][split]
        self.full_num_games = len(rows)
        if per_type_limit is not None:
            if split == "train" or manifest["benchmark"] != "scienceworld":
                raise ValueError("Per-type evaluation selection is only supported for ScienceWorld dev/test")
            rows = select_tasks(rows, per_type_limit)
        self.task_ids = tuple(row["task_id"] for row in rows)
        self.task_types = {row["task_id"]: row["task_type"] for row in rows}
        self.allowed_ids = set(self.task_ids)
        self.num_games = len(self.task_ids)
        if expected_tasks is not None and self.num_games != expected_tasks:
            raise ValueError(f"Evaluation protocol expects {expected_tasks} tasks, found {self.num_games}; check simulator/splits")
        self.num_processes = int(num_processes)
        self.active_processes = 0
        self.slots_per_worker = 16 if manifest["benchmark"] == "webshop" else 1
        actor = ray.remote(**resources)(TaskWorkerGroup)
        self.workers = [actor.remote(manifest["benchmark"], self.backend_options, max_steps,
                                     reward_mode, manifest.get("catalog_fingerprint"),
                                     min(self.slots_per_worker, self.num_processes - start))
                        for start in range(0, self.num_processes, self.slots_per_worker)]
        self.current_ids = [None] * self.num_processes

    def _dispatch(self, method, indices, arguments):
        grouped = {}
        for position, (index, args) in enumerate(zip(indices, arguments)):
            if not 0 <= index < self.active_processes:
                raise IndexError(f"Inactive environment index: {index}")
            actor, slot = divmod(index, self.slots_per_worker)
            grouped.setdefault(actor, []).append((position, slot, args))
        futures = [self.workers[actor].call_many.remote(method, [(slot, args) for _, slot, args in requests])
                   for actor, requests in grouped.items()]
        results = [None] * len(indices)
        for requests, replies in zip(grouped.values(), self.ray.get(futures)):
            for (position, _, _), reply in zip(requests, replies):
                results[position] = reply
        return results

    def reset(self, kwargs):
        if kwargs is None:
            raise ValueError("Task-suite resets require explicit task_id entries")
        items = list(kwargs)
        if not 0 < len(items) <= self.num_processes:
            raise ValueError("Reset batch exceeds environment capacity")
        self.active_processes = len(items)
        return self.reset_selected(list(range(len(items))), items)

    def reset_selected(self, indices, kwargs):
        if len(indices) != len(kwargs):
            raise ValueError("Reset indices and kwargs differ in length")
        arguments = []
        for index, item in zip(indices, kwargs):
            task_id = str(item["task_id"])
            if task_id not in self.allowed_ids:
                raise ValueError(f"Task {task_id!r} does not belong to this split")
            self.current_ids[index] = task_id
            arguments.append((task_id, item.get("prefix_actions", [])))
        results = self._dispatch("reset", indices, arguments)
        for index, (_, info) in zip(indices, results):
            if info["task_id"] != self.current_ids[index]:
                raise RuntimeError("Worker task identity mismatch")
        return [r[0] for r in results], [r[1] for r in results]

    def step_selected(self, indices, actions):
        if len(indices) != len(actions) or len(set(indices)) != len(indices):
            raise ValueError("Invalid selected action batch")
        results = self._dispatch("step", indices, [(action,) for action in actions])
        return tuple([r[column] for r in results] for column in range(4))

    def step(self, actions):
        if len(actions) != self.active_processes:
            raise ValueError("Action count differs from active environment count")
        return self.step_selected(list(range(self.active_processes)), actions)

    def close(self):
        try:
            self.ray.get([w.close.remote() for w in self.workers])
        finally:
            for worker in self.workers:
                self.ray.kill(worker)
