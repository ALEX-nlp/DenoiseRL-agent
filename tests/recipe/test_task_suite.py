"""CPU contract tests using real adapter/collector methods and fake simulators.

Load optional-dependency modules in an isolated namespace; no Ray/JVM/GPU is
needed. Native smoke_env tests are a separate, explicit integration check.
"""
import ast
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
from typing import Optional
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from recipe.denoise_v2.gamefile_curriculum import TaskTypePoolCurriculum
from recipe.denoise_v2.task_suite.launch import build_overrides, resolve_checkpoint
from agent_system.alfworld_evaluation import build_gamefile_reset_kwargs, validate_gamefile_coverage
from agent_system.scienceworld_protocol import render_prompt
from agent_system.webshop_protocol import WEBSHOP_DATA_PROFILES, webshop_data_profile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "agent_system/environments/env_package/task_suite"
package = types.ModuleType("_denoise_test_task_suite")
package.__path__ = [str(PACKAGE)]
sys.modules[package.__name__] = package


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


backends = load_module(package.__name__ + ".backends", PACKAGE / "backends.py")
runtime = load_module(package.__name__ + ".runtime", PACKAGE / "runtime.py")
webshop_projection = load_module("agent_system.environments.env_package.webshop.projection",
                                 PACKAGE.parent / "webshop/projection.py").webshop_projection
webshop_prompts = load_module("_test_webshop_prompts", ROOT / "agent_system/environments/prompts/webshop.py")


def load_definitions(path, namespace):
    tree = ast.parse(path.read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    # Postpone annotations to avoid importing optional torch/Ray types.
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + definitions, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


base_ns = load_definitions(ROOT / "agent_system/environments/base.py", {"defaultdict": defaultdict, "np": np})
memory_ns = load_definitions(ROOT / "agent_system/memory/memory.py", {"BaseMemory": object})
manager_ns = load_definitions(PACKAGE / "manager.py", {
    "EnvironmentManagerBase": base_ns["EnvironmentManagerBase"], "SimpleMemory": memory_ns["SimpleMemory"],
    "np": np, "re": __import__("re"), "render_prompt": render_prompt,
    "WEBSHOP_TEMPLATE": webshop_prompts.WEBSHOP_TEMPLATE,
    "WEBSHOP_TEMPLATE_NO_HIS": webshop_prompts.WEBSHOP_TEMPLATE_NO_HIS,
})
Manager = manager_ns["TaskEnvironmentManager"]
Collector = load_definitions(ROOT / "recipe/denoise_v2/collector.py", {
    "TrajectoryCollector": object, "np": np, "uuid": uuid, "json": json, "Path": Path,
    "defaultdict": defaultdict, "TaskTypePoolCurriculum": TaskTypePoolCurriculum,
})["DenoiseTrajectoryCollector"]


class Config(dict):
    __getattr__ = dict.__getitem__


def collector(enabled=True, rho=0.0, rollouts=16):
    c = Collector.__new__(Collector)
    c.enabled, c.v2_enabled, c.mode = enabled, True, "online"
    c.online_prefix_strategy = "full_then_ratio"
    c.main_rollout_n, c.sub_rollout_k = (0, rollouts) if enabled else (rollouts, 0)
    c.online_prefix_candidates_per_group = 1
    c.online_prefix_rng = np.random.default_rng(0)
    c.online_prefix_ratio = 0.3
    c.online_avoid_terminal_prefix = True
    c.online_max_prefix_steps = None
    c.online_full_rollout_max_steps = 4
    c.v2_cfg = {"initial_rho": rho, "max_rho": 0.3 if enabled else 0, "alpha": 0.1 if enabled else 0}
    c.config = Config(env=Config(seed=0, rollout=Config(n=rollouts)), data=Config(train_batch_size=1),
                      algorithm=Config(filter_groups=Config(enable=False)))
    c.configure_v2(["a", "b", "c"], {i: "family" for i in "abc"}, reset_key="task_id")
    return c


class FakeBackend:
    catalog_fingerprint = "catalog"
    def reset(self, task_id):
        self.task_id, self.moves = task_id, 0
        return "initial", self.info()
    def info(self):
        return {"task_id": self.task_id, "task_description": "test task", "admissible_actions": ["advance"],
                "task_score": self.moves / 4, "won": self.moves == 4, "is_action_valid": True}
    def step(self, action):
        self.moves += 1
        return f"state {self.moves}", self.moves == 4, self.info()
    def close(self):
        pass


class Remote:
    def __init__(self, target):
        self.target = target
    def __getattr__(self, name):
        return types.SimpleNamespace(remote=getattr(self.target, name))


def manager(capacity=16, max_steps=4):
    vector = runtime.RayTaskEnvs.__new__(runtime.RayTaskEnvs)
    vector.ray = types.SimpleNamespace(get=lambda results: results)
    group = runtime.TaskWorkerGroup.__new__(runtime.TaskWorkerGroup)
    group.slots = [runtime.TaskWorker("webshop", {}, max_steps, "score", backend=FakeBackend()) for _ in range(capacity)]
    vector.workers = [Remote(group)]
    vector.slots_per_worker = capacity
    vector.num_processes = capacity
    vector.allowed_ids = set("abc")
    vector.current_ids = [None] * capacity
    vector.task_types = {i: "family" for i in "abc"}
    cfg = Config(env=Config(history_length=2, task_suite=Config(benchmark="webshop")))
    return Manager(vector, cfg)


class RuntimeTests(unittest.TestCase):
    def test_webshop_binary_training_reward_preserves_native_score(self):
        for max_steps, expected in ((4, [0, 0, 0, 10, 0]), (3, [0, 0, 0, 0])):
            with self.subTest(max_steps=max_steps):
                worker = runtime.TaskWorker("webshop", {}, max_steps, "success", backend=FakeBackend(), success_reward=10)
                worker.reset("a")
                self.assertEqual([worker.step("advance")[1] for _ in range(max_steps + 1)], expected)
                self.assertEqual(worker.info["task_score"], max_steps / 4)
        # Replay uses the same total budget and cannot pay prefix progress twice.
        worker.reset("a", ["advance", "advance"])
        self.assertEqual(worker.step("advance")[1], 0)

    def test_terminal_score_paid_once_and_done_is_absorbing(self):
        worker = runtime.TaskWorker("webshop", {}, 4, "score", backend=FakeBackend())
        worker.reset("a")
        rewards = [worker.step("advance")[1] for _ in range(5)]
        self.assertEqual(rewards, [0, 0, 0, 1, 0])
        self.assertEqual(worker.steps, 4)
        self.assertFalse(worker.info["truncated"])

    def test_prefix_progress_is_not_double_counted(self):
        worker = runtime.TaskWorker("scienceworld", {}, 4, "score", backend=FakeBackend())
        _, info = worker.reset("a", ["advance", "advance"])
        self.assertEqual(len(info["prefix_history"]), 2)
        self.assertEqual(worker.steps, 2)
        self.assertEqual(worker.step("advance")[1], 0)
        self.assertEqual(worker.step("advance")[1], 1)

    def test_step_budget_counts_replayed_actions_and_retains_partial_score(self):
        worker = runtime.TaskWorker("scienceworld", {}, 3, "score", backend=FakeBackend())
        worker.reset("a", ["advance", "advance"])
        _, reward, done, info = worker.step("advance")
        self.assertTrue(done)
        self.assertFalse(info["won"])
        self.assertEqual(reward, .75)
        self.assertTrue(info["truncated"])
        self.assertTrue(worker.step("advance")[3]["truncated"])
        self.assertFalse(worker.reset("b")[1]["truncated"])

    def test_terminal_prefix_rejected(self):
        worker = runtime.TaskWorker("webshop", {}, 4, "score", backend=FakeBackend())
        with self.assertRaisesRegex(ValueError, "terminal"):
            worker.reset("a", ["advance"] * 4)

    def test_manifest_rejects_overlap_between_splits(self):
        manifest = {"version": 1, "benchmark": "webshop", "splits": {
            split: [{"task_id": "a", "task_type": "family"}] for split in ("train", "dev", "test")}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                runtime.load_manifest(path, "webshop")

    def test_selected_replay_restores_history_without_changing_other_envs(self):
        m = manager(2)
        m.reset([{"task_id": "a"}, {"task_id": "b"}])
        m.step_selected([0], ["<action>advance</action>"])
        obs, infos = m.reset_selected_with_prefixes([0], [["advance", "advance"]])
        self.assertEqual(m.pre_text_obs, ["state 2", "initial"])
        self.assertEqual([len(h) for h in m.memory._data], [2, 0])
        self.assertIn("already taken 2 step(s)", obs["text"][0])
        self.assertEqual(infos[0]["task_id"], "a")
        m.reset_selected_with_prefixes([], [])
        with self.assertRaisesRegex(ValueError, "split"):
            m.envs.reset([{"task_id": "test-only"}])

    def test_action_parser_preserves_object_names_rejects_multiple_actions(self):
        actions, valid = manager_ns["project_actions"]([
            "<think>x</think><action>focus on Red Box</action>",
            "<action>look</action><action>move</action>", "bad"])
        self.assertEqual(actions[0], "focus on Red Box")
        self.assertEqual(valid, [True, False, False])

    def test_webshop_uses_upstream_projection_without_mutating_responses(self):
        m = manager(5)
        m.reset([{"task_id": "a"}] * 5)
        outputs = ["<think>Choose</think><action>click[RED Box]</action>",
                   "<action>advance</action>", "<think>中文</think><action>advance</action>",
                   "<think>x</think><action>first</action><action>second</action>", "malformed response without tags"]
        original = list(outputs)
        # As in GiGPO, validity concerns model format, independent of native action legality.
        for slot in m.envs.workers[0].target.slots:
            backend = slot.backend
            native_info = backend.info
            backend.info = lambda native_info=native_info: {**native_info(), "is_action_valid": False}
        _, _, _, infos = m.step(outputs)
        self.assertEqual(outputs, original)
        self.assertEqual([h[-1]["action"] for h in m.memory._data],
                         ["click[red box]", "advance", "advance", "first", original[-1][-20:]])
        self.assertEqual([i["is_action_valid"] for i in infos], [True, False, False, True, False])

    def test_webshop_prompt_matches_gigpo_with_replay_history_and_length_fallback(self):
        m = manager(1)
        m.reset([{"task_id": "a"}])
        task = "buy a red mug"
        raw = lambda page: f"WebShop [SEP] Instruction: [SEP] {task} [SEP] {page}"
        m.pre_text_obs = [raw("Search")]
        m.webshop_tasks = [task]
        m.infos[0]["admissible_actions"] = ["search[<query>]", "click[Search]"]
        initial = webshop_prompts.WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=task, current_observation="'Search'",
            available_actions="'search[<your query>]',\n'click[Search]',")
        self.assertEqual(m.build_mixed_text_obs_after_prefix()[0], initial)
        m.memory._data[0] = [{"text_obs": raw(page), "action": action} for page, action in
                             [("Search", "search[mug]"), ("Results", "click[item]"), ("Item", "click[red]")]]
        m.pre_text_obs = [raw("Item [SEP] Red selected")]
        expected = webshop_prompts.WEBSHOP_TEMPLATE.format(
            task_description=task, current_observation="'Item' [SEP] 'Red selected'",
            available_actions="'search[<your query>]',\n'click[Search]',", step_count=3,
            history_length=2, current_step=4,
            action_history="[Observation 2: ''Results'', Action 2: 'click[item]']\n"
                           "[Observation 3: ''Item'', Action 3: 'click[red]']")
        self.assertEqual(m.build_mixed_text_obs_after_prefix()[0], expected)
        m.memory._data[0][-1]["text_obs"] = "x" * 14000
        prompt = m.build_mixed_text_obs_after_prefix()[0]
        self.assertNotIn("Prior to this step", prompt)
        self.assertIn("'Red selected'", prompt)
        # Terminal screens can omit the task; retain the task extracted at reset.
        m.pre_text_obs = ["Your score (min 0.0, max 1.0): 1.0"]
        self.assertIn(f"Your task is to: {task}.", m.build_mixed_text_obs_after_prefix()[0])

    def test_selected_dispatch_preserves_order_across_actor_groups(self):
        m = manager(4)
        slots = m.envs.workers[0].target.slots
        groups = []
        for start in (0, 2):
            group = runtime.TaskWorkerGroup.__new__(runtime.TaskWorkerGroup)
            group.slots = slots[start:start + 2]
            groups.append(Remote(group))
        m.envs.workers, m.envs.slots_per_worker = groups, 2
        m.reset([{"task_id": key} for key in "abca"])
        _, _, _, infos = m.step_selected([2, 0], ["<action>advance</action>"] * 2)
        self.assertEqual([info["task_id"] for info in infos], ["c", "a"])
        self.assertEqual(m.pre_text_obs, ["state 1", "initial", "state 1", "initial"])

    def test_shared_task_replay_detects_simulator_nondeterminism(self):
        info = {"task_id": "a", "task_description": "task", "task_score": 0, "admissible_actions": ["look"]}
        with self.assertRaisesRegex(RuntimeError, "different states"):
            Manager._verify_shared_states(["state one", "state two"], [info, info], [["look"], ["look"]])


class CurriculumIntegrationTests(unittest.TestCase):
    def test_shared_prefix_for_all_configured_rollouts(self):
        for n in (8, 16):
            with self.subTest(rollouts=n):
                c, m = collector(rho=.3, rollouts=n), manager(capacity=n)
                kwargs = c._build_env_kwargs(n)
                obs, _ = m.reset(kwargs)
                calls = []
                c._ensure_online_ready = lambda: None
                def generate(batch, current_obs, indices):
                    calls.append(list(indices))
                    return ["<action>advance</action>"] * len(indices)
                c._generate_denoise_actions = generate
                batch = types.SimpleNamespace(batch=list(range(n)))
                _, metrics = c._run_full_then_ratio_prefixes(batch, obs, m, kwargs)
                self.assertEqual(calls, [[0]] * 4)
                self.assertEqual(list(metrics["denoise_prefix_len"]), [2] * n)
                self.assertEqual(m.pre_text_obs, ["state 2"] * n)
                self.assertTrue(all(len(history) == 2 for history in m.memory._data))

    def test_eight_rollout_update_is_per_trajectory_and_rejects_missing_samples(self):
        for enabled in (True, False):
            c = collector(enabled, rho=.1 if enabled else 0, rollouts=8)
            active = c.v2_curriculum.active_problem_ids[0]
            kwargs = c._build_env_kwargs(8)
            self.assertEqual([item["task_id"] for item in kwargs], [active] * 8)
            self.assertEqual([item["denoise_is_sub"] for item in kwargs], [enabled] * 8)
            # Unequal trajectory lengths must not change the 4/8 success rate.
            repeats = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
            count = int(repeats.sum())
            batch = types.SimpleNamespace(non_tensor_batch={
                "traj_uid": np.repeat([f"traj{i}" for i in range(8)], repeats),
                "denoise_v2_problem_id": np.asarray([active] * count),
                "denoise_v2_task_type": np.asarray(["family"] * count),
                "episode_success": np.repeat([1, 1, 1, 1, 0, 0, 0, 0], repeats),
            })
            c.after_training_step(batch)
            self.assertAlmostEqual(c.v2_curriculum.mean_rho(), .075 if enabled else 0)
            self.assertNotEqual(c.v2_curriculum.active_problem_ids, (active,))
        c = collector(rollouts=8)
        active = c.v2_curriculum.active_problem_ids[0]
        missing = types.SimpleNamespace(non_tensor_batch={
            "traj_uid": np.asarray([f"traj{i}" for i in range(7)]),
            "denoise_v2_problem_id": np.asarray([active] * 7),
            "denoise_v2_task_type": np.asarray(["family"] * 7), "episode_success": np.ones(7),
        })
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            c.after_training_step(missing)

    def test_group_size_checkpoint_compatibility(self):
        old, new = collector(rollouts=16), collector(rollouts=8)
        state = old.v2_state_dict()
        with self.assertRaisesRegex(ValueError, "Rollouts per task differ"):
            new.load_v2_state_dict(state)
        del state["rollouts_per_task"]
        old.load_v2_state_dict(state)  # Legacy 16-sample checkpoints still resume.
        with self.assertRaisesRegex(ValueError, "Rollouts per task differ"):
            new.load_v2_state_dict(state)
        restored = collector(rollouts=8)
        restored.load_v2_state_dict(new.v2_state_dict())
        self.assertEqual(restored.v2_state_dict(), new.v2_state_dict())

    def test_invalid_group_layout_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            collector(rollouts=1)
        c = collector(rollouts=8)
        c.main_rollout_n, c.sub_rollout_k = 4, 4
        with self.assertRaisesRegex(ValueError, "all sharing one prefix"):
            c.configure_v2(["a"], {"a": "family"})

    def test_zero_rho_skips_shadow_and_second_reset(self):
        c, m = collector(), manager()
        kwargs = c._build_env_kwargs(16)
        obs, _ = m.reset(kwargs)
        c._ensure_online_ready = lambda: None
        c._generate_denoise_actions = lambda *args: self.fail("zero rho generated weak-model actions")
        _, metrics = c._run_full_then_ratio_prefixes(types.SimpleNamespace(batch=list(range(16))), obs, m, kwargs)
        self.assertEqual(list(metrics["denoise_prefix_len"]), [0] * 16)
        self.assertEqual(m.pre_text_obs, ["initial"] * 16)

    def test_baseline_pins_same_tasks_and_never_increases_rho(self):
        baseline, denoise = collector(False), collector(True)
        self.assertEqual(baseline.v2_curriculum.active_problem_ids, denoise.v2_curriculum.active_problem_ids)
        kwargs = baseline._build_env_kwargs(16)
        self.assertFalse(any(item["denoise_is_sub"] for item in kwargs))
        active = baseline.v2_curriculum.active_problem_ids[0]
        batch = types.SimpleNamespace(non_tensor_batch={
            "traj_uid": np.repeat([f"traj{i}" for i in range(16)], 2),
            "denoise_v2_problem_id": np.asarray([active] * 32),
            "denoise_v2_task_type": np.asarray(["family"] * 32), "episode_success": np.ones(32),
        })
        baseline.after_training_step(batch)
        self.assertEqual(baseline.v2_curriculum.mean_rho(), 0)
        self.assertNotEqual(baseline.v2_curriculum.active_problem_ids, (active,))

    def test_resume_rejects_changed_environment_and_restores_rng(self):
        first, second = collector(), collector()
        first.v2_environment_fingerprint = "data1"
        second.v2_environment_fingerprint = "data2"
        state = first.v2_state_dict()
        with self.assertRaisesRegex(ValueError, "differs"):
            second.load_v2_state_dict(state)
        second.v2_environment_fingerprint = "data1"
        second.load_v2_state_dict(state)
        self.assertEqual(first.online_prefix_rng.integers(1000), second.online_prefix_rng.integers(1000))

    def test_validation_pins_partial_batch_and_repeats(self):
        kwargs = build_gamefile_reset_kwargs(("a", "b", "c"), start=2, count=1, repeats=2, reset_key="task_id")
        self.assertEqual([item["task_id"] for item in kwargs], ["c", "c"])
        validate_gamefile_coverage(("a", "b"), ["a", "b", "a", "b"], repeats=2)
        with self.assertRaises(RuntimeError):
            validate_gamefile_coverage(("a", "b"), ["a", "a"], repeats=1)

    def test_full_webshop_test_covers_all_500_tasks_including_last_four(self):
        task_ids = tuple(str(i) for i in range(500))
        batches = [build_gamefile_reset_kwargs(task_ids, start=start, count=min(16, 500 - start),
                                               repeats=1, reset_key="task_id") for start in range(0, 500, 16)]
        self.assertEqual(len(batches[-1]), 4)
        visited = [item["task_id"] for batch in batches for item in batch]
        self.assertEqual(visited, list(task_ids))
        validate_gamefile_coverage(task_ids, visited, repeats=1)


class NativeAdapterTests(unittest.TestCase):
    def test_webshop_group_shares_catalog_with_distinct_sessions(self):
        original = runtime.WebShopBackend
        class Backend(FakeBackend):
            def __init__(self, options, server=None, session_prefix=None):
                self.env = types.SimpleNamespace(server=server if server is not None else object())
                self.session_prefix = session_prefix
        runtime.WebShopBackend = Backend
        try:
            group = runtime.TaskWorkerGroup("webshop", {}, 4, "score", "catalog", 2)
            self.assertIs(group.slots[0].backend.env.server, group.slots[1].backend.env.server)
            self.assertNotEqual(group.slots[0].backend.session_prefix, group.slots[1].backend.session_prefix)
            group.call_many("reset", [(0, ("a",)), (1, ("a",))])
            group.call_many("step", [(0, ("advance",))])
            self.assertEqual(group.slots[1].steps, 0)
        finally:
            runtime.WebShopBackend = original

    def test_webshop_native_split_and_category_mapping(self):
        backend = backends.WebShopBackend.__new__(backends.WebShopBackend)
        backend.rho_grouping = "category"
        backend.data_profile = "full_human"
        backend.goals = [{"category": "books"}] * 1600
        backend.goals[1501] = {"category": "electronics"}
        backend.goals[1503] = {}
        splits = backend.catalog()
        self.assertEqual([len(splits[s]) for s in ("train", "dev", "test")], [100, 1000, 500])
        self.assertEqual(splits["train"][0], {"task_id": "1500", "task_type": "books"})
        self.assertEqual(splits["train"][2]["task_type"], "books")
        self.assertEqual(splits["train"][1]["task_type"], "electronics")
        self.assertEqual(splits["train"][3]["task_type"], "shopping")

    def test_small_catalog_uses_synthetic_goals_and_matching_index(self):
        env_module = types.ModuleType("web_agent_site.envs")
        env_module.WebAgentTextEnv = unittest.mock.Mock()
        env_module.WebAgentTextEnv.return_value.server.goals = [{"category": "books"}] * 1600
        with patch.dict(sys.modules, {"web_agent_site.envs": env_module}):
            for profile, human in (("gigpo_small", False), ("full_human", True)):
                backend = backends.WebShopBackend({"data_profile": profile, "file_path": "products",
                                                   "attr_path": "attrs", "search_index_path": "/matching-index"})
                kwargs = env_module.WebAgentTextEnv.call_args.kwargs
                self.assertEqual(kwargs["human_goals"], human)
                self.assertEqual(kwargs["seed"], 42)
                self.assertIsNone(kwargs["num_products"])
                self.assertEqual(kwargs["search_index_path"], "/matching-index")
                if not human:
                    splits = backend.catalog()
                    self.assertEqual([len(splits[s]) for s in ("train", "dev", "test")], [1100, 0, 500])
                    self.assertEqual(splits["train"][0]["task_id"], "500")
                    self.assertEqual(splits["test"][-1]["task_id"], "499")

    def test_small_manifest_requires_exact_goal_partition_and_no_dev(self):
        manifest = {"version": 1, "benchmark": "webshop", "backend_options": {"data_profile": "gigpo_small"},
                    "splits": {"train": [{"task_id": str(i), "task_type": "shopping"} for i in range(500, 520)],
                               "test": [{"task_id": str(i), "task_type": "shopping"} for i in range(500)], "dev": []}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(runtime.load_manifest(path, "webshop"), manifest)
            manifest["splits"]["dev"].append(manifest["splits"]["train"].pop(0))
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "GiGPO small requires"):
                runtime.load_manifest(path, "webshop")

    def test_native_server_opens_explicit_index_without_truncating_products(self):
        engine_ns = load_definitions(PACKAGE.parent / "webshop/webshop/web_agent_site/engine/engine.py", {
            "os": os, "BASE_DIR": "/native/engine", "LuceneSearcher": unittest.mock.Mock(),
        })
        server_path = PACKAGE.parent / "webshop/webshop/web_agent_site/envs/web_agent_text_env.py"
        tree = ast.parse(server_path.read_text())
        server_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SimServer")
        # Only the constructor is needed; Flask route decorators are unrelated.
        server_class.body = [node for node in server_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"]
        products = [{"asin": "a"}]
        load_products = unittest.mock.Mock(return_value=(products, {}, {}, {}))
        get_goals = unittest.mock.Mock(return_value=[])
        ns = {"load_products": load_products, "get_goals": get_goals, "random": __import__("random"), "np": np,
              "init_search_engine": engine_ns["init_search_engine"]}
        exec(compile(ast.Module(body=[server_class], type_ignores=[]), str(server_path), "exec"), ns)
        ns["SimServer"](42, "local", "products.json", "attrs.json", human_goals=False,
                        search_index_path="/small-index")
        engine_ns["LuceneSearcher"].assert_called_once_with("/small-index")
        load_products.assert_called_once_with(filepath="products.json", attrpath="attrs.json",
                                              num_products=None, human_goals=False)
        get_goals.assert_called_once_with(products, {}, False)

    def test_webshop_six_structural_groups_and_boundaries(self):
        cases = [
            (0, 1, "options_0__attrs_1_2"),
            (0, 2, "options_0__attrs_1_2"),
            (0, 3, "options_0__attrs_3_plus"),
            (1, 2, "options_1__attrs_1_2"),
            (1, 3, "options_1__attrs_3_plus"),
            (2, 2, "options_2_plus__attrs_1_2"),
            (2, 3, "options_2_plus__attrs_3_plus"),
            (4, 5, "options_2_plus__attrs_3_plus"),
        ]
        observed = set()
        for options, attributes, expected in cases:
            for option_values in (list(range(options)), dict.fromkeys(range(options), "value")):
                with self.subTest(options=options, attributes=attributes, format=type(option_values)):
                    goal = {"goal_options": option_values, "attributes": ["attribute"] * attributes}
                    observed.add(backends.webshop_task_type(goal, "structure"))
                    self.assertEqual(backends.webshop_task_type(goal, "structure"), expected)
        self.assertEqual(observed, set(backends.WEBSHOP_STRUCTURE_GROUPS))
        self.assertEqual(len(observed), 6)

    def test_webshop_structure_requires_valid_annotations(self):
        for goal in ({}, {"attributes": [], "goal_options": []},
                     {"attributes": "waterproof", "goal_options": []},
                     {"attributes": ["waterproof"]},
                     {"attributes": ["waterproof"], "goal_options": "blue"}):
            with self.subTest(goal=goal), self.assertRaises(ValueError):
                backends.webshop_task_type(goal, "structure")
        with self.assertRaisesRegex(ValueError, "Unknown WebShop rho grouping"):
            backends.WebShopBackend({"rho_grouping": "typo"})

    def test_webshop_structural_rho_shared_across_categories_and_checkpoint_guarded(self):
        goals = {
            "a": {"category": "fashion", "attributes": ["waterproof"], "goal_options": ["blue"]},
            "b": {"category": "electronics", "attributes": ["portable"], "goal_options": ["black"]},
            "c": {"category": "fashion", "attributes": ["waterproof"], "goal_options": []},
        }
        groups = {key: backends.webshop_task_type(goal, "structure") for key, goal in goals.items()}
        curriculum = TaskTypePoolCurriculum(list(goals), groups, batch_size=3, initial_rho=.2)
        curriculum.update({"a": 1.0, "b": 0.0, "c": 1.0})
        self.assertAlmostEqual(curriculum.rho_for_problem("a"), .15)
        self.assertAlmostEqual(curriculum.rho_for_problem("b"), .15)
        self.assertAlmostEqual(curriculum.rho_for_problem("c"), .25)
        previous = TaskTypePoolCurriculum(list(goals), {key: goal["category"] for key, goal in goals.items()},
                                          batch_size=3, initial_rho=.2)
        with self.assertRaisesRegex(ValueError, "task types"):
            previous.load_state_dict(curriculum.state_dict())

    def test_scienceworld_variations_share_task_type_without_merging_task_names(self):
        backend = backends.ScienceWorldBackend.__new__(backends.ScienceWorldBackend)
        backend.simplifications = ""
        backend.env = types.SimpleNamespace(
            get_task_names=lambda: ["boil", "melt"],
            load=lambda *args, **kwargs: None,
            get_variations_train=lambda: [0, 1],
            get_variations_dev=lambda: [2],
            get_variations_test=lambda: [3],
        )
        rows = backend.catalog()["train"]
        self.assertEqual([row["task_id"] for row in rows], ["boil::0", "boil::1", "melt::0", "melt::1"])
        self.assertEqual([row["task_type"] for row in rows], ["boil", "boil", "melt", "melt"])

    def test_scienceworld_normalizes_absolute_score_not_delta(self):
        backend = backends.ScienceWorldBackend.__new__(backends.ScienceWorldBackend)
        backend.task_id = "boil::0"
        backend.env = types.SimpleNamespace(get_task_description=lambda: "Boil water")
        info = backend._info({"score": 60, "reward": 10, "valid": ["look around"]})
        self.assertEqual(info["task_score"], .6)
        self.assertFalse(info["won"])
        self.assertEqual(backend._info({"score": -1})["task_score"], 0)


class PreparationTests(unittest.TestCase):
    def test_full_webshop_index_uses_explicit_product_and_attribute_files(self):
        converter = ROOT / "agent_system/environments/env_package/webshop/webshop/search_engine/convert_product_file_format.py"
        utils = types.ModuleType("web_agent_site.utils")
        utils.DEFAULT_FILE_PATH, utils.DEFAULT_ATTR_PATH = "small-products.json", "small-attrs.json"
        engine = types.ModuleType("web_agent_site.engine.engine")
        def load_products(filepath, attrpath):
            self.assertEqual((filepath, attrpath), ("full-products.json", "full-attrs.json"))
            return ([{"asin": "full-only-product", "Title": "Title", "Description": "Description",
                      "BulletPoints": ["Feature"], "options": {"color": ["Blue"]}}],)
        engine.load_products = load_products
        tqdm = types.ModuleType("tqdm")
        tqdm.tqdm = lambda items, **kwargs: items
        argv = [str(converter), "--file-path", "full-products.json", "--attr-path", "full-attrs.json"]
        with tempfile.TemporaryDirectory() as directory:
            # Building in a separate work directory must not overwrite an
            # existing resources directory used by an earlier index.
            (Path(directory) / "resources").mkdir()
            old_docs = Path(directory) / "resources/documents.jsonl"
            old_docs.write_text("previous-documents")
            output_root = Path(directory) / "staging"
            original_cwd = Path.cwd()
            try:
                os.chdir(directory)
                with patch.object(sys, "argv", argv + ["--output-root", str(output_root)]), patch.object(sys, "path", list(sys.path)), patch.dict(sys.modules, {
                    "web_agent_site.utils": utils, "web_agent_site.engine.engine": engine, "tqdm": tqdm,
                }):
                    runpy.run_path(str(converter), run_name="__main__")
                document = json.loads((output_root / "resources/documents.jsonl").read_text())
                self.assertEqual(document["id"], "full-only-product")
                self.assertIn("color: blue", document["contents"])
                self.assertEqual(old_docs.read_text(), "previous-documents")
            finally:
                os.chdir(original_cwd)

    def test_grouping_cli_defaults_to_structure_and_preserves_task_splits(self):
        # Exercise the real CLI, catalog and manifest writer without native
        # WebShop assets or parquet dependencies. Only the simulators/IO are fake.
        def make_backend(benchmark, options):
            self.assertEqual(benchmark, "webshop")
            backend = backends.WebShopBackend.__new__(backends.WebShopBackend)
            backend.rho_grouping = options["rho_grouping"]
            backend.data_profile = options["data_profile"]
            backend.goals = [{"category": "fashion" if i % 2 else "electronics",
                              "attributes": ["portable"], "goal_options": []} for i in range(1600)]
            backend.catalog_fingerprint = backends.fingerprint(backend.goals)
            backend.close = lambda: None
            return backend

        prepare = load_definitions(ROOT / "recipe/denoise_v2/task_suite/prepare_tasks.py", {
            "argparse": argparse, "Counter": Counter, "json": json, "Path": Path,
            "__file__": str(ROOT / "recipe/denoise_v2/task_suite/prepare_tasks.py"),
            "WEBSHOP_DATA_PROFILES": WEBSHOP_DATA_PROFILES, "webshop_data_profile": webshop_data_profile,
            "WEBSHOP_RHO_GROUPINGS": backends.WEBSHOP_RHO_GROUPINGS,
            "WEBSHOP_STRUCTURE_GROUPS": backends.WEBSHOP_STRUCTURE_GROUPS,
            "make_backend": make_backend, "fingerprint": backends.fingerprint,
            "load_manifest": runtime.load_manifest,
        })["main"]
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.Dataset = types.SimpleNamespace(from_dict=lambda rows: types.SimpleNamespace(
            to_parquet=lambda path: Path(path).write_text("placeholder")))
        manifests = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("items_shuffle.json", "items_ins_v2.json", "items_human_ins.json"):
                (root / filename).write_text("{}")
            for name in ("items_shuffle_1000.json", "items_ins_v2_1000.json"):
                (root / name).write_text("{}")
            for data_profile, train_count, dev_count in (("full_human", 100, 1000), ("gigpo_small", 1100, 0)):
                for mode, extra in (("structure", []), ("category", ["--webshop-rho-grouping", "category"])):
                    output_dir = root / data_profile / mode
                    args = ["prepare_tasks", "--benchmark", "webshop", "--output", str(output_dir),
                            "--webshop-data-dir", str(root), "--webshop-data-profile", data_profile] + extra
                    output = io.StringIO()
                    with patch.object(sys, "argv", args), patch.dict(sys.modules, {"datasets": fake_datasets}), redirect_stdout(output):
                        prepare()
                    manifest = runtime.load_manifest(output_dir / "tasks.json", "webshop")
                    manifests[mode] = manifest
                    options = manifest["backend_options"]
                    self.assertEqual(options["rho_grouping"], mode)
                    self.assertEqual(Path(options["file_path"]).name, WEBSHOP_DATA_PROFILES[data_profile]["products"])
                    self.assertEqual(Path(options["search_index_path"]).name, WEBSHOP_DATA_PROFILES[data_profile]["index"])
                    self.assertEqual((output_dir / "dev.parquet").exists(), bool(dev_count))
                    report = json.loads(output.getvalue())
                    self.assertEqual(report["tasks"], {"train": train_count, "dev": dev_count, "test": 500})
                    if mode == "structure":
                        counts = report["task_type_counts"]["train"]
                        self.assertEqual(set(counts), set(backends.WEBSHOP_STRUCTURE_GROUPS))
                        self.assertEqual(counts["options_0__attrs_1_2"], train_count)
                        self.assertEqual(sum(counts.values()), train_count)
                    else:
                        self.assertEqual(report["task_type_counts"]["train"], {"electronics": train_count // 2, "fashion": train_count // 2})
                for split in ("train", "dev", "test"):
                    self.assertEqual([row["task_id"] for row in manifests["structure"]["splits"][split]],
                                     [row["task_id"] for row in manifests["category"]["splits"][split]])
                self.assertEqual(manifests["structure"]["catalog_fingerprint"], manifests["category"]["catalog_fingerprint"])
                self.assertNotEqual(backends.fingerprint(manifests["structure"]), backends.fingerprint(manifests["category"]))


class LauncherTests(unittest.TestCase):
    def args(self, benchmark="webshop", method="baseline", mode="train", **extra):
        return types.SimpleNamespace(benchmark=benchmark, method=method, mode=mode, data_dir=None,
                                     seed=0, eval_split=None, checkpoint=None, base_model=False, **extra)
    def values(self, args):
        return {key: json.loads(value) for key, value in (arg.split("=", 1) for arg in build_overrides(args))}

    def test_webshop_defaults_to_full_test_and_gigpo_training_reward(self):
        for method in ("baseline", "denoise"):
            for mode in ("train", "eval"):
                args = self.args(method=method, mode=mode)
                args.base_model = True
                values = self.values(args)
                self.assertEqual(values["env.task_suite.eval_split"], "test")
                self.assertEqual(values["env.task_suite.eval_expected_tasks"], 500)
                self.assertTrue(values["data.val_files"].endswith("/test.parquet"))
                self.assertTrue(values["data.train_files"].endswith("/train.parquet"))
                self.assertEqual(values["env.task_suite.reward_mode"], "success" if mode == "train" else "score")
                self.assertEqual(values["env.task_suite.success_reward"], 10)
                self.assertEqual(values["trainer.keep_latest_and_best"], mode == "train")
                self.assertEqual(values["trainer.best_checkpoint_metric"], "val/test/success_rate")

    def test_small_profile_changes_only_data_paths_metadata_and_run_identity(self):
        for method in ("baseline", "denoise"):
            small = self.values(self.args(method=method, webshop_data_profile="gigpo_small"))
            full = self.values(self.args(method=method, webshop_data_profile="full_human"))
            allowed = {"data.train_files", "data.val_files", "env.task_suite.manifest_path",
                       "env.task_suite.webshop_data_profile", "env.webshop.use_small", "env.webshop.human_goals",
                       "trainer.experiment_name", "trainer.default_local_dir", "trainer.rollout_data_dir",
                       "trainer.validation_data_dir"}
            self.assertTrue({key for key in small if small[key] != full[key]} <= allowed)
            self.assertIn("/webshop_gigpo_small/tasks.json", small["env.task_suite.manifest_path"])
            self.assertEqual(small["env.task_suite.webshop_data_profile"], "gigpo_small")
            self.assertFalse(small["env.webshop.human_goals"])
            self.assertTrue(small["env.webshop.use_small"])
        args = self.args()
        args.eval_split = "dev"
        with self.assertRaisesRegex(ValueError, "no dev split"):
            self.values(args)

    @unittest.skipUnless(importlib.util.find_spec("hydra"), "Hydra is an optional test dependency")
    def test_webshop_validation_keeps_native_score_reward_and_full_task_pool(self):
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(ROOT / "recipe/denoise_v2/config"), version_base=None):
            cfg = compose(config_name="task_suite_trainer", overrides=build_overrides(self.args()))
        with patch.dict(manager_ns, {"load_manifest": lambda *args: {"backend_options": {"data_profile": "gigpo_small"}}, "RayTaskEnvs": unittest.mock.Mock()}):
            train, val = manager_ns["make_task_envs"](cfg)
            train_call, val_call = manager_ns["RayTaskEnvs"].call_args_list
        self.assertEqual(train_call.args[1], "train")
        self.assertEqual(train_call.args[5], "success")
        self.assertEqual(train_call.kwargs["success_reward"], 10)
        self.assertEqual(val_call.args[1], "test")
        self.assertEqual(val_call.args[5], "score")
        self.assertIsNone(val_call.kwargs["per_type_limit"])
        self.assertEqual(val_call.kwargs["expected_tasks"], 500)
        self.assertEqual(set(val), {"test"})
        with patch.dict(manager_ns, {"load_manifest": lambda *args: {"backend_options": {}},
                                     "RayTaskEnvs": unittest.mock.Mock()}):
            with self.assertRaisesRegex(ValueError, "profile mismatch"):
                manager_ns["make_task_envs"](cfg)
            manager_ns["RayTaskEnvs"].assert_not_called()
        # Native partial credit is still the reported validation score, despite reward=0 in training.
        worker = runtime.TaskWorker("webshop", {}, 3, val_call.args[5], backend=FakeBackend())
        worker.reset("a")
        self.assertEqual(sum(worker.step("advance")[1] for _ in range(3)), .75)
    def test_baseline_and_denoise_have_equal_rollout_and_reward_budgets(self):
        for benchmark in ("webshop", "scienceworld"):
            baseline = self.values(self.args(benchmark))
            denoise = self.values(self.args(benchmark, "denoise"))
            for key in ("env.rollout.n", "data.train_batch_size", "env.max_steps", "env.task_suite.reward_mode",
                        "actor_rollout_ref.model.path", "actor_rollout_ref.actor.optim.lr", "trainer.total_training_steps"):
                self.assertEqual(baseline[key], denoise[key])
            self.assertIsNone(baseline["env.denoise.online.model_path"])
            self.assertEqual(baseline["env.denoise.v2.max_rho"], 0)
            self.assertEqual(denoise["env.rollout.n"], 8 if benchmark == "scienceworld" else 16)
            # Denoise uses the shared ALFWorld controller, not benchmark overrides.
            for name in ("initial_rho", "min_rho", "max_rho", "target_accuracy", "alpha"):
                self.assertNotIn(f"env.denoise.v2.{name}", denoise)

    def test_eval_requires_checkpoint_or_explicit_base_model(self):
        args = self.args(mode="eval")
        with self.assertRaisesRegex(ValueError, "requires"):
            self.values(args)
        args.base_model = True
        values = self.values(args)
        self.assertFalse(values["env.denoise.enable"])
        self.assertFalse(values["env.denoise.v2.enabled"])
        self.assertEqual(values["trainer.resume_mode"], "disable")

    def test_checkpoint_resolution_validates_tracker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "global_step_25/actor").mkdir(parents=True)
            (root / "latest_checkpointed_iteration.txt").write_text("25\n")
            self.assertEqual(resolve_checkpoint(root), str(root / "global_step_25"))
            self.assertEqual(resolve_checkpoint(root / "global_step_25/actor"), str(root / "global_step_25"))
            (root / "latest_checkpointed_iteration.txt").write_text("30\n")
            with self.assertRaises(ValueError):
                resolve_checkpoint(root)

    @unittest.skipUnless(importlib.util.find_spec("hydra"), "Hydra is an optional test dependency")
    def test_all_profiles_compose_with_the_actual_hydra_schema(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        import os
        old_cwd = Path.cwd()
        try:
            os.chdir(ROOT)
            with initialize_config_dir(config_dir=str(ROOT / "recipe/denoise_v2/config"), version_base=None):
                alfworld = compose(config_name="denoise_v2_trainer")
                OmegaConf.resolve(alfworld)
            for benchmark in ("webshop", "scienceworld"):
                for method in ("baseline", "denoise"):
                    for mode in ("train", "eval"):
                        args = self.args(benchmark, method, mode)
                        args.base_model = True
                        with initialize_config_dir(config_dir=str(ROOT / "recipe/denoise_v2/config"), version_base=None):
                            cfg = compose(config_name="task_suite_trainer", overrides=build_overrides(args))
                            OmegaConf.resolve(cfg)
                            self.assertEqual(cfg.env.rollout.n, 8 if benchmark == "scienceworld" else 16)
                            self.assertEqual(cfg.env.denoise.enable, method == "denoise" and mode == "train")
                            self.assertEqual(cfg.env.task_suite.benchmark, benchmark)
                            self.assertEqual(cfg.data.train_batch_size, 16)
                            if benchmark == "webshop":
                                sampling = cfg.actor_rollout_ref.rollout.val_kwargs
                                self.assertTrue(sampling.do_sample)
                                self.assertEqual(sampling.temperature, 0.6)
                                self.assertEqual(sampling.top_p, 0.95)
                                self.assertEqual(sampling.top_k, -1)
                            if method == "denoise" and mode == "train":
                                for name in ("initial_rho", "min_rho", "max_rho", "target_accuracy", "alpha"):
                                    self.assertEqual(cfg.env.denoise.v2[name], alfworld.env.denoise.v2[name])
        finally:
            os.chdir(old_cwd)

    @unittest.skipUnless(importlib.util.find_spec("hydra"), "Hydra is an optional test dependency")
    def test_webshop_eval_is_independent_of_training_sampling_overrides(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        for method in ("baseline", "denoise"):
            for mode, split in (("train", "dev"), ("eval", "dev"), ("eval", "test")):
                for do_sample in (True, False):
                    with self.subTest(method=method, mode=mode, split=split, do_sample=do_sample):
                        args = self.args(method=method, mode=mode)
                        args.base_model, args.eval_split = True, split
                        args.webshop_data_profile = "full_human" if split == "dev" else "gigpo_small"
                        sampling = {"do_sample": do_sample, "temperature": 0.7, "top_p": 0.9, "top_k": 40}
                        overrides = build_overrides(args) + [
                            f"actor_rollout_ref.rollout.{key}={json.dumps(value)}"
                            for key, value in sampling.items()
                        ]
                        with initialize_config_dir(config_dir=str(ROOT / "recipe/denoise_v2/config"), version_base=None):
                            cfg = compose(config_name="task_suite_trainer", overrides=overrides)
                        resolved = OmegaConf.to_container(cfg.actor_rollout_ref.rollout.val_kwargs, resolve=True)
                        self.assertEqual({key: resolved[key] for key in sampling}, {
                            "do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": -1,
                        })
                        self.assertEqual(resolved["n"], 1)


if __name__ == "__main__":
    unittest.main()
