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
})
Manager = manager_ns["TaskEnvironmentManager"]
Collector = load_definitions(ROOT / "recipe/denoise_v2/collector.py", {
    "TrajectoryCollector": object, "np": np, "uuid": uuid, "json": json, "Path": Path,
    "defaultdict": defaultdict, "TaskTypePoolCurriculum": TaskTypePoolCurriculum,
})["DenoiseTrajectoryCollector"]


class Config(dict):
    __getattr__ = dict.__getitem__


def collector(enabled=True, rho=0.0):
    c = Collector.__new__(Collector)
    c.enabled, c.v2_enabled, c.mode = enabled, True, "online"
    c.online_prefix_strategy = "full_then_ratio"
    c.main_rollout_n, c.sub_rollout_k = (0, 16) if enabled else (16, 0)
    c.online_prefix_candidates_per_group = 1
    c.online_prefix_rng = np.random.default_rng(0)
    c.online_prefix_ratio = 0.3
    c.online_avoid_terminal_prefix = True
    c.online_max_prefix_steps = None
    c.online_full_rollout_max_steps = 4
    c.v2_cfg = {"initial_rho": rho, "max_rho": 0.3 if enabled else 0, "alpha": 0.1 if enabled else 0}
    c.config = Config(env=Config(seed=0, rollout=Config(n=16)), data=Config(train_batch_size=1),
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
    def test_terminal_score_paid_once_and_done_is_absorbing(self):
        worker = runtime.TaskWorker("webshop", {}, 4, "score", backend=FakeBackend())
        worker.reset("a")
        rewards = [worker.step("advance")[1] for _ in range(5)]
        self.assertEqual(rewards, [0, 0, 0, 1, 0])
        self.assertEqual(worker.steps, 4)

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
        self.assertIn("Actions already taken: 2", obs["text"][0])
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
    def test_shared_prefix_for_all_16_rollouts(self):
        c, m = collector(rho=.3), manager()
        kwargs = c._build_env_kwargs(16)
        obs, _ = m.reset(kwargs)
        calls = []
        c._ensure_online_ready = lambda: None
        def generate(batch, current_obs, indices):
            calls.append(list(indices))
            return ["<action>advance</action>"] * len(indices)
        c._generate_denoise_actions = generate
        batch = types.SimpleNamespace(batch=list(range(16)))
        _, metrics = c._run_full_then_ratio_prefixes(batch, obs, m, kwargs)
        self.assertEqual(calls, [[0]] * 4)
        self.assertEqual(list(metrics["denoise_prefix_len"]), [2] * 16)
        self.assertEqual(m.pre_text_obs, ["state 2"] * 16)
        self.assertTrue(all(len(history) == 2 for history in m.memory._data))

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
        backend.goals = [{"category": "books"}] * 1600
        backend.goals[1501] = {"category": "electronics"}
        backend.goals[1503] = {}
        splits = backend.catalog()
        self.assertEqual([len(splits[s]) for s in ("train", "dev", "test")], [100, 1000, 500])
        self.assertEqual(splits["train"][0], {"task_id": "1500", "task_type": "books"})
        self.assertEqual(splits["train"][2]["task_type"], "books")
        self.assertEqual(splits["train"][1]["task_type"], "electronics")
        self.assertEqual(splits["train"][3]["task_type"], "shopping")

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
            backend.goals = [{"category": "fashion" if i % 2 else "electronics",
                              "attributes": ["portable"], "goal_options": []} for i in range(1600)]
            backend.catalog_fingerprint = backends.fingerprint(backend.goals)
            backend.close = lambda: None
            return backend

        prepare = load_definitions(ROOT / "recipe/denoise_v2/task_suite/prepare_tasks.py", {
            "argparse": argparse, "Counter": Counter, "json": json, "Path": Path,
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
            for mode, extra in (("structure", []), ("category", ["--webshop-rho-grouping", "category"])):
                args = ["prepare_tasks", "--benchmark", "webshop", "--output", str(root / mode),
                        "--webshop-data-dir", str(root)] + extra
                output = io.StringIO()
                with patch.object(sys, "argv", args), patch.dict(sys.modules, {"datasets": fake_datasets}), redirect_stdout(output):
                    prepare()
                manifest = runtime.load_manifest(root / mode / "tasks.json", "webshop")
                manifests[mode] = manifest
                self.assertEqual(manifest["backend_options"]["rho_grouping"], mode)
                report = json.loads(output.getvalue())
                self.assertEqual(report["tasks"], {"train": 100, "dev": 1000, "test": 500})
                if mode == "structure":
                    counts = report["task_type_counts"]["train"]
                    self.assertEqual(set(counts), set(backends.WEBSHOP_STRUCTURE_GROUPS))
                    self.assertEqual(counts["options_0__attrs_1_2"], 100)
                    self.assertEqual(sum(counts.values()), 100)
                else:
                    self.assertEqual(report["task_type_counts"]["train"], {"electronics": 50, "fashion": 50})
            for split in ("train", "dev", "test"):
                self.assertEqual([row["task_id"] for row in manifests["structure"]["splits"][split]],
                                 [row["task_id"] for row in manifests["category"]["splits"][split]])
            self.assertEqual(manifests["structure"]["catalog_fingerprint"], manifests["category"]["catalog_fingerprint"])
            self.assertNotEqual(backends.fingerprint(manifests["structure"]), backends.fingerprint(manifests["category"]))


class LauncherTests(unittest.TestCase):
    def args(self, benchmark="webshop", method="baseline", mode="train", **extra):
        return types.SimpleNamespace(benchmark=benchmark, method=method, mode=mode, data_dir=None,
                                     seed=0, eval_split="dev", checkpoint=None, base_model=False, **extra)
    def values(self, args):
        return {key: json.loads(value) for key, value in (arg.split("=", 1) for arg in build_overrides(args))}
    def test_baseline_and_denoise_have_equal_rollout_and_reward_budgets(self):
        for benchmark in ("webshop", "scienceworld"):
            baseline = self.values(self.args(benchmark))
            denoise = self.values(self.args(benchmark, "denoise"))
            for key in ("env.rollout.n", "data.train_batch_size", "env.max_steps", "env.task_suite.reward_mode",
                        "actor_rollout_ref.model.path", "actor_rollout_ref.actor.optim.lr", "trainer.total_training_steps"):
                self.assertEqual(baseline[key], denoise[key])
            self.assertIsNone(baseline["env.denoise.online.model_path"])
            self.assertEqual(baseline["env.denoise.v2.max_rho"], 0)
            self.assertEqual(denoise["env.rollout.n"], 16)
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
                            self.assertEqual(cfg.env.rollout.n, 16)
                            self.assertEqual(cfg.env.denoise.enable, method == "denoise" and mode == "train")
                            self.assertEqual(cfg.env.task_suite.benchmark, benchmark)
                            self.assertEqual(cfg.data.train_batch_size, 16)
                            if method == "denoise" and mode == "train":
                                for name in ("initial_rho", "min_rho", "max_rho", "target_accuracy", "alpha"):
                                    self.assertEqual(cfg.env.denoise.v2[name], alfworld.env.denoise.v2[name])
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
