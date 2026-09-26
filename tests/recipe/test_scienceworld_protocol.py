"""Protocol and launch-pipeline contracts, without requiring Ray, Java or GPUs."""

from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent_system.scienceworld_protocol import bounded_chat, render_prompt, score_metrics, select_tasks, task_digest, write_report
from recipe.denoise_v2.task_suite import launch
from tests.recipe.test_task_suite import backends, runtime, Manager, Config, memory_ns, Collector


def values(overrides):
    return {key: json.loads(value) for key, value in (item.split("=", 1) for item in overrides if "=" in item)}


class SelectionTests(unittest.TestCase):
    def test_numeric_first_variations_and_short_types(self):
        rows = [{"task_id": f"a::{i}", "task_type": "a"} for i in [101, 99, 100, 102]]
        rows += [{"task_id": "b::7", "task_type": "b"}]
        selected = select_tasks(rows, 3)
        self.assertEqual([row["task_id"] for row in selected], ["a::99", "a::100", "a::101", "b::7"])
        self.assertEqual(selected, select_tasks(list(reversed(rows)), 3))
        self.assertEqual(len(select_tasks(rows)), len(rows))
        with self.assertRaises(ValueError):
            select_tasks(rows, 0)

    def test_paper_counts_and_macro_average(self):
        # Appendix A: short test splits are retained; large ones contribute 10.
        counts = [9, 9, 9, 9, 100, 100, 100, 5, 5, 100, 100, 100, 100, 100, 100,
                  100, 100, 8, 9, 9, 100, 100, 100, 5, 4, 100, 9, 100, 100, 100]
        rows = [{"task_id": f"type{t}::{i}", "task_type": f"type{t}"} for t, n in enumerate(counts) for i in range(n)]
        selected = select_tasks(rows, 10)
        self.assertEqual(len(selected), 270)
        self.assertEqual(len(Counter(row["task_type"] for row in selected)), 30)
        metrics = score_metrics(["a::0", "a::1", "b::0"], [1, 1, 0], {"a::0": "a", "a::1": "a", "b::0": "b"})
        self.assertAlmostEqual(metrics["score"], 200 / 3)
        self.assertEqual(metrics["score_macro"], 50)
        self.assertEqual(metrics["task_type/a/score"], 100)
        self.assertNotEqual(task_digest(["a", "b"]), task_digest(["b", "a"]))

    def test_progress_does_not_claim_completion_early(self):
        with tempfile.TemporaryDirectory() as directory:
            write_report(directory, 25, {"complete": False, "completed_episodes": 8})
            self.assertFalse((Path(directory) / "25.summary.json").exists())
            write_report(directory, 25, {"complete": True, "completed_episodes": 89})
            self.assertEqual(json.loads((Path(directory) / "progress.json").read_text()),
                             json.loads((Path(directory) / "25.summary.json").read_text()))

    def test_runtime_keeps_training_pool_and_selects_validation_before_spawning(self):
        rows = [{"task_id": f"a::{i}", "task_type": "a"} for i in range(12)]
        manifest = {"benchmark": "scienceworld", "backend_options": {"simplifications": ""},
                    "splits": {"train": rows, "dev": rows, "test": rows}}
        ray = SimpleNamespace(remote=Mock(return_value=lambda cls: SimpleNamespace(remote=Mock())))
        with patch.dict(sys.modules, {"ray": ray}):
            train = runtime.RayTaskEnvs(manifest, "train", 0, {}, 100, "score")
            dev = runtime.RayTaskEnvs(manifest, "dev", 0, {}, 600, "score", per_type_limit=3,
                                      backend_overrides={"simplifications": "easy"})
            self.assertEqual(len(train.task_ids), 12)
            self.assertEqual(dev.task_ids, ("a::0", "a::1", "a::2"))
            self.assertEqual(dev.full_num_games, 12)
            self.assertEqual(dev.max_steps, 600)
            self.assertEqual(dev.backend_options["simplifications"], "easy")
            self.assertEqual(manifest["backend_options"]["simplifications"], "")
            ray.remote.reset_mock()
            with self.assertRaisesRegex(ValueError, "expects 270"):
                runtime.RayTaskEnvs(manifest, "test", 8, {}, 600, "score", per_type_limit=10, expected_tasks=270)
            ray.remote.assert_not_called()


class PromptTests(unittest.TestCase):
    def test_trimming_preserves_task_current_state_and_action_format(self):
        class Tokenizer:
            def apply_chat_template(self, chat, **kwargs):
                return "<user>" + chat[0]["content"] + "<assistant>"
            def encode(self, text, **kwargs):
                return list(text)
        parts = {"task": "Melt the ice", "steps": 12, "templates": "heat OBJ", "state": "ice in freezer",
                 "history": ["old " * 3000, "recent move\n"]}
        budget = len(render_prompt(parts, history=[])) + 100
        chat, rendered = bounded_chat(parts, Tokenizer(), budget, {})
        self.assertLessEqual(len(rendered), budget)
        self.assertIn("Melt the ice", rendered)
        self.assertIn("ice in freezer", rendered)
        self.assertIn("<action>command</action>", rendered)
        self.assertIn("recent move", rendered)
        self.assertNotIn("old old", rendered)
        self.assertEqual(chat[0]["role"], "user")
        self.assertEqual(len(parts["history"]), 2)
        with self.assertRaisesRegex(ValueError, "refusing to silently truncate"):
            bounded_chat(parts, Tokenizer(), 5, {})

    def test_manager_uses_templates_and_prefix_subsets_keep_prompt_parts(self):
        manager = Manager.__new__(Manager)
        manager.config = Config(env=Config(task_suite=Config(benchmark="scienceworld"), history_length=4))
        manager.memory = memory_ns["SimpleMemory"]()
        manager.memory.reset(1)
        manager.pre_text_obs = ["kitchen"]
        manager.infos = [{"task_description": "Boil water", "action_templates": ["heat OBJ"],
                          "admissible_actions": ["a giant enumerated action"] * 10000}]
        obs = manager._observations()
        self.assertIn("Boil water", obs["text"][0])
        self.assertNotIn("giant", obs["text"][0])
        collector = Collector.__new__(Collector)
        subset = collector._subset_obs(obs, [0])
        self.assertEqual(subset["scienceworld_prompt_parts"], obs["scienceworld_prompt_parts"])
        self.assertEqual(collector._current_obs_from_envs(manager, None), obs)


class ScoreTests(unittest.TestCase):
    def backend(self, scores, stagnant=False):
        backend = backends.ScienceWorldBackend.__new__(backends.ScienceWorldBackend)
        scores = iter(scores)
        backend.env = SimpleNamespace(step=lambda action: ("obs", 0, False, {"score": next(scores), "valid": ["look around"]}),
                                      get_task_description=lambda: "Task")
        backend.task_id = "type::0"
        backend.score_mode = "last_nonnegative"
        backend.last_nonnegative_score = 0
        backend.previous_actions = {"look around"}
        backend.recent_deltas = [0.0]
        backend.options = {"stop_on_stagnation": stagnant}
        return backend

    def test_irrecoverable_failure_retains_last_not_best_score(self):
        backend = self.backend([60, 40, -1])
        worker = runtime.TaskWorker("scienceworld", {}, 10, "score", backend=backend)
        worker.task_id, worker.done, worker.steps = "type::0", False, 0
        self.assertEqual(worker.step("look around")[1], 0)
        self.assertEqual(worker.step("look around")[1], 0)
        _, reward, done, info = worker.step("look around")
        self.assertTrue(done)
        self.assertEqual(reward, .4)
        self.assertFalse(info["won"])
        self.assertEqual(info["raw_score"], -1)
        self.assertEqual(worker.step("look around")[1], 0)

    def test_stagnation_rule_and_disabled_training_rule(self):
        for enabled in [True, False]:
            backend = self.backend([20] * 100, stagnant=enabled)
            for _ in range(98):
                self.assertFalse(backend.step("look around")[1])
            _, done, info = backend.step("look around")
            self.assertEqual(done, enabled)
            self.assertEqual(info["task_score"], .2)


class LaunchPipelineTests(unittest.TestCase):
    def args(self, mode="train", **updates):
        result = dict(benchmark="scienceworld", method="baseline", mode=mode, data_dir=None,
                      seed=0, eval_split=None, checkpoint=None, base_model=True, eval_protocol="swiftsage")
        result.update(updates)
        return SimpleNamespace(**result)

    def test_monitor_and_final_protocols_are_distinct(self):
        train = values(launch.build_overrides(self.args()))
        self.assertFalse(train["trainer.val_before_train"])
        self.assertEqual(train["env.task_suite.eval_split"], "dev")
        self.assertEqual(train["env.task_suite.eval_per_type_limit"], 3)
        final = values(launch.build_overrides(self.args("eval")))
        self.assertTrue(final["trainer.val_before_train"])
        self.assertEqual(final["env.task_suite.eval_split"], "test")
        self.assertEqual(final["env.task_suite.eval_expected_tasks"], 270)
        self.assertEqual(final["env.task_suite.eval_max_steps"], 600)
        self.assertEqual(final["env.task_suite.eval_env_step_limit"], 300)
        full = values(launch.build_overrides(self.args("eval", eval_protocol="full", eval_split="dev")))
        self.assertIsNone(full["env.task_suite.eval_per_type_limit"])
        with self.assertRaisesRegex(ValueError, "reserved"):
            launch.build_overrides(self.args(eval_split="test"))

    def test_success_runs_final_saved_checkpoint_and_failure_never_runs_test(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "global_step_2"
            (checkpoint / "actor").mkdir(parents=True)
            calls = []
            def run(command, **kwargs):
                calls.append((command, kwargs))
                config = values(command)
                if len(calls) == 1:
                    Path(config["trainer.completion_path"]).write_text(json.dumps({
                        "checkpoint": str(checkpoint), "manifest_path": directory + "/tasks.json",
                        "experiment_name": "trained", "validation_data_dir": directory + "/validation"}))
            argv = ["launch", "--benchmark", "scienceworld", "--method", "baseline", "trainer.total_training_steps=2"]
            with patch.object(sys, "argv", argv), patch.object(launch.subprocess, "run", side_effect=run), redirect_stdout(io.StringIO()):
                launch.main()
            self.assertEqual(len(calls), 2)
            final = values(calls[1][0])
            self.assertEqual(final["trainer.resume_from_path"], str(checkpoint.resolve()))
            self.assertEqual(final["env.task_suite.eval_split"], "test")
            self.assertTrue(final["trainer.val_only"])
            self.assertFalse(final["env.denoise.v2.enabled"])
            self.assertIsNone(final["env.denoise.online.model_path"])
            self.assertEqual(final["trainer.experiment_name"], "trained_test_swiftsage")
            self.assertNotEqual(final["trainer.validation_data_dir"], directory + "/validation")
            with patch.object(sys, "argv", argv), patch.object(launch.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "train")) as failed, redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    launch.main()
                self.assertEqual(failed.call_count, 1)
            with patch.object(sys, "argv", argv), patch.object(launch.subprocess, "run") as empty, redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "stale checkpoint"):
                    launch.main()
                self.assertEqual(empty.call_count, 1)


if __name__ == "__main__":
    unittest.main()
