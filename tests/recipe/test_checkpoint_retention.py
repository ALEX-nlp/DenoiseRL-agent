"""Exercise real save hooks with filesystem-backed fake workers, without Ray/GPU."""

import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from recipe.denoise_v2.checkpoint_retention import LatestBestCheckpoints


ROOT = Path(__file__).resolve().parents[2]
METRIC = "val/test/success_rate"


def load_class(path, name, namespace, methods=None):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if methods:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


Base = load_class(ROOT / "verl/trainer/ppo/ray_trainer.py", "RayPPOTrainer", {
    "os": os, "torch": SimpleNamespace(save=lambda data, path: Path(path).write_text(json.dumps(data))),
}, methods={"_save_checkpoint", "_maybe_save_checkpoint"})
Trainer = load_class(ROOT / "recipe/denoise_v2/online_ray_trainer.py", "OnlineDenoisePPOTrainer", {
    "RayPPOTrainer": Base, "LatestBestCheckpoints": LatestBestCheckpoints, "Path": Path, "json": json,
})


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def trainer(root, enabled=True):
    instance = Trainer.__new__(Trainer)
    instance.config = SimpleNamespace(trainer=Config(
        default_local_dir=str(root), default_hdfs_dir=None, keep_latest_and_best=enabled,
        best_checkpoint_metric=METRIC, save_freq=25, val_only=False,
        max_actor_ckpt_to_keep=1, max_critic_ckpt_to_keep=1,
    ))
    def save_worker(local_path, remote_path, step, max_ckpt_to_keep):
        path = Path(local_path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "weights.pt").write_text(str(step))
        (path / "optimizer.pt").write_text(str(step))
    instance.actor_rollout_wg = SimpleNamespace(save_checkpoint=Mock(side_effect=save_worker))
    instance.critic_wg = SimpleNamespace(save_checkpoint=Mock(side_effect=save_worker))
    instance.use_critic = True
    instance.train_dataloader = SimpleNamespace(state_dict=lambda: {"position": instance.global_steps})
    instance.traj_collector = SimpleNamespace(v2_enabled=True, v2_state_dict=lambda: {"step": instance.global_steps})
    return instance


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.trainer = trainer(self.root)

    def save(self, step, value=None, final=False, instance=None):
        instance = instance or self.trainer
        instance.global_steps = step
        return instance._maybe_save_checkpoint(None if value is None else {METRIC: value}, is_last_step=final)

    def steps(self):
        return sorted(int(p.name.split("_")[-1]) for p in self.root.glob("global_step_*") if p.is_dir())

    def test_preserves_historical_best_and_latest_complete_resume_state(self):
        self.save(0, .1)
        self.save(25, .6)
        self.save(50, .4)
        self.save(75, .5)
        self.assertEqual(self.steps(), [25, 75])
        self.assertEqual((self.root / "latest_checkpointed_iteration.txt").read_text(), "75")
        self.assertEqual((self.root / "best_checkpointed_iteration.txt").read_text(), "25")
        for step in self.steps():
            for filename in ("actor/weights.pt", "actor/optimizer.pt", "critic/weights.pt", "data.pt", "denoise_v2_curriculum.json"):
                self.assertTrue((self.root / f"global_step_{step}" / filename).is_file())
        for worker in (self.trainer.actor_rollout_wg, self.trainer.critic_wg):
            self.assertTrue(all(call.kwargs["max_ckpt_to_keep"] is None for call in worker.save_checkpoint.call_args_list))
        self.save(100, .7)
        self.assertEqual(self.steps(), [100])

    def test_ties_keep_earlier_best_and_off_schedule_improvement_is_saved(self):
        self.save(25, .5)
        self.save(30, .6)
        self.assertEqual(self.steps(), [30])
        self.save(50, .6)
        self.assertEqual(self.steps(), [30, 50])
        self.assertEqual((self.root / "best_checkpointed_iteration.txt").read_text(), "30")
        self.assertFalse(self.save(51, .4))
        self.assertEqual(self.steps(), [30, 50])

    def test_final_is_saved_even_when_periodic_saving_is_disabled(self):
        self.save(0, .5)
        self.trainer.config.trainer.save_freq = -1
        self.save(7, final=True)
        self.assertEqual(self.steps(), [0, 7])
        state = json.loads((self.root / "checkpoint_retention.json").read_text())
        self.assertEqual(state["best"], {"step": 0, "value": .5})

    def test_restart_restores_best_score_before_rotating(self):
        self.save(25, .8)
        self.save(50, .5)
        restarted = trainer(self.root)
        self.save(75, .6, instance=restarted)
        self.assertEqual(self.steps(), [25, 75])
        self.save(100, .9, instance=restarted)
        self.assertEqual(self.steps(), [100])

    def test_resumed_initial_validation_does_not_rewrite_saved_weights(self):
        self.save(25, .8)
        restarted = trainer(self.root)
        restarted.global_steps = 25
        restarted._maybe_save_checkpoint({METRIC: .9}, initial=True)
        restarted.actor_rollout_wg.save_checkpoint.assert_not_called()
        self.assertEqual(restarted._checkpoint_retention.best, {"step": 25, "value": .9})

    def test_fresh_run_cannot_overwrite_an_existing_experiment(self):
        self.save(25, .8)
        fresh = trainer(self.root)
        fresh.config.trainer.resume_mode = "disable"
        with self.assertRaisesRegex(ValueError, "new EXPERIMENT_NAME"):
            self.save(0, .9, instance=fresh)
        fresh.actor_rollout_wg.save_checkpoint.assert_not_called()
        self.assertEqual(self.steps(), [25])

    def test_failed_curriculum_save_does_not_advance_trackers_or_delete_old_models(self):
        self.save(25, .8)
        self.save(50, .5)
        before = (self.root / "checkpoint_retention.json").read_text()
        self.trainer.traj_collector.v2_state_dict = Mock(side_effect=OSError("disk full"))
        with self.assertRaisesRegex(OSError, "disk full"):
            self.save(75, .9)
        self.assertTrue((self.root / "global_step_25/actor/weights.pt").exists())
        self.assertTrue((self.root / "global_step_50/actor/weights.pt").exists())
        self.assertEqual((self.root / "latest_checkpointed_iteration.txt").read_text(), "50")
        self.assertEqual((self.root / "checkpoint_retention.json").read_text(), before)

    def test_auto_resume_recovers_interrupted_tracker_update_from_committed_state(self):
        self.save(25, .8)
        retention = self.trainer._checkpoint_retention
        write = retention._write_atomic
        def interrupted(path, text):
            if path.name == "latest_checkpointed_iteration.txt":
                raise OSError("interrupted tracker write")
            return write(path, text)
        with patch.object(retention, "_write_atomic", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.save(50, .9)
        self.assertEqual((self.root / "latest_checkpointed_iteration.txt").read_text(), "25")
        restarted = trainer(self.root)
        restarted.config.trainer.resume_mode = "auto"
        restarted.traj_collector.load_v2_state_dict = Mock()
        def load(instance):
            instance.global_steps = int((self.root / "latest_checkpointed_iteration.txt").read_text())
        with patch.object(Base, "_load_checkpoint", load, create=True):
            restarted._load_checkpoint()
        self.assertEqual(restarted.global_steps, 50)
        restarted.traj_collector.load_v2_state_dict.assert_called_once_with({"step": 50})
        self.assertEqual((self.root / "best_checkpointed_iteration.txt").read_text(), "50")

    def test_missing_and_nonfinite_metric_cannot_replace_best(self):
        self.save(25, .8)
        self.trainer.global_steps = 50
        with self.assertRaisesRegex(ValueError, "missing"):
            self.trainer._maybe_save_checkpoint({"training/reward": 10})
        for step, value in ((50, float("nan")), (75, float("inf"))):
            self.save(step, value)
            self.assertEqual((self.root / "best_checkpointed_iteration.txt").read_text(), "25")
        self.assertEqual(self.steps(), [25, 75])

    def test_does_not_follow_symlinks_or_remove_other_outputs(self):
        outside = self.root / "other_run"
        outside.mkdir()
        (outside / "weights.pt").write_text("keep")
        (self.root / "global_step_999").symlink_to(outside, target_is_directory=True)
        (self.root / "global_step_backup").mkdir()
        self.save(25, .8)
        self.save(50, .9)
        self.assertTrue((outside / "weights.pt").exists())
        self.assertTrue((self.root / "global_step_999").is_symlink())
        self.assertTrue((self.root / "global_step_backup").is_dir())

    def test_existing_best_missing_or_metric_changed_fails_before_save(self):
        self.save(25, .8)
        with self.assertRaisesRegex(ValueError, "metric changed"):
            LatestBestCheckpoints(self.root, "val/test/test_score")
        (self.root / "global_step_25/data.pt").unlink()
        restarted = trainer(self.root)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.save(50, .9, instance=restarted)
        restarted.actor_rollout_wg.save_checkpoint.assert_not_called()

    def test_disabled_policy_preserves_legacy_save_schedule_and_fifo(self):
        instance = trainer(self.root, enabled=False)
        self.save(0, .5, instance=instance)
        self.save(24, .8, instance=instance)
        instance.actor_rollout_wg.save_checkpoint.assert_not_called()
        self.save(25, .8, instance=instance)
        self.assertEqual(instance.actor_rollout_wg.save_checkpoint.call_args.kwargs["max_ckpt_to_keep"], 1)
        self.assertEqual((self.root / "latest_checkpointed_iteration.txt").read_text(), "25")
        self.assertFalse((self.root / "checkpoint_retention.json").exists())

    def test_eval_only_never_saves_or_prunes(self):
        self.trainer.config.trainer.val_only = True
        self.save(0, .8)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
