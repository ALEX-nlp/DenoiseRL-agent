"""Exercise tracking failures without requiring W&B, Ray, or GPU dependencies."""

import importlib.util
import logging
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch


def load_tracking():
    path = Path(__file__).resolve().parents[3] / "verl/utils/tracking.py"
    spec = importlib.util.spec_from_file_location("_test_tracking", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: module}):
        spec.loader.exec_module(module)
    return module


tracking = load_tracking()


class UsageError(Exception):
    pass


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.wandb = types.ModuleType("wandb")
        self.wandb.errors = types.SimpleNamespace(UsageError=UsageError)
        self.wandb.init = Mock()
        self.wandb.log = Mock()
        self.wandb.finish = Mock()
        self.console = Mock()
        console_module = types.ModuleType("verl.utils.logger.aggregate_logger")
        console_module.LocalLogger = Mock(return_value=self.console)
        self.modules = patch.dict(sys.modules, {
            "wandb": self.wandb,
            console_module.__name__: console_module,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_missing_api_key_retries_offline_and_keeps_logging(self):
        for message in (
            "No API key configured. Use `wandb login` to log in.",
            "api_key not configured (no-tty). call wandb.login(key=[your_api_key])",
            "API key not configured: test-only-private-value",
        ):
            with self.subTest(message=message), patch.dict(os.environ, {"WANDB_MODE": "online"}, clear=True):
                self.wandb.init.reset_mock()
                self.wandb.init.side_effect = [UsageError(message), None]
                config = {"trainer": {"total_training_steps": 500}}
                with self.assertLogs(tracking.__name__, level="WARNING") as output:
                    logger = tracking.Tracking("project", "experiment", ["console", "wandb"], config)
                self.assertEqual(self.wandb.init.call_args_list, [
                    call(project="project", name="experiment", config=config),
                    call(project="project", name="experiment", config=config, mode="offline"),
                ])
                self.assertIn("offline", " ".join(output.output))
                self.assertIn("wandb sync", " ".join(output.output))
                self.assertNotIn("test-only-private-value", " ".join(output.output))
                self.assertEqual(os.environ["WANDB_MODE"], "online")
                logger.log({"train/reward": 1.0}, step=7)
                self.wandb.log.assert_called_with(data={"train/reward": 1.0}, step=7)
                self.console.log.assert_called_with(data={"train/reward": 1.0}, step=7)
                del logger

    def test_successful_init_preserves_requested_mode(self):
        for mode in ("online", "offline", "disabled"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"WANDB_MODE": mode}, clear=True):
                self.wandb.init.reset_mock()
                logger = tracking.Tracking("project", "experiment", "wandb")
                self.wandb.init.assert_called_once_with(project="project", name="experiment", config=None)
                self.assertEqual(os.environ["WANDB_MODE"], mode)
                del logger

    def test_other_failures_are_not_hidden_or_leaked_to_warning(self):
        for error in (
            UsageError("Invalid project: test-only-secret"),
            UsageError("API key must be 40 characters long"),
            RuntimeError("service unavailable"),
        ):
            with self.subTest(error=error), patch.object(logging.getLogger(tracking.__name__), "warning") as warning:
                self.wandb.init.reset_mock()
                self.wandb.init.side_effect = error
                with self.assertRaises(type(error)) as raised:
                    tracking.Tracking("project", "experiment", "wandb")
                self.assertIs(raised.exception, error)
                self.wandb.init.assert_called_once()
                warning.assert_not_called()

    def test_offline_failure_is_propagated_without_more_retries(self):
        error = OSError("offline directory is not writable")
        self.wandb.init.side_effect = [UsageError("No API key configured"), error]
        with self.assertLogs(tracking.__name__, level="WARNING"), self.assertRaises(OSError) as raised:
            tracking.Tracking("project", "experiment", "wandb")
        self.assertIs(raised.exception, error)
        self.assertEqual(self.wandb.init.call_count, 2)
        self.wandb.finish.assert_not_called()

    def test_console_only_does_not_initialize_wandb(self):
        logger = tracking.Tracking("project", "experiment", "console")
        logger.log({"train/reward": 1.0}, step=7)
        self.console.log.assert_called_once_with(data={"train/reward": 1.0}, step=7)
        self.wandb.init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
