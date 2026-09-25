"""Installer isolation and index recovery contracts; never runs an installer."""

from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from recipe.denoise_v2.task_suite import environment_check as checks
from recipe.denoise_v2.task_suite import prepare_webshop_assets as assets
from recipe.denoise_v2.task_suite import setup_environment as setup


class InstallationIsolationTests(unittest.TestCase):
    def invoke(self, argv, prefixes, run):
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["setup_environment"] + argv))
            stack.enter_context(patch.object(setup.platform, "system", return_value="Linux"))
            stack.enter_context(patch.object(setup.platform, "machine", return_value="x86_64"))
            stack.enter_context(patch.dict(os.environ, {"CONDA_EXE": "/mock/conda"}))
            stack.enter_context(patch.object(setup, "environment_prefix", side_effect=prefixes))
            stack.enter_context(patch.object(setup, "run", side_effect=run))
            stack.enter_context(redirect_stdout(io.StringIO()))
            setup.main()

    def test_source_and_existing_environments_cannot_be_install_targets(self):
        for name in ("base", "molu", "source", "../molu", "/tmp/new", "--clone"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                setup.validate_name(name, "source")
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            with self.assertRaisesRegex(ValueError, "Refusing"):
                self.invoke(["--benchmark", "webshop"], [prefix],
                            lambda *args, **kwargs: self.fail("Changed an existing environment"))
            with self.assertRaisesRegex(ValueError, "Refusing"):
                self.invoke(["--benchmark", "webshop", "--resume"], [prefix],
                            lambda *args, **kwargs: self.fail("Resumed an unowned environment"))

    def test_clone_reads_source_but_installs_only_into_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "molu", Path(directory) / "denoise-webshop"
            source.mkdir()
            target.mkdir()  # Simulate Conda creating this path.
            commands = []
            self.invoke(["--benchmark", "webshop", "--mode", "clone"], [source, None, target], commands.append)
            self.assertEqual(commands[0][-6:], ["--prefix", str(source), "python", "-m", "pip", "check"])
            self.assertIn("--copy", commands[1])
            self.assertIn("--override-channels", commands[1])
            self.assertIn("http://nexus.sii.shaipower.online/repository/anaconda/pkgs/main", commands[1])
            for command in commands[2:]:
                self.assertEqual(str(command[command.index("--prefix") + 1]), str(target))
                self.assertNotIn(str(source), map(str, command))
            self.assertFalse(any("flash-attn==2.7.4.post1" in c for c in commands))
            java_install = next(c for c in commands if "openjdk=11" in c)
            self.assertIn("http://nexus.sii.shaipower.online/repository/anaconda/cloud/conda-forge", java_install)
            for command in commands:
                if "pip" in command and "install" in command:
                    self.assertEqual(command[command.index("--index-url") + 1],
                                     "http://nexus.sii.shaipower.online/repository/pypi_proxy/simple/")
                    self.assertEqual(command[command.index("--trusted-host") + 1], "nexus.sii.shaipower.online")
                    self.assertEqual(command[command.index("--timeout") + 1], "120")
            self.assertEqual(json.loads((target / setup.MARKER / "setup.json").read_text())["status"],
                             "dependencies_checked")
            self.assertEqual(list(source.iterdir()), [])

    def test_broken_source_aborts_before_cloning(self):
        commands = []
        def fail(command):
            commands.append(command)
            raise subprocess.CalledProcessError(1, command)
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke(["--benchmark", "scienceworld", "--mode", "clone"],
                        [Path("/envs/molu"), None], fail)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][-3:], ["-m", "pip", "check"])

    def test_failed_install_can_resume_only_with_matching_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            def fail_install(command):
                if "install" in command:
                    raise subprocess.CalledProcessError(1, command)
            with self.assertRaises(subprocess.CalledProcessError):
                self.invoke(["--benchmark", "scienceworld"], [None, target], fail_install)
            marker = target / setup.MARKER / "setup.json"
            self.assertEqual(json.loads(marker.read_text())["status"], "failed")
            # Older installers did not record the mirror. Their failed
            # environments must remain resumable when switching download sources.
            saved = json.loads(marker.read_text())
            saved.pop("mirror")
            saved.pop("pip_index_url")
            marker.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "parameters differ"):
                self.invoke(["--benchmark", "webshop", "--resume"], [target],
                            lambda command: self.fail("Modified mismatched environment"))
            commands = []
            self.invoke(["--benchmark", "scienceworld", "--resume", "--mirror", "official"],
                        [target], commands.append)
            self.assertFalse(any("create" in c for c in commands))
            self.assertEqual(json.loads(marker.read_text())["status"], "dependencies_checked")
            self.assertEqual(json.loads(marker.read_text())["mirror"], "official")
            for command in commands:
                self.assertNotIn("tuna.tsinghua.edu.cn", " ".join(map(str, command)))
                self.assertNotIn("nexus.sii.shaipower.online", " ".join(map(str, command)))
                if "pip" in command and "install" in command:
                    self.assertEqual(command[command.index("--index-url") + 1], "https://pypi.org/simple")
                    self.assertNotIn("--trusted-host", command)
            self.assertIn("https://conda.anaconda.org/conda-forge", commands[0])

    def test_pip_redirection_is_removed(self):
        with patch.dict(os.environ, {"PIP_TARGET": "/source", "PYTHONPATH": "/source",
                                     "PIP_PREFIX": "/source", "PYTHONHOME": "/source",
                                     "PIP_INDEX_URL": "https://old-index.invalid/simple/",
                                     "PIP_EXTRA_INDEX_URL": "https://public-index.invalid/simple/",
                                     "PIP_FIND_LINKS": "https://old-wheels.invalid/",
                                     "PIP_TRUSTED_HOST": "old-index.invalid",
                                     "PIP_CONFIG_FILE": "/source/pip.conf", "PIP_NO_INDEX": "1",
                                     "JAVA_HOME": "/other/java", "MAX_JOBS": "4"}):
            env = setup.isolated_env()
        for name in ("PIP_TARGET", "PIP_PREFIX", "PYTHONPATH", "PYTHONHOME", "JAVA_HOME",
                     "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS", "PIP_TRUSTED_HOST", "PIP_NO_INDEX"):
            self.assertNotIn(name, env)
        self.assertEqual(env["PIP_CONFIG_FILE"], os.devnull)
        self.assertEqual(env["MAX_JOBS"], "4")
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")

    def test_clone_retry_does_not_replace_original_constraints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "core.txt"
            path.write_text("torch==2.6.0\n")
            with patch.object(checks, "installed_core", return_value={key: "new" for key in checks.CORE}), \
                 patch.object(sys, "version_info", (3, 10)), \
                 patch.object(checks, "verify_constraints", side_effect=RuntimeError("version mismatch")):
                with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                    checks.capture_core(path)
            self.assertEqual(path.read_text(), "torch==2.6.0\n")

    def test_dry_run_has_no_conda_or_filesystem_effects(self):
        for mode in ("fresh", "clone"):
            with self.subTest(mode=mode), patch.object(sys, "argv", [
                "setup_environment", "--benchmark", "webshop", "--mode", mode, "--dry-run"
            ]), patch.object(setup, "run", side_effect=AssertionError("Executed dry-run")), \
                 patch.object(setup, "environment_prefix", side_effect=AssertionError("Read Conda")), \
                 redirect_stdout(io.StringIO()) as output:
                setup.main()
            self.assertIn("/CONDA_ENVS/denoise-webshop", output.getvalue())
            self.assertNotIn("setup.sh", output.getvalue())
            self.assertIn("--override-channels", output.getvalue())
            self.assertIn("http://nexus.sii.shaipower.online/repository/anaconda/pkgs/main", output.getvalue())
            self.assertIn("http://nexus.sii.shaipower.online/repository/anaconda/cloud/conda-forge", output.getvalue())
            self.assertIn("--index-url http://nexus.sii.shaipower.online/repository/pypi_proxy/simple/", output.getvalue())
            self.assertIn("--trusted-host nexus.sii.shaipower.online", output.getvalue())

    def test_resume_can_select_other_internal_pip_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            records = target / setup.MARKER
            records.mkdir()
            marker = records / "setup.json"
            marker.write_text(json.dumps({"benchmark": "scienceworld", "mode": "fresh", "source_prefix": None,
                                          "mirror": "tuna", "pip_index_url": "https://pypi.tuna.tsinghua.edu.cn/simple",
                                          "status": "failed"}))
            endpoint = "http://nexus.sii.shaipower.online/repository/pypi/simple/"
            commands = []
            self.invoke(["--benchmark", "scienceworld", "--resume", "--pip-index-url", endpoint],
                        [target], commands.append)
            for command in commands:
                self.assertNotIn("tuna.tsinghua.edu.cn", " ".join(map(str, command)))
                if "pip" in command and "install" in command:
                    self.assertEqual(command[command.index("--index-url") + 1], endpoint)
                    self.assertEqual(command[command.index("--trusted-host") + 1], "nexus.sii.shaipower.online")
            saved = json.loads(marker.read_text())
            self.assertEqual(saved["mirror"], "internal")
            self.assertEqual(saved["pip_index_url"], endpoint)

    def test_legacy_setup_stops_before_running_any_installer(self):
        env = dict(os.environ)
        env.pop("WEBSHOP_ALLOW_LEGACY_INSTALL", None)
        # Only the guard's shell builtins can run; pip/conda are unavailable.
        env["PATH"] = "/nonexistent"
        result = subprocess.run(["/bin/bash", str(assets.WEBSHOP / "setup.sh"), "-d", "all"],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("disabled by default", result.stdout)
        self.assertEqual(result.stderr, "")


class AssetRecoveryTests(unittest.TestCase):
    def test_assets_require_completed_webshop_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            with self.assertRaises(RuntimeError):
                assets.require_webshop_environment(prefix)
            marker = prefix / ".denoise-task-suite/setup.json"
            marker.parent.mkdir()
            for benchmark, status in (("scienceworld", "dependencies_checked"), ("webshop", "failed")):
                marker.write_text(json.dumps({"benchmark": benchmark, "status": status}))
                with self.assertRaises(RuntimeError):
                    assets.require_webshop_environment(prefix)
            marker.write_text(json.dumps({"benchmark": "webshop", "status": "dependencies_checked"}))
            assets.require_webshop_environment(prefix)

    def test_index_build_failure_keeps_previous_index_and_success_backs_it_up(self):
        for fail in (True, False):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "data").mkdir()
                for filename in assets.ASSETS:
                    (root / "data" / filename).write_text("{}")
                search = root / "search_engine"
                (search / "indexes").mkdir(parents=True)
                (search / "indexes/old").write_text("old-index")
                commands = []
                def run(command, **kwargs):
                    commands.append(command)
                    if "pyserini.index.lucene" in command:
                        if fail:
                            raise subprocess.CalledProcessError(1, command)
                        output = Path(command[command.index("--index") + 1])
                        output.mkdir()
                        (output / "new").write_text("new-index")
                with patch.object(sys, "argv", ["prepare_assets", "--build-index"]), \
                     patch.object(assets, "WEBSHOP", root), \
                     patch.object(assets, "require_webshop_environment"), \
                     patch.object(assets.subprocess, "run", side_effect=run), redirect_stdout(io.StringIO()):
                    if fail:
                        with self.assertRaises(subprocess.CalledProcessError):
                            assets.main()
                    else:
                        assets.main()
                self.assertEqual(len(commands), 2)  # Existing JSONs did not trigger downloads.
                self.assertFalse(any("pip" in c or "conda" in c for c in commands))
                if fail:
                    self.assertEqual((search / "indexes/old").read_text(), "old-index")
                    self.assertEqual(list(search.glob("indexes.backup-*")), [])
                else:
                    self.assertEqual((search / "indexes/new").read_text(), "new-index")
                    backup, = search.glob("indexes.backup-*")
                    self.assertEqual((backup / "old").read_text(), "old-index")
                self.assertEqual(list(search.glob(".full-index-*")), [])

    def test_missing_new_index_restores_old_index_after_failed_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            for filename in assets.ASSETS:
                (root / "data" / filename).write_text("{}")
            (root / "search_engine/indexes").mkdir(parents=True)
            old = root / "search_engine/indexes/old"
            old.write_text("old-index")
            with patch.object(sys, "argv", ["prepare_assets", "--build-index"]), \
                 patch.object(assets, "WEBSHOP", root), patch.object(assets, "require_webshop_environment"), \
                 patch.object(assets.subprocess, "run"), redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):
                    assets.main()
            self.assertEqual(old.read_text(), "old-index")

    def test_webshop_text_registration_does_not_import_browser_dependencies(self):
        registration = types.ModuleType("gym.envs.registration")
        registration.register = lambda **kwargs: None
        text_env = types.ModuleType("web_agent_site.envs.web_agent_text_env")
        text_env.WebAgentTextEnv = object()
        with patch.dict(sys.modules, {"gym.envs.registration": registration,
                                     "web_agent_site.envs.web_agent_text_env": text_env,
                                     "web_agent_site.envs.web_agent_site_env": None}):
            namespace = runpy.run_path(str(assets.WEBSHOP / "web_agent_site/envs/__init__.py"))
        self.assertIs(namespace["WebAgentTextEnv"], text_env.WebAgentTextEnv)
        with self.assertRaises(AttributeError):
            namespace["__getattr__"]("unknown")


if __name__ == "__main__":
    unittest.main()
