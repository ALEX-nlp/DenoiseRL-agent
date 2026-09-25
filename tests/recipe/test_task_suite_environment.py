"""Installer isolation and index recovery contracts; never runs an installer."""

from contextlib import ExitStack, redirect_stdout
import hashlib
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

    def test_pip_check_conflicts_do_not_block_cloning(self):
        for benchmark in ("webshop", "scienceworld"):
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "molu"
                target = Path(directory) / f"denoise-{benchmark}"
                source.mkdir()
                target.mkdir()  # Simulate Conda creating this path.
                commands = []
                def run(command):
                    commands.append(command)
                    if command[-3:] == ["-m", "pip", "check"]:
                        raise subprocess.CalledProcessError(1, command)
                self.invoke(["--benchmark", benchmark, "--mode", "clone"], [source, None, target], run)
                self.assertEqual(commands[0][-3:], ["-m", "pip", "check"])
                self.assertIn("--clone", commands[1])
                self.assertTrue(any("install" in c for c in commands[2:]))
                self.assertIn("--report-dir", commands[-1])
                self.assertEqual(json.loads((target / setup.MARKER / "setup.json").read_text())["status"],
                                 "dependencies_checked")
                self.assertEqual(list(source.iterdir()), [])

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


class EnvironmentCheckTests(unittest.TestCase):
    def test_pip_findings_are_recorded_but_import_errors_still_block(self):
        # Exercise the full post-install checker without a GPU or native JVM.
        # A clean check, advisory conflicts and a genuine import failure must
        # produce different reports/exit behavior.
        for pip_code, import_failure in ((0, False), (1, False), (1, True)):
            with self.subTest(pip_code=pip_code, import_failure=import_failure), \
                 tempfile.TemporaryDirectory() as directory:
                torch = types.ModuleType("torch")
                torch.cuda = types.SimpleNamespace(is_available=lambda: False)
                torch.version = types.SimpleNamespace(cuda="12.4")
                scienceworld = types.ModuleType("scienceworld")
                scienceworld.ScienceWorldEnv = lambda **kwargs: types.SimpleNamespace(
                    get_task_names=lambda: ["boil"], close=lambda: None)
                pip_stdout = "decord 0.6.0 is not supported on this platform\n" if pip_code else "No broken requirements found.\n"
                pip_stderr = "dependency diagnostic\n" if pip_code else ""
                def run(command, **kwargs):
                    if command[-3:] == ["-m", "pip", "check"]:
                        self.assertFalse(kwargs["check"])
                        return subprocess.CompletedProcess(command, pip_code, pip_stdout, pip_stderr)
                    if "freeze" in command:
                        return subprocess.CompletedProcess(command, 0, "torch==2.6.0\n", "")
                    if command == ["java", "-version"]:
                        return subprocess.CompletedProcess(command, 0, "", "openjdk version 11")
                    self.fail(f"Unexpected command: {command}")
                def import_module(name):
                    if import_failure and name == "vllm":
                        raise ImportError("simulated native import failure")
                    return types.SimpleNamespace()
                with patch.object(sys, "path", list(sys.path)), \
                     patch.dict(sys.modules, {"torch": torch, "scienceworld": scienceworld}), \
                     patch.object(checks.platform, "platform", return_value="Linux-test"), \
                     patch.object(checks, "installed_core", return_value={}), \
                     patch.object(checks.subprocess, "run", side_effect=run), \
                     patch.object(checks.importlib, "import_module", side_effect=import_module), \
                     redirect_stdout(io.StringIO()):
                    if import_failure:
                        with self.assertRaisesRegex(RuntimeError, "Environment checks failed"):
                            checks.check_environment("scienceworld", Path(directory))
                    else:
                        checks.check_environment("scienceworld", Path(directory))
                result = json.loads((Path(directory) / "check.json").read_text())
                self.assertEqual(result["pip_check"], pip_stdout + pip_stderr)
                self.assertEqual(result["pip_check_returncode"], pip_code)
                self.assertEqual(bool(result["warnings"]), bool(pip_code))
                self.assertEqual(bool(result["errors"]), import_failure)
                if import_failure:
                    self.assertEqual(result["errors"], ["vllm: simulated native import failure"])


class AssetRecoveryTests(unittest.TestCase):
    def setUp(self):
        # Small files exercise the same integrity checks without downloading GBs.
        checksum = hashlib.sha256(b"{}").hexdigest()
        patcher = patch.object(assets, "ASSET_CHECKSUMS", {name: (2, checksum) for name in assets.ASSETS})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_hf_download_pins_revision_and_reuses_completed_files_on_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            calls = []
            fail_once = [True]
            hub = types.ModuleType("huggingface_hub")
            def download(**kwargs):
                calls.append(kwargs)
                stage = Path(kwargs["local_dir"])
                stage.mkdir(exist_ok=True)
                if kwargs["filename"] == "items_ins_v2.json" and fail_once[0]:
                    fail_once[0] = False
                    (stage / "retained.incomplete").write_bytes(b"partial")
                    raise ConnectionError("interrupted")
                path = stage / kwargs["filename"]
                path.write_bytes(b"{}")
                return str(path)
            hub.hf_hub_download = download
            with patch.dict(sys.modules, {"huggingface_hub": hub}), \
                 patch.dict(os.environ, {}, clear=False), \
                 patch.object(assets.subprocess, "run", side_effect=AssertionError("Unexpected gdown call")), \
                 redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "Retry the same command"):
                    assets.prepare_assets(data, download=True, hf_endpoint="https://hf-mirror.com")
                self.assertEqual((data / "items_shuffle.json").read_bytes(), b"{}")
                self.assertFalse((data / "items_ins_v2.json").exists())
                self.assertTrue((data / ".hf-download/retained.incomplete").is_file())
                assets.prepare_assets(data, download=True, hf_endpoint="https://hf-mirror.com")
            self.assertEqual([c["filename"] for c in calls], ["items_shuffle.json", "items_ins_v2.json",
                                                             "items_ins_v2.json", "items_human_ins.json"])
            for call in calls:
                self.assertEqual(call["repo_id"], "HongbangYuan/webshop")
                self.assertEqual(call["revision"], "0129d4a81dbdb827e76afd20a1e2c38b61098613")
                self.assertEqual(call["repo_type"], "dataset")
                self.assertEqual(call["endpoint"], "https://hf-mirror.com")
                self.assertFalse(call["token"])
            self.assertTrue(all((data / name).is_file() for name in assets.ASSETS))

    def test_incomplete_or_corrupt_download_is_not_promoted(self):
        for content in (b"<html>error page</html>", b"[]"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                data = Path(directory)
                stage = data / "download.partial"
                stage.write_bytes(content)
                with patch.object(assets, "download_asset", return_value=stage), redirect_stdout(io.StringIO()):
                    with self.assertRaises(ValueError):
                        assets.prepare_assets(data, download=True)
                self.assertFalse((data / "items_shuffle.json").exists())
                self.assertEqual(stage.read_bytes(), content)

    def test_existing_invalid_data_is_preserved_and_does_not_trigger_download(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            path = data / "items_shuffle.json"
            path.write_bytes(b"[]")
            with patch.object(assets, "download_asset", side_effect=AssertionError("Overwrote existing data")), \
                 redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    assets.prepare_assets(data, download=True)
            self.assertEqual(path.read_bytes(), b"[]")

    def test_google_drive_failure_has_actionable_alternative(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(assets.subprocess, "run",
                side_effect=subprocess.CalledProcessError(1, ["gdown"])):
            with self.assertRaisesRegex(RuntimeError, "--source huggingface"):
                assets.download_asset("items_shuffle.json", Path(directory), "google-drive")

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
