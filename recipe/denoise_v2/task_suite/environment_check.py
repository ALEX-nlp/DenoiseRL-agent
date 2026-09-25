"""Check an isolated task-suite environment and record its exact installed state."""

import argparse
import importlib
from importlib.metadata import distributions, version, PackageNotFoundError
import json
from pathlib import Path
import platform
import subprocess
import sys

CORE = {"torch", "torchvision", "torchaudio", "vllm", "flash-attn", "triton", "xformers",
        "transformers", "tokenizers", "peft", "ray", "tensordict", "torchdata"}


def installed_core():
    return {d.metadata["Name"].lower().replace("_", "-"): d.version for d in distributions()
            if d.metadata["Name"].lower().replace("_", "-") in CORE
            or d.metadata["Name"].lower().startswith("nvidia-")}


def capture_core(path):
    core = installed_core()
    missing = CORE - set(core)
    if missing:
        raise RuntimeError(f"Clone lacks training packages {sorted(missing)}; use --mode fresh")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Clone mode requires a Python 3.10 source; use --mode fresh for another Python version")
    # Preserve the original lock on retry instead of blessing a partial install.
    if path.exists():
        verify_constraints(path)
    else:
        path.write_text("".join(f"{key}=={value}\n" for key, value in sorted(core.items())))


def verify_constraints(path):
    from packaging.requirements import Requirement
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            # A constraint does not require optional packages to be installed.
            continue
        if installed not in requirement.specifier:
            raise RuntimeError(f"Protected constraint changed: {requirement}, installed {installed}")


def check_environment(benchmark, report_dir, constraints=None):
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    report_dir.mkdir(parents=True, exist_ok=True)
    result = {"python": sys.version, "prefix": sys.prefix, "platform": platform.platform(),
              "benchmark": benchmark, "core": installed_core(), "warnings": [], "errors": []}
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True)
    (report_dir / "pip-freeze.txt").write_text(freeze.stdout)
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False)
    result["pip_check"] = check.stdout + check.stderr
    result["pip_check_returncode"] = check.returncode
    if check.returncode:
        result["warnings"].append(
            f"pip check exited with status {check.returncode}; dependency/platform findings are advisory "
            "and have not been fixed."
        )
    modules = ["torch", "vllm", "flash_attn", "transformers", "peft", "ray", "tensordict",
               "torchdata.stateful_dataloader", "recipe.denoise_v2.main_online_denoise"]
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:
            result["errors"].append(f"{module}: {exc}")
    try:
        if constraints:
            verify_constraints(constraints)
        java = subprocess.run(["java", "-version"], capture_output=True, text=True, check=True)
        result["java"] = java.stdout + java.stderr
        if benchmark == "webshop":
            sys.path.insert(0, str(root / "agent_system/environments/env_package/webshop/webshop"))
            importlib.import_module("web_agent_site.envs.web_agent_text_env")
            import spacy
            result["spacy_model"] = spacy.load("en_core_web_sm").meta["version"]
        else:
            from scienceworld import ScienceWorldEnv
            env = ScienceWorldEnv(envStepLimit=10)
            try:
                result["task_types"] = len(env.get_task_names())
            finally:
                env.close()
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda"] = torch.version.cuda
    except Exception as exc:
        result["errors"].append(str(exc))
    (report_dir / "check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise RuntimeError("Environment checks failed; see check.json. No training has started.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-core", type=Path)
    parser.add_argument("--benchmark", choices=["webshop", "scienceworld"])
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--constraints", type=Path)
    args = parser.parse_args()
    if args.capture_core:
        capture_core(args.capture_core)
    elif args.benchmark and args.report_dir:
        check_environment(args.benchmark, args.report_dir, args.constraints)
    else:
        parser.error("Specify --capture-core, or --benchmark and --report-dir")


if __name__ == "__main__":
    main()
