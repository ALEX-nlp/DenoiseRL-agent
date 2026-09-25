"""Create a separate Conda training environment; never install into the source."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MARKER = ".denoise-task-suite"


def isolated_env():
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "JAVA_HOME", "PIP_TARGET", "PIP_PREFIX",
                "PIP_USER", "PIP_REQUIREMENT", "PIP_CONSTRAINT"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def run(command, capture=False):
    print("+ " + shlex.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), cwd=ROOT, env=isolated_env(), check=True,
                          text=True, stdout=subprocess.PIPE if capture else None).stdout


def validate_name(name, source):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in {"base", "molu", source}:
        raise ValueError("Choose a new environment name, different from base/molu/source.")


def environment_prefix(conda, name):
    info = json.loads(run([conda, "env", "list", "--json"], capture=True))
    matches = [Path(p).resolve() for p in info["envs"] if Path(p).name == name]
    if len(matches) > 1:
        raise ValueError(f"Multiple environments named {name!r}; choose a unique name")
    return matches[0] if matches else None


def installation_commands(conda, prefix, benchmark, mode):
    prefix = Path(prefix)
    target_python = [conda, "run", "--no-capture-output", "--prefix", str(prefix), "python"]
    records = prefix / MARKER
    constraints = HERE / "envs/train-constraints.txt" if mode == "fresh" else records / "core-constraints.txt"
    commands = []
    if mode == "clone":
        commands.append(target_python + [str(HERE / "environment_check.py"), "--capture-core", str(constraints)])
    commands.append([conda, "install", "--yes", "--prefix", str(prefix), "--freeze-installed",
                     "-c", "conda-forge", "openjdk=11", "python=3.10"])
    if mode == "fresh":
        commands.append(target_python + ["-m", "pip", "install", "-c", str(constraints),
                                        "-r", str(HERE / "envs/train-requirements.txt")])
        commands.append(target_python + ["-m", "pip", "install", "--no-build-isolation", "--no-deps",
                                        "flash-attn==2.7.4.post1"])
    requirements = (HERE / "envs/webshop-requirements.txt" if benchmark == "webshop"
                    else HERE / "requirements-scienceworld.txt")
    commands.append(target_python + ["-m", "pip", "install", "-c", str(constraints), "-r", str(requirements)])
    # Dependencies were installed above (fresh), or are inherited and checked
    # below (clone). Do not let the editable install re-resolve the GPU stack.
    commands.append(target_python + ["-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(ROOT)])
    commands.append(target_python + [str(HERE / "environment_check.py"), "--benchmark", benchmark,
                                    "--report-dir", str(records), "--constraints", str(constraints)])
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["webshop", "scienceworld"], required=True)
    parser.add_argument("--mode", choices=["fresh", "clone"], default="fresh")
    parser.add_argument("--source-env", default="molu")
    parser.add_argument("--name", help="Default: denoise-webshop or denoise-scienceworld")
    parser.add_argument("--resume", action="store_true", help="Resume only an environment created by this script")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without reading or changing Conda")
    args = parser.parse_args()
    name = args.name or f"denoise-{args.benchmark}"
    validate_name(name, args.source_env)
    if args.dry_run:
        create = ["conda", "create", "--yes", "--name", name]
        create += ["--clone", args.source_env, "--copy"] if args.mode == "clone" else ["python=3.10", "pip"]
        if not args.resume:
            print(shlex.join(create))
        for command in installation_commands("conda", f"/CONDA_ENVS/{name}", args.benchmark, args.mode):
            print(shlex.join(command))
        return
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This CUDA profile targets Linux x86_64. Use --dry-run to inspect it elsewhere.")
    conda = os.getenv("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise RuntimeError("Conda is not available")
    source = environment_prefix(conda, args.source_env) if args.mode == "clone" else None
    if args.mode == "clone" and source is None:
        raise ValueError(f"Source environment {args.source_env!r} not found")
    prefix = environment_prefix(conda, name)
    identity = {"benchmark": args.benchmark, "mode": args.mode, "source_prefix": str(source) if source else None}
    if prefix:
        marker = prefix / MARKER / "setup.json"
        if prefix == source or not args.resume or not marker.is_file():
            raise ValueError(f"Refusing to change existing environment {prefix}. Use a new name.")
        saved = json.loads(marker.read_text())
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("Resume parameters differ from the environment's original setup")
    else:
        if args.resume:
            raise ValueError("Cannot resume: environment does not exist")
        if source:
            # Read-only source check. A broken clone should not be mistaken for
            # a clean environment; fresh mode is the recovery path.
            run([conda, "run", "--no-capture-output", "--prefix", str(source), "python", "-m", "pip", "check"])
            run([conda, "create", "--yes", "--name", name, "--clone", str(source), "--copy"])
        else:
            run([conda, "create", "--yes", "--name", name, "python=3.10", "pip"])
        prefix = environment_prefix(conda, name)
        if prefix is None or prefix == source:
            raise RuntimeError("Could not verify the newly created destination")
    records = prefix / MARKER
    records.mkdir(exist_ok=True)
    identity["status"] = "installing"
    (records / "setup.json").write_text(json.dumps(identity, indent=2) + "\n")
    try:
        for command in installation_commands(conda, prefix, args.benchmark, args.mode):
            run(command)
    except Exception:
        identity["status"] = "failed"
        (records / "setup.json").write_text(json.dumps(identity, indent=2) + "\n")
        print(f"Setup failed in {prefix}. Source environment was not modified.\n"
              "After addressing the error, rerun the same command with --resume.", flush=True)
        raise
    identity["status"] = "dependencies_checked"
    (records / "setup.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(f"Environment prepared: {name}\nconda activate {name}\n"
          f"Version snapshot: {records / 'pip-freeze.txt'}\n"
          "Next: prepare task assets/manifest, run smoke_env, then run a short GPU training test.")


if __name__ == "__main__":
    main()
