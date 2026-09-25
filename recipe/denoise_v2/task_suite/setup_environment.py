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
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MARKER = ".denoise-task-suite"
DEFAULT_MIRROR = "internal"
INTERNAL_HOST = "nexus.sii.shaipower.online"
INTERNAL_REPOSITORY = f"http://{INTERNAL_HOST}/repository"
MIRRORS = {
    "internal": {
        "defaults": [f"{INTERNAL_REPOSITORY}/anaconda/pkgs/main",
                     f"{INTERNAL_REPOSITORY}/anaconda/pkgs/r"],
        "conda_forge": f"{INTERNAL_REPOSITORY}/anaconda/cloud/conda-forge",
        "pip": f"{INTERNAL_REPOSITORY}/pypi_proxy/simple/",
    },
    "tuna": {
        "defaults": ["https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main",
                     "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r"],
        "conda_forge": "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge",
        "pip": "https://pypi.tuna.tsinghua.edu.cn/simple",
    },
    "official": {
        "defaults": ["https://repo.anaconda.com/pkgs/main", "https://repo.anaconda.com/pkgs/r"],
        "conda_forge": "https://conda.anaconda.org/conda-forge",
        "pip": "https://pypi.org/simple",
    },
}


def conda_channels(mirror, include_forge=False):
    profile = MIRRORS[mirror]
    channels = ([profile["conda_forge"]] if include_forge else []) + profile["defaults"]
    # Explicit URLs also cover `-c conda-forge`, without modifying ~/.condarc
    # or inheriting slow channels from the user's existing configuration.
    return ["--override-channels"] + [arg for channel in channels for arg in ("--channel", channel)]


def creation_command(conda, name, mode, source, mirror=DEFAULT_MIRROR):
    command = [conda, "create", "--yes", "--name", name] + conda_channels(mirror)
    return command + (["--clone", str(source), "--copy"] if mode == "clone" else ["python=3.10", "pip"])


def pip_source_args(mirror, index_url=None):
    index_url = index_url or MIRRORS[mirror]["pip"]
    args = ["--index-url", index_url, "--timeout", "120"]
    if urlsplit(index_url).hostname == INTERNAL_HOST:
        args += ["--trusted-host", INTERNAL_HOST]
    return args


def isolated_env():
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "JAVA_HOME", "PIP_TARGET", "PIP_PREFIX",
                "PIP_USER", "PIP_REQUIREMENT", "PIP_CONSTRAINT", "PIP_INDEX_URL",
                "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS", "PIP_NO_INDEX", "PIP_TRUSTED_HOST"):
        env.pop(key, None)
    # Use the explicit per-command index/trusted host, including in pip build
    # subprocesses. Old pip.conf files must not add public fallback indexes.
    env["PIP_CONFIG_FILE"] = os.devnull
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


def installation_commands(conda, prefix, benchmark, mode, mirror=DEFAULT_MIRROR, pip_index_url=None):
    prefix = Path(prefix)
    target_python = [conda, "run", "--no-capture-output", "--prefix", str(prefix), "python"]
    pip_install = target_python + ["-m", "pip", "install"] + pip_source_args(mirror, pip_index_url)
    records = prefix / MARKER
    constraints = HERE / "envs/train-constraints.txt" if mode == "fresh" else records / "core-constraints.txt"
    commands = []
    if mode == "clone":
        commands.append(target_python + [str(HERE / "environment_check.py"), "--capture-core", str(constraints)])
    commands.append([conda, "install", "--yes", "--prefix", str(prefix), "--freeze-installed"]
                    + conda_channels(mirror, include_forge=True) + ["openjdk=11", "python=3.10"])
    if mode == "fresh":
        commands.append(pip_install + ["-c", str(constraints), "-r", str(HERE / "envs/train-requirements.txt")])
        commands.append(pip_install + ["--no-build-isolation", "--no-deps", "flash-attn==2.7.4.post1"])
    requirements = (HERE / "envs/webshop-requirements.txt" if benchmark == "webshop"
                    else HERE / "requirements-scienceworld.txt")
    commands.append(pip_install + ["-c", str(constraints), "-r", str(requirements)])
    # Dependencies were installed above (fresh), or are inherited and checked
    # below (clone). Do not let the editable install re-resolve the GPU stack.
    commands.append(pip_install + ["--no-deps", "--no-build-isolation", "-e", str(ROOT)])
    commands.append(target_python + [str(HERE / "environment_check.py"), "--benchmark", benchmark,
                                    "--report-dir", str(records), "--constraints", str(constraints)])
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["webshop", "scienceworld"], required=True)
    parser.add_argument("--mode", choices=["fresh", "clone"], default="fresh")
    parser.add_argument("--source-env", default="molu")
    parser.add_argument("--mirror", choices=MIRRORS, default=DEFAULT_MIRROR,
                        help="Conda and pip download sources (default: internal)")
    parser.add_argument("--pip-index-url", help="Override the profile's pip index, e.g. the internal /pypi/simple/ endpoint")
    parser.add_argument("--name", help="Default: denoise-webshop or denoise-scienceworld")
    parser.add_argument("--resume", action="store_true", help="Resume only an environment created by this script")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without reading or changing Conda")
    args = parser.parse_args()
    name = args.name or f"denoise-{args.benchmark}"
    validate_name(name, args.source_env)
    if args.dry_run:
        if not args.resume:
            print(shlex.join(creation_command("conda", name, args.mode, args.source_env, args.mirror)))
        for command in installation_commands("conda", f"/CONDA_ENVS/{name}", args.benchmark, args.mode,
                                             args.mirror, args.pip_index_url):
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
        if prefix == source:
            raise ValueError(f"Refusing to install into source environment {prefix}. Use a new name.")
        if not marker.is_file():
            raise ValueError(
                f"Refusing to change existing environment {prefix}: installer record is missing at {marker}. "
                "--resume requires this record. The environment may have been created manually or "
                "interrupted before setup was recorded; inspect it first, or choose a new --name."
            )
        if not args.resume:
            raise ValueError(
                f"Environment {prefix} already exists and has an installer record. "
                "To continue installation, rerun the original setup command with --resume "
                "and the same --benchmark, --mode and --source-env. "
                "The --mirror and --pip-index-url options may change."
            )
        saved = json.loads(marker.read_text())
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError(
                f"Resume parameters differ from the environment's original setup. "
                f"Check {marker}: benchmark={saved.get('benchmark')!r}, mode={saved.get('mode')!r}, "
                f"source_prefix={saved.get('source_prefix')!r}."
            )
    else:
        if args.resume:
            raise ValueError("Cannot resume: environment does not exist")
        if source:
            # Read-only source check. A broken clone should not be mistaken for
            # a clean environment; fresh mode is the recovery path.
            run([conda, "run", "--no-capture-output", "--prefix", str(source), "python", "-m", "pip", "check"])
        run(creation_command(conda, name, args.mode, source, args.mirror))
        prefix = environment_prefix(conda, name)
        if prefix is None or prefix == source:
            raise RuntimeError("Could not verify the newly created destination")
    records = prefix / MARKER
    records.mkdir(exist_ok=True)
    # Download sources may change on --resume without changing environment identity.
    identity["mirror"] = args.mirror
    identity["pip_index_url"] = args.pip_index_url or MIRRORS[args.mirror]["pip"]
    identity["status"] = "installing"
    (records / "setup.json").write_text(json.dumps(identity, indent=2) + "\n")
    try:
        for command in installation_commands(conda, prefix, args.benchmark, args.mode, args.mirror, args.pip_index_url):
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
