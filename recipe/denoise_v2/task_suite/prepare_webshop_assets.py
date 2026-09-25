"""Download/reuse WebShop assets and build its full index without installing packages."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[3]
WEBSHOP = ROOT / "agent_system/environments/env_package/webshop/webshop"
ASSETS = {
    "items_shuffle.json": "1A2whVgOO0euk5O13n2iYDM0bQRkkRduB",
    "items_ins_v2.json": "1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi",
    "items_human_ins.json": "14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O",
}


def require_webshop_environment(prefix):
    marker = Path(prefix) / ".denoise-task-suite/setup.json"
    if not marker.is_file():
        raise RuntimeError("Activate the WebShop environment created by setup_environment first")
    state = json.loads(marker.read_text())
    if state.get("benchmark") != "webshop" or state.get("status") != "dependencies_checked":
        raise RuntimeError("WebShop environment setup has not passed its dependency checks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download missing files; reuse existing complete files")
    parser.add_argument("--build-index", action="store_true", help="Rebuild full index, keeping the previous index as a backup")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    require_webshop_environment(sys.prefix)
    data = WEBSHOP / "data"
    data.mkdir(exist_ok=True)
    for filename, file_id in ASSETS.items():
        path = data / filename
        if path.is_file() and path.stat().st_size:
            print(f"Reuse {path}", flush=True)
            continue
        if not args.download:
            raise FileNotFoundError(f"Missing {path}; rerun with --download")
        temporary = path.with_suffix(".json.partial")
        subprocess.run([sys.executable, "-m", "gdown", f"https://drive.google.com/uc?id={file_id}",
                        "--output", str(temporary)], check=True)
        if not temporary.is_file() or not temporary.stat().st_size:
            raise RuntimeError(f"Download did not produce {filename}")
        temporary.replace(path)
    if args.build_index:
        search = WEBSHOP / "search_engine"
        with tempfile.TemporaryDirectory(prefix=".full-index-", dir=search) as directory:
            work = Path(directory)
            subprocess.run([sys.executable, str(search / "convert_product_file_format.py"),
                            "--file-path", str(data / "items_shuffle.json"),
                            "--attr-path", str(data / "items_ins_v2.json"),
                            "--output-root", str(work)], cwd=search, check=True)
            subprocess.run([sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
                            "--input", str(work / "resources"), "--index", str(work / "indexes"),
                            "--generator", "DefaultLuceneDocumentGenerator", "--threads", str(args.threads),
                            "--storePositions", "--storeDocvectors", "--storeRaw"], cwd=search, check=True)
            index = search / "indexes"
            backup = search / f"indexes.backup-{uuid.uuid4().hex[:8]}"
            if index.exists():
                index.rename(backup)
            try:
                shutil.move(str(work / "indexes"), str(index))
            except Exception:
                if backup.exists() and not index.exists():
                    backup.rename(index)
                raise
            print(f"Full index ready: {index}")
            if backup.exists():
                print(f"Previous index retained: {backup}")
    elif not (WEBSHOP / "search_engine/indexes").is_dir():
        raise RuntimeError("Data ready but index missing; rerun with --build-index")


if __name__ == "__main__":
    main()
