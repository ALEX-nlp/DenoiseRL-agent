"""Download/reuse WebShop assets and build its full index without installing packages."""

import argparse
import hashlib
import json
import os
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
HF_DATASET = "HongbangYuan/webshop"
HF_REVISION = "0129d4a81dbdb827e76afd20a1e2c38b61098613"
# Full-catalog mirror, not the 1000-product subset. Large-file SHA256 values
# come from HF LFS metadata; the human-instruction hash was computed directly.
# All three sizes/hashes match YWZBrandon/webshop-data at ce990fff5aee388db2706f07820c578ab68e0453.
ASSET_CHECKSUMS = {
    "items_shuffle.json": (5479720229, "2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8"),
    "items_ins_v2.json": (186295270, "1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e"),
    "items_human_ins.json": (5137548, "cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3"),
}


def verify_asset(path, filename):
    size, expected = ASSET_CHECKSUMS[filename]
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"Incomplete or different WebShop data: {path}; expected {size} bytes. "
                         "Move the invalid file aside before retrying; existing files are not overwritten.")
    print(f"Verify SHA256: {path}", flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"WebShop SHA256 mismatch: {path}; expected {expected}. "
                         "Move the invalid file aside before retrying.")


def download_asset(filename, data, source, hf_endpoint=None):
    if source == "huggingface":
        # Read these before importing the Hub library; keep interrupted
        # downloads in a stable local directory for automatic resumption.
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        from huggingface_hub import hf_hub_download
        print(f"Download {filename} from {HF_DATASET}@{HF_REVISION}", flush=True)
        try:
            return Path(hf_hub_download(
                repo_id=HF_DATASET, repo_type="dataset", filename=filename,
                revision=HF_REVISION, local_dir=str(data / ".hf-download"),
                endpoint=hf_endpoint, token=False,
            ))
        except Exception as exc:
            raise RuntimeError(
                f"Hugging Face download failed for {filename}. Retry the same command to resume, "
                "use --hf-endpoint for an accessible HF mirror, or copy the three JSON files "
                f"from a connected machine into {data} and run --build-index."
            ) from exc
    temporary = data / f"{filename}.partial"
    try:
        subprocess.run([sys.executable, "-m", "gdown", f"https://drive.google.com/uc?id={ASSETS[filename]}",
                        "--continue", "--output", str(temporary)], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Google Drive could not serve the file; retry with --source huggingface.") from exc
    return temporary


def prepare_assets(data, download=False, source="huggingface", hf_endpoint=None):
    data.mkdir(exist_ok=True)
    for filename in ASSETS:
        path = data / filename
        if path.is_file() and path.stat().st_size:
            verify_asset(path, filename)
            print(f"Reuse {path}", flush=True)
            continue
        if not download:
            raise FileNotFoundError(f"Missing {path}; rerun with --download")
        temporary = download_asset(filename, data, source, hf_endpoint)
        # Do not let an HTML error page or partial download become a dataset.
        verify_asset(temporary, filename)
        temporary.replace(path)


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
    parser.add_argument("--source", choices=["huggingface", "google-drive"], default="huggingface",
                        help="Download source (default: a pinned full-catalog Hugging Face mirror)")
    parser.add_argument("--hf-endpoint", help="Optional HF mirror endpoint; otherwise honors HF_ENDPOINT / the Hub default")
    parser.add_argument("--build-index", action="store_true", help="Rebuild full index, keeping the previous index as a backup")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    require_webshop_environment(sys.prefix)
    data = WEBSHOP / "data"
    prepare_assets(data, args.download, args.source, args.hf_endpoint)
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
