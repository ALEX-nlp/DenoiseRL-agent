"""Download/reuse WebShop assets and build a matching index without installing packages."""

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
from agent_system.webshop_protocol import WEBSHOP_DATA_PROFILES, webshop_data_profile

ROOT = Path(__file__).resolve().parents[3]
WEBSHOP = ROOT / "agent_system/environments/env_package/webshop/webshop"
ASSETS = {
    "items_shuffle.json": "1A2whVgOO0euk5O13n2iYDM0bQRkkRduB",
    "items_ins_v2.json": "1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi",
    "items_human_ins.json": "14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O",
}
SMALL_ASSETS = {
    "items_shuffle_1000.json": "1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib",
    "items_ins_v2_1000.json": "1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu",
}
HF_DATASET = "HongbangYuan/webshop"
HF_REVISION = "0129d4a81dbdb827e76afd20a1e2c38b61098613"
# Full-catalog SHA256 values come from HF LFS metadata. Small-catalog and
# human-instruction hashes were computed directly from the pinned mirror.
# The three legacy full-profile assets also match YWZBrandon/webshop-data
# at ce990fff5aee388db2706f07820c578ab68e0453.
ASSET_CHECKSUMS = {
    "items_shuffle_1000.json": (4467013, "30a4765c3a327af72d9a9a95a6b2486d516f0fa1d3ecd83681901ce82a21b269"),
    "items_ins_v2_1000.json": (147099, "f88a36314a397b53b3d9c3fa5878e5f7b26d35019a51ec83fbedeca61a948f6f"),
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
        asset_id = {**ASSETS, **SMALL_ASSETS}[filename]
        subprocess.run([sys.executable, "-m", "gdown", f"https://drive.google.com/uc?id={asset_id}",
                        "--continue", "--output", str(temporary)], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Google Drive could not serve the file; retry with --source huggingface.") from exc
    return temporary


def prepare_assets(data, download=False, source="huggingface", hf_endpoint=None, data_profile="full_human"):
    profile = webshop_data_profile(data_profile)
    data.mkdir(exist_ok=True)
    # Native load_products opens human annotations even for synthetic goals.
    for filename in (profile["products"], profile["attributes"], "items_human_ins.json"):
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
    parser.add_argument("--webshop-data-profile", choices=WEBSHOP_DATA_PROFILES, default="gigpo_small")
    parser.add_argument("--download", action="store_true", help="Download missing files; reuse existing complete files")
    parser.add_argument("--source", choices=["huggingface", "google-drive"], default="huggingface",
                        help="Download source (default: a pinned Hugging Face mirror)")
    parser.add_argument("--hf-endpoint", help="Optional HF mirror endpoint; otherwise honors HF_ENDPOINT / the Hub default")
    parser.add_argument("--build-index", action="store_true", help="Rebuild this profile's index, retaining a backup")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    require_webshop_environment(sys.prefix)
    profile = webshop_data_profile(args.webshop_data_profile)
    data = WEBSHOP / "data"
    prepare_assets(data, args.download, args.source, args.hf_endpoint, args.webshop_data_profile)
    index = WEBSHOP / "search_engine" / profile["index"]
    if args.build_index:
        search = WEBSHOP / "search_engine"
        with tempfile.TemporaryDirectory(prefix=".profile-index-", dir=search) as directory:
            work = Path(directory)
            subprocess.run([sys.executable, str(search / "convert_product_file_format.py"),
                            "--file-path", str(data / profile["products"]),
                            "--attr-path", str(data / profile["attributes"]),
                            "--output-root", str(work)], cwd=search, check=True)
            subprocess.run([sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
                            "--input", str(work / "resources"), "--index", str(work / "indexes"),
                            "--generator", "DefaultLuceneDocumentGenerator", "--threads", str(args.threads),
                            "--storePositions", "--storeDocvectors", "--storeRaw"], cwd=search, check=True)
            backup = search / f"{profile['index']}.backup-{uuid.uuid4().hex[:8]}"
            if index.exists():
                index.rename(backup)
            try:
                shutil.move(str(work / "indexes"), str(index))
            except Exception:
                if backup.exists() and not index.exists():
                    backup.rename(index)
                raise
            print(f"{args.webshop_data_profile} index ready: {index}")
            if backup.exists():
                print(f"Previous index retained: {backup}")
    elif not index.is_dir():
        raise RuntimeError("Data ready but index missing; rerun with --build-index")


if __name__ == "__main__":
    main()
