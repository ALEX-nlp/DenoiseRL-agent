"""Discover native task splits and create task manifests and placeholder parquet."""

import argparse
from collections import Counter
import json
from pathlib import Path

from agent_system.environments.env_package.task_suite.backends import (
    WEBSHOP_RHO_GROUPINGS, WEBSHOP_STRUCTURE_GROUPS, make_backend, fingerprint,
)
from agent_system.environments.env_package.task_suite.runtime import load_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["webshop", "scienceworld"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--webshop-data-dir", type=Path, default=Path("agent_system/environments/env_package/webshop/webshop/data"))
    parser.add_argument("--catalog-seed", type=int, default=42)
    parser.add_argument("--webshop-rho-grouping", choices=WEBSHOP_RHO_GROUPINGS, default="structure",
                        help="WebShop rho groups: structure (default, six requirement groups) or category (legacy)")
    parser.add_argument("--jar-path")
    parser.add_argument("--simplifications", default="")
    parser.add_argument("--train-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.train_batch_size < 1:
        parser.error("--train-batch-size must be positive")
    if args.benchmark == "webshop":
        options = {"file_path": str((args.webshop_data_dir / "items_shuffle.json").resolve()),
                   "attr_path": str((args.webshop_data_dir / "items_ins_v2.json").resolve()),
                   "human_attr_path": str((args.webshop_data_dir / "items_human_ins.json").resolve()),
                   "catalog_seed": args.catalog_seed, "rho_grouping": args.webshop_rho_grouping}
        for key in ("file_path", "attr_path", "human_attr_path"):
            if not Path(options[key]).is_file():
                raise FileNotFoundError(options[key])
    else:
        options = {"jar_path": str(Path(args.jar_path).resolve()) if args.jar_path else None,
                   "simplifications": args.simplifications}
    backend = make_backend(args.benchmark, options)
    try:
        splits = backend.catalog()
        if hasattr(backend, "runtime_signature"):
            options["runtime_signature"] = backend.runtime_signature
        manifest = {"version": 1, "benchmark": args.benchmark, "backend_options": options,
                    "catalog_fingerprint": backend.catalog_fingerprint, "splits": splits}
    finally:
        backend.close()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "tasks.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    load_manifest(manifest_path, args.benchmark)
    if len(splits["train"]) < args.train_batch_size:
        raise ValueError("Training task pool is smaller than train batch size")
    import datasets
    def write_rows(split, size):
        dataset = datasets.Dataset.from_dict({
            "data_source": [args.benchmark] * size,
            "prompt": [[{"role": "user", "content": "Complete the environment task."}] for _ in range(size)],
            "ability": ["agent"] * size,
            "reward_model": [{"style": "rule", "ground_truth": ""} for _ in range(size)],
            "extra_info": [{"split": split, "index": i} for i in range(size)],
        })
        dataset.to_parquet(str(args.output / f"{split}.parquet"))
    # One placeholder batch per trainer epoch; task identity/order comes from
    # the checkpointed curriculum, never from placeholder row indices.
    write_rows("train", args.train_batch_size)
    for split in ("dev", "test"):
        write_rows(split, len(splits[split]))
    task_type_counts = {}
    for split, rows in splits.items():
        counts = Counter(row["task_type"] for row in rows)
        if args.benchmark == "webshop" and args.webshop_rho_grouping == "structure":
            # Show empty structural groups too; only observed training groups
            # get a live rho state in the curriculum.
            counts = {group: counts[group] for group in WEBSHOP_STRUCTURE_GROUPS}
        task_type_counts[split] = dict(sorted(counts.items()))
    print(json.dumps({"manifest": str(manifest_path), "tasks": {k: len(v) for k, v in splits.items()},
                      "task_type_counts": task_type_counts,
                      "manifest_fingerprint": fingerprint(manifest)}, indent=2))


if __name__ == "__main__":
    main()
