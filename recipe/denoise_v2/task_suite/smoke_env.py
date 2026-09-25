"""Exercise a real simulator's reset/replay contract before allocating GPUs."""
import argparse
from agent_system.environments.env_package.task_suite.runtime import load_manifest, TaskWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["webshop", "scienceworld"], required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest, args.benchmark)
    worker = TaskWorker(args.benchmark, manifest["backend_options"], 100, "score", manifest["catalog_fingerprint"])
    try:
        for row in manifest["splits"]["train"][:2]:
            initial, info = worker.reset(row["task_id"])
            action = "search[shirt]" if args.benchmark == "webshop" else "look around"
            expected, _, done, expected_info = worker.step(action)
            if done:
                raise RuntimeError("Smoke-test action unexpectedly terminated the task")
            actual, actual_info = worker.reset(row["task_id"], [action])
            if actual != expected or actual_info["task_id"] != expected_info["task_id"]:
                raise RuntimeError("Replay does not reproduce the same task state")
            clean, _ = worker.reset(row["task_id"])
            if clean != initial:
                raise RuntimeError("Reset is not deterministic")
            print(f"PASS {row['task_id']}: reset, selected task, prefix replay")
    finally:
        worker.close()


if __name__ == "__main__":
    main()
