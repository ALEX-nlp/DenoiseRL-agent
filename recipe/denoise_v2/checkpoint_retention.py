"""Keep complete, resumable latest/best checkpoints on a shared local filesystem."""

import json
import math
from pathlib import Path
import re
import shutil


class LatestBestCheckpoints:
    def __init__(self, root, metric):
        self.root = Path(root)
        self.metric = metric
        self.state_path = self.root / "checkpoint_retention.json"
        self.best = None
        self.latest_step = None
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if state["metric"] != metric:
                raise ValueError("Checkpoint selection metric changed; use a new experiment directory")
            self.best = state["best"]
            self.latest_step = state["latest_step"]
            if not isinstance(self.latest_step, int) or self.latest_step < 0:
                raise ValueError("Invalid latest-checkpoint metadata")
            if self.best is not None:
                if not math.isfinite(self.best["value"]) or not isinstance(self.best["step"], int) or self.best["step"] < 0:
                    raise ValueError("Invalid best-checkpoint metadata")
                self._require_complete(self.best["step"])

    def _path(self, step):
        return self.root / f"global_step_{step}"

    def _require_complete(self, step):
        path = self._path(step)
        if path.is_symlink() or not (path / "actor").is_dir() or not (path / "data.pt").is_file():
            raise ValueError(f"Incomplete checkpoint: {path}")

    @staticmethod
    def _write_atomic(path, text):
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    def restore_trackers(self):
        """Recover an interrupted pointer update from the committed state file."""
        if self.latest_step is not None:
            self._require_complete(self.latest_step)
            self._write_atomic(self.root / "latest_checkpointed_iteration.txt", str(self.latest_step))
            if self.best is not None:
                self._write_atomic(self.root / "best_checkpointed_iteration.txt", str(self.best["step"]))

    def maybe_save(self, step, metrics, scheduled, final, save, initial=False):
        value = None
        if metrics is not None:
            if self.metric not in metrics:
                raise ValueError(f"Best-checkpoint metric {self.metric!r} is missing from validation")
            value = float(metrics[self.metric])
            if not math.isfinite(value):
                print(f"[checkpoint] Ignoring non-finite {self.metric}={value} for best selection")
                value = None
        improved = value is not None and (self.best is None or value > self.best["value"])
        if not (scheduled or final or improved):
            return False

        # The callback must finish actor/optimizer, dataloader AND curriculum state
        # before pointers move or any old checkpoint is removed.
        saved_steps = {self.latest_step, self.best["step"] if self.best is not None else None}
        # On resume, initial evaluation does not change weights or training state.
        # Do not rewrite the only good copy of an already complete checkpoint.
        if not (initial and step in saved_steps):
            save()
        self._require_complete(step)
        best = {"step": step, "value": value} if improved else self.best
        state = {"metric": self.metric, "latest_step": step, "best": best}
        self._write_atomic(self.state_path, json.dumps(state, indent=2) + "\n")
        self.best = best
        self.latest_step = step
        self.restore_trackers()

        keep = {step}
        if best is not None:
            keep.add(best["step"])
        for path in self.root.iterdir():
            match = re.fullmatch(r"global_step_(\d+)", path.name)
            # Never traverse a symlink or touch another experiment's directory.
            if match and int(match[1]) not in keep and path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
                print(f"[checkpoint] Removed {path}")
        print(f"[checkpoint] latest={step}, best={best}, metric={self.metric}")
        return True
