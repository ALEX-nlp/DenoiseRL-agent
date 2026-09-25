"""Helpers for deterministic, exhaustive ALFWorld validation."""

from collections import Counter
from os import fspath


def collapse_trajectory_rows(
    trajectory_ids,
    outputs: list[str],
    scores: list[float],
) -> tuple[list[str], list[str], list[float]]:
    """Collapse step-expanded rollout rows into first-seen trajectory order."""
    if not (len(trajectory_ids) == len(outputs) == len(scores)):
        raise ValueError(
            "Validation trajectory row lengths differ: "
            f"ids={len(trajectory_ids)}, outputs={len(outputs)}, scores={len(scores)}."
        )

    ordered_ids = []
    output_by_id = {}
    score_by_id = {}
    for trajectory_id, output, score in zip(trajectory_ids, outputs, scores):
        if trajectory_id not in output_by_id:
            ordered_ids.append(trajectory_id)
            output_by_id[trajectory_id] = []
            score_by_id[trajectory_id] = float(score)
        elif score_by_id[trajectory_id] != float(score):
            raise RuntimeError(
                "Validation rows for one trajectory have inconsistent episode scores: "
                f"trajectory_id={trajectory_id!r}."
            )
        output_by_id[trajectory_id].append(output)

    collapsed_outputs = ["\n".join(output_by_id[trajectory_id]) for trajectory_id in ordered_ids]
    collapsed_scores = [score_by_id[trajectory_id] for trajectory_id in ordered_ids]
    return ordered_ids, collapsed_outputs, collapsed_scores


def ordered_validation_gamefiles(val_envs) -> tuple[str, ...]:
    """Return the complete validation pool in a deterministic order."""
    envs = getattr(val_envs, "envs", None)
    raw_gamefiles = getattr(envs, "task_ids", getattr(envs, "game_files", ()))
    if not raw_gamefiles:
        return ()

    gamefiles = tuple(sorted(fspath(gamefile) for gamefile in raw_gamefiles))
    if len(set(gamefiles)) != len(gamefiles):
        raise ValueError("ALFWorld validation pool contains duplicate gamefiles.")

    declared_num_games = getattr(envs, "num_games", None)
    if declared_num_games is not None and int(declared_num_games) != len(gamefiles):
        raise ValueError(
            "ALFWorld validation pool size does not match env.num_games: "
            f"pool={len(gamefiles)}, num_games={declared_num_games}."
        )
    return gamefiles


def build_gamefile_reset_kwargs(
    gamefiles: tuple[str, ...],
    *,
    start: int,
    count: int,
    repeats: int,
    reset_key: str = "gamefile",
) -> tuple[dict[str, str], ...]:
    """Build reset kwargs matching DataProto.repeat(..., interleave=True)."""
    if start < 0 or count < 0:
        raise ValueError(f"start and count must be non-negative, got {start=} and {count=}.")
    if repeats < 1:
        raise ValueError(f"repeats must be at least one, got {repeats}.")

    selected = gamefiles[start : start + count]
    if len(selected) != count:
        raise ValueError(
            "Requested validation slice exceeds the ALFWorld gamefile pool: "
            f"start={start}, count={count}, pool={len(gamefiles)}."
        )

    return tuple(
        {
            reset_key: gamefile,
            "validation_gamefile": gamefile,
        }
        for gamefile in selected
        for _ in range(repeats)
    )


def validate_gamefile_coverage(
    expected_gamefiles: tuple[str, ...],
    observed_gamefiles: list[str],
    *,
    repeats: int,
) -> dict[str, float]:
    """Fail unless every expected gamefile was evaluated exactly ``repeats`` times."""
    if repeats < 1:
        raise ValueError(f"repeats must be at least one, got {repeats}.")

    expected = Counter({gamefile: repeats for gamefile in expected_gamefiles})
    observed = Counter(observed_gamefiles)
    missing = sorted((expected - observed).elements())
    extra = sorted((observed - expected).elements())
    wrong_counts = {
        gamefile: observed[gamefile]
        for gamefile in expected_gamefiles
        if observed[gamefile] != repeats
    }
    if missing or extra or wrong_counts:
        raise RuntimeError(
            "Incomplete ALFWorld validation coverage: "
            f"expected_unique={len(expected_gamefiles)}, "
            f"observed_unique={len(observed)}, "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"wrong_counts={dict(list(wrong_counts.items())[:5])}."
        )

    return {
        "gamefile_count": float(len(expected_gamefiles)),
        "gamefile_episode_count": float(len(observed_gamefiles)),
        "gamefile_unique_count": float(len(observed)),
        "gamefile_coverage": 1.0,
    }
