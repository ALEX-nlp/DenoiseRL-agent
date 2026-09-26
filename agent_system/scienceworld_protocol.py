"""Reproducible ScienceWorld evaluation and bounded, task-preserving prompts.

References: SwiftSage NeurIPS 2023 appendix A and the official science_world
branch (eval_utils.load_variation / eval_agent_fast_slow.parse_args).
"""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def select_tasks(rows, per_type_limit=None):
    """Select first numeric variation IDs per type, independent of input order."""
    if per_type_limit is not None and (isinstance(per_type_limit, bool) or int(per_type_limit) != per_type_limit or per_type_limit < 1):
        raise ValueError("Evaluation per_type_limit must be a positive integer or null")
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_type"]].append(row)
    selected = []
    for name in sorted(groups):
        ordered = sorted(groups[name], key=lambda row: int(row["task_id"].rsplit("::", 1)[1]))
        selected.extend(ordered if per_type_limit is None else ordered[:int(per_type_limit)])
    return selected


def task_digest(task_ids):
    return hashlib.sha256(json.dumps(list(task_ids), separators=(",", ":")).encode()).hexdigest()


def score_metrics(task_ids, scores, task_types):
    """Report both episode-weighted and task-type-weighted scores (0--100)."""
    if len(task_ids) != len(scores) or len(scores) == 0:
        raise ValueError("Expected one score per nonempty evaluation episode")
    grouped = defaultdict(list)
    for task_id, score in zip(task_ids, scores):
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid ScienceWorld normalized score: {score}")
        grouped[task_types[task_id]].append(score)
    means = {name: sum(values) / len(values) for name, values in grouped.items()}
    metrics = {"score": 100 * sum(scores) / len(scores),
               "score_macro": 100 * sum(means.values()) / len(means),
               "task_type_count": len(means)}
    metrics.update({f"task_type/{name}/score": 100 * value for name, value in means.items()})
    return metrics


def write_report(directory, step, report):
    """Persist partial progress atomically; final reports also retain the step."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (f"{step}.summary.json" if report["complete"] else "progress.json")
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    if report["complete"]:
        # The latest progress file should also clearly say that evaluation ended.
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(directory / "progress.json")


ACTION_INSTRUCTION = (
    "Choose one executable action using the templates and visible object names. "
    "You may briefly reason inside <think>...</think>. "
    "End with exactly one <action>command</action> and nothing after it. "
    "For example: <action>look around</action>. Do not use [action] or [action>."
)


def render_prompt(parts, history=None):
    history = parts["history"] if history is None else history
    return (
        f"You are an agent in scienceworld.\nTask: {parts['task']}\n"
        f"Actions already taken: {parts['steps']}\n"
        f"Action templates (replace placeholders with visible object names):\n{parts['templates']}\n"
        f"Recent history:\n{''.join(history)}"
        f"Current observation:\n{parts['state']}\n{ACTION_INSTRUCTION}"
    )


def bounded_chat(parts, tokenizer, max_length, chat_kwargs):
    """Drop oldest history before tokenization; never truncate task/chat framing."""
    history = list(parts["history"])
    while True:
        chat = [{"role": "user", "content": render_prompt(parts, history)}]
        rendered = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, **chat_kwargs)
        if len(tokenizer.encode(rendered, add_special_tokens=False)) <= max_length:
            return chat, rendered
        if not history:
            raise ValueError(
                "ScienceWorld task/current observation/action templates exceed max_prompt_length. "
                "Increase the prompt budget; refusing to silently truncate the task."
            )
        history.pop(0)
