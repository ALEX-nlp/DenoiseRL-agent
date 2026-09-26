"""Native simulator adapters. A task ID always identifies an exact initial task."""

import hashlib
import json
from pathlib import Path


WEBSHOP_RHO_GROUPINGS = ("structure", "category")
WEBSHOP_STRUCTURE_GROUPS = tuple(
    f"options_{options}__attrs_{attributes}"
    for options in ("0", "1", "2_plus")
    for attributes in ("1_2", "3_plus")
)


def webshop_task_type(goal, grouping):
    """Group training goals by annotated requirements, never solver outcomes."""
    if grouping == "category":
        return str(goal.get("category") or "shopping")
    if grouping != "structure":
        raise ValueError(f"Unknown WebShop rho grouping: {grouping!r}")
    attributes, options = goal.get("attributes"), goal.get("goal_options")
    # Human goals have at least one attribute; do not silently put malformed
    # annotations into a valid structural group. Upstream accepts list/dict options.
    if not isinstance(attributes, (list, tuple)) or not attributes:
        raise ValueError("WebShop structure grouping requires non-empty attributes")
    if not isinstance(options, (list, tuple, dict)):
        raise ValueError("WebShop structure grouping requires goal_options as a list or dict")
    option_bucket = str(len(options)) if len(options) < 2 else "2_plus"
    attribute_bucket = "1_2" if len(attributes) <= 2 else "3_plus"
    return f"options_{option_bucket}__attrs_{attribute_bucket}"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class WebShopBackend:
    def __init__(self, options, server=None, session_prefix=None):
        # Manifests prepared before structural grouping used category grouping.
        self.rho_grouping = options.get("rho_grouping", "category")
        if self.rho_grouping not in WEBSHOP_RHO_GROUPINGS:
            raise ValueError(f"Unknown WebShop rho grouping: {self.rho_grouping!r}")
        import sys
        root = Path(__file__).resolve().parents[1] / "webshop" / "webshop"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from web_agent_site.envs import WebAgentTextEnv
        if options.get("human_attr_path"):
            from web_agent_site.engine import engine
            engine.HUMAN_ATTR_PATH = options["human_attr_path"]
        # The simulator shuffles AND constructs goals using this seed. It must
        # be identical for every worker and split, independent of training seed.
        self.env = WebAgentTextEnv(
            observation_mode="text", human_goals=True,
            file_path=options["file_path"], attr_path=options["attr_path"],
            seed=int(options.get("catalog_seed", 42)), num_products=None,
            server=server, session_prefix=session_prefix,
        )
        self.goals = self.env.server.goals
        self.catalog_fingerprint = fingerprint(self.goals) if server is None else None

    def catalog(self):
        if len(self.goals) <= 1500:
            raise ValueError("WebShop requires the full goal pool (>1500 goals) for train/dev/test splits.")
        return {
            split: [
                {"task_id": str(i), "task_type": webshop_task_type(self.goals[i], self.rho_grouping)}
                for i in indices
            ]
            for split, indices in {
                "test": range(500), "dev": range(500, 1500), "train": range(1500, len(self.goals))
            }.items()
        }

    def _info(self, score=0.0, valid=True):
        available = self.env.get_available_actions()
        actions = (["search[<query>]"] if available["has_search_bar"] else [])
        actions += [f"click[{label}]" for label in available["clickables"]]
        return {"task_id": self.task_id, "task_description": self.env.instruction_text,
                "admissible_actions": actions, "task_score": float(score),
                "won": score >= 1.0, "is_action_valid": valid}

    def reset(self, task_id):
        self.task_id = str(task_id)
        index = int(task_id)
        if not 0 <= index < len(self.goals):
            raise ValueError(f"Unknown WebShop goal: {task_id}")
        obs, _ = self.env.reset(session=index)
        return obs, self._info()

    def step(self, action):
        available = self.env.get_available_actions()
        valid = ((action.startswith("search[") and action.endswith("]") and bool(action[7:-1].strip())
                  and available["has_search_bar"])
                 or action.lower() in {f"click[{s}]".lower() for s in available["clickables"]})
        obs, score, done, _ = self.env.step(action)
        return obs, bool(done), self._info(score, valid)

    def close(self):
        self.env.close()


class ScienceWorldBackend:
    def __init__(self, options):
        from importlib.metadata import version
        from scienceworld import ScienceWorldEnv
        from scienceworld.constants import JAR_PATH
        jar = Path(options.get("jar_path") or JAR_PATH)
        digest = hashlib.sha256()
        with jar.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        self.runtime_signature = {"scienceworld_version": version("scienceworld"), "jar_sha256": digest.hexdigest()}
        if options.get("runtime_signature", self.runtime_signature) != self.runtime_signature:
            raise ValueError("ScienceWorld version/JAR differs from the prepared task manifest")
        self.options = options
        self.env = ScienceWorldEnv(serverPath=options.get("jar_path"),
                                   envStepLimit=int(options.get("env_step_limit", options.get("max_steps", 100))))
        self.simplifications = options.get("simplifications", "")
        self.score_mode = options.get("score_mode", "terminal")
        if self.score_mode not in {"terminal", "last_nonnegative"}:
            raise ValueError(f"Unknown ScienceWorld score mode: {self.score_mode}")
        self.catalog_fingerprint = None

    def catalog(self):
        result = {split: [] for split in ("train", "dev", "test")}
        for name in sorted(self.env.get_task_names()):
            self.env.load(name, 0, self.simplifications, generateGoldPath=False)
            for split in result:
                for variation in sorted(getattr(self.env, f"get_variations_{split}")()):
                    result[split].append({"task_id": f"{name}::{variation}", "task_type": name})
        self.catalog_fingerprint = fingerprint(result)
        return result

    def _info(self, info):
        raw_score = float(info["score"])
        score = raw_score
        if getattr(self, "score_mode", "terminal") == "last_nonnegative":
            score = self.last_nonnegative_score
        return {"task_id": self.task_id,
                "task_description": info.get("taskDesc", self.env.get_task_description()),
                "admissible_actions": list(info.get("valid", [])),
                "action_templates": getattr(self, "action_templates", []),
                "task_score": max(0.0, min(1.0, score / 100.0)),
                "raw_score": raw_score, "won": raw_score >= 100,
                "look": info.get("look", ""), "inventory": info.get("inv", "")}

    def reset(self, task_id):
        self.task_id = str(task_id)
        name, variation = self.task_id.rsplit("::", 1)
        self.env.load(name, int(variation), self.simplifications, generateGoldPath=False)
        obs, info = self.env.reset()
        self.action_templates = list(self.env.get_possible_actions())
        self.last_nonnegative_score = max(0.0, float(info["score"]))
        self.recent_deltas = [0.0]
        if info["taskName"] != name or int(info["variationIdx"]) != int(variation):
            raise RuntimeError("ScienceWorld loaded a different task or variation")
        self.previous_actions = set(info.get("valid", []))
        return obs, self._info(info)

    def step(self, action):
        valid = action in self.previous_actions
        obs, _delta_reward, done, info = self.env.step(action)
        raw_score = float(info["score"])
        self.recent_deltas.append(raw_score - self.last_nonnegative_score)
        done = bool(done) or raw_score < 0
        stagnant = (self.options.get("stop_on_stagnation", False)
                    and len(self.recent_deltas) >= 100 and sum(self.recent_deltas[-30:]) == 0)
        if raw_score >= 0 and not stagnant:
            self.last_nonnegative_score = raw_score
        done = done or stagnant
        self.previous_actions = set(info.get("valid", []))
        normalized = self._info(info)
        normalized["is_action_valid"] = valid
        # The native wrapper also terminates on its moves budget, before the
        # outer action budget in long evaluations. Success/failure is terminal,
        # not a cutoff; stagnation is a separate policy rule.
        move_limit = getattr(self.env, "envStepLimit", self.options.get("env_step_limit", self.options.get("max_steps", 100)))
        normalized["truncated"] = bool(done and not stagnant and 0 <= raw_score < 100
                                       and info.get("moves", 0) > move_limit)
        return obs, bool(done), normalized

    def close(self):
        self.env.close()


def make_backend(benchmark, options):
    if benchmark == "webshop":
        return WebShopBackend(options)
    if benchmark == "scienceworld":
        return ScienceWorldBackend(options)
    raise ValueError(f"Unknown benchmark: {benchmark}")
