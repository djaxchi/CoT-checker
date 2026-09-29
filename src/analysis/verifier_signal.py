"""Controlled input interventions for frozen step verifiers.

Labels describe explicit mathematical relations, not PRM800K annotations.
No probe fitting or threshold selection occurs in this experiment.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DOMAINS = ("affine", "multiplication", "inequality", "substitution")


def sha256(path: Path) -> str:
    """Hash a file without loading model weights into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    """Read a JSONL file in its frozen order."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def styled(statement: str, style: str) -> str:
    """Change only the discourse wrapper around a mathematical assertion."""
    return {"plain": "", "therefore": "Therefore, ",
            "confident": "I am certain that "}[style] + statement + "."


def expected_labels(row: dict) -> dict:
    """Recompute labels from the arithmetic witness, independently of stored labels."""
    w = row["witness"]
    a, b, x = w["a"], w["b"], w["x"]
    if row["arm"] == "reversal":
        p, c = row["prefix_variant"], row["candidate_variant"]
        if row["domain"] == "affine":
            valid = a * (x + c) + b == a * (x + p) + b
        elif row["domain"] == "multiplication":
            valid = a * (x + p) == a * (x + c)
        elif row["domain"] == "inequality":
            # Prefix: sign*a*x < sign*a*b. Dividing by a negative reverses <.
            valid = (p == 0 and c == 0) or (p == 1 and c == 1)
        elif row["domain"] == "substitution":
            valid = (x + p) ** 2 + b == (x + c) ** 2 + b
        else:
            raise ValueError("unknown domain")
        local, inherited, conclusion = int(not valid), 0, int(not valid)
    elif row["arm"] == "inheritance":
        p = row["prefix_variant"]
        inherited = int(a * (x + p) != (a * x + b) - b)
        if row["role"] == "before":
            local, conclusion = inherited, inherited
            inherited = 0
        else:
            c = row["candidate_variant"]
            local = int(a * (x + c) != a * (x + p))
            conclusion = int(a * (x + c) + b != a * x + b)
    else:
        raise ValueError("unknown arm")
    return {"local_invalid": local, "prefix_invalid": inherited,
            "conclusion_invalid": conclusion,
            "trace_invalid": int(bool(local or inherited)), "label": local}


def render(row: dict) -> tuple[str, str, str]:
    """Render only problem, prefix, and candidate text; never include labels."""
    w = row["witness"]
    a, b, x = w["a"], w["b"], w["x"]
    p, c = row["prefix_variant"], row["candidate_variant"]
    if row["arm"] == "inheritance":
        problem = f"Solve {a}*x + {b} = {a*x+b} for real x."
        intermediate = f"Subtracting {b} from both sides gives {a}*x = {a*(x+p)}."
        if row["role"] == "before":
            return problem, "", intermediate
        return problem, intermediate, styled(f"x = {x+c}", row["style"])
    domain = row["domain"]
    if domain == "affine":
        problem = "Solve the equation established in the next line for real x."
        prefix = f"The equation is {a}*x + {b} = {a*(x+p)+b}."
        statement = f"x = {x+c}"
    elif domain == "multiplication":
        problem = "Compute the product of the two integers specified next."
        prefix = f"The two integers are {a} and {x+p}."
        statement = f"their product is {a*(x+c)}"
    elif domain == "inequality":
        problem = "Solve the inequality established in the next line for real x."
        sign = 1 if p == 0 else -1
        prefix = f"The inequality is {sign*a}*x < {sign*a*b}."
        statement = f"x {'<' if c == 0 else '>'} {b}"
    elif domain == "substitution":
        problem = f"Evaluate f(x) = x^2 + {b} at the value specified next."
        prefix = f"The input value is x = {x+p}."
        statement = f"f(x) = {(x+c)**2+b}"
    else:
        raise ValueError(domain)
    return problem, prefix, styled(statement, row["style"])


def build_dataset(config: dict) -> list[dict]:
    """Create deterministic complete families; keep all variants in one partition."""
    rng = random.Random(config["seed"])
    rows = []
    specifications = [("reversal", d, config["reversal_families_per_domain"])
                      for d in DOMAINS] + [("inheritance", "affine", config["inheritance_families"])]
    for arm, domain, count in specifications:
        witnesses = set()
        for index in range(count):
            while True:
                witness = (rng.randint(2, 9), rng.randint(2, 35), rng.randint(2, 25))
                a, b, x = witness
                # Keep the two mathematical contexts disjoint across families,
                # including neighboring numeric instances and dev/test partitions.
                if arm == "inheritance":
                    keys = {witness}
                elif domain == "inequality":
                    keys = {(a, b)}
                elif domain == "multiplication":
                    keys = {(a, x), (a, x+1)}
                elif domain == "substitution":
                    keys = {(b, x), (b, x+1)}
                else:
                    keys = {(a, b, x), (a, b, x+1)}
                if not keys & witnesses:
                    witnesses.update(keys)
                    break
            family = f"{arm}_{domain}_{index:03d}"
            partition = "dev" if index < int(count * config["dev_fraction"]) else "test"
            variants = [("target", p, c, s) for p in (0, 1) for c in (0, 1)
                        for s in config["styles"]]
            if arm == "inheritance":
                variants += [("before", p, -1, "plain") for p in (0, 1)]
            for role, p, c, style in variants:
                row = {"uid": f"{family}_{role}_p{p}_c{c}_{style}",
                       "family_id": family, "problem_id": family, "arm": arm,
                       "domain": domain, "partition": partition, "role": role,
                       "prefix_variant": p, "candidate_variant": c, "style": style,
                       "witness": dict(zip(("a", "b", "x"), witness)),
                       "step_idx": int(role == "target"), "solution_id": None}
                row.update(expected_labels(row))
                row["problem"], row["prefix"], row["candidate_step"] = render(row)
                rows.append(row)
    validate_dataset(rows, config)
    return rows


def validate_dataset(rows: list[dict], config: dict) -> None:
    """Reject changed texts, labels, duplicate IDs, and incomplete crossed families."""
    if not rows or len({r["uid"] for r in rows}) != len(rows):
        raise ValueError("empty data or duplicate uid")
    texts = {(r["problem"], r["prefix"], r["candidate_step"]) for r in rows}
    if len(texts) != len(rows):
        raise ValueError("duplicate rendered inputs across families")
    groups = defaultdict(list)
    for row in rows:
        if any(row[k] != v for k, v in expected_labels(row).items()):
            raise ValueError(f"incorrect mathematical label: {row['uid']}")
        if tuple(row[k] for k in ("problem", "prefix", "candidate_step")) != render(row):
            raise ValueError(f"text does not match witness: {row['uid']}")
        groups[row["family_id"]].append(row)
    for family, members in groups.items():
        if len({r["partition"] for r in members}) != 1:
            raise ValueError(f"family crosses partitions: {family}")
        expected = {("target", p, c, s) for p in (0, 1) for c in (0, 1)
                    for s in config["styles"]}
        if members[0]["arm"] == "inheritance":
            expected |= {("before", p, -1, "plain") for p in (0, 1)}
        actual = [(r["role"], r["prefix_variant"], r["candidate_variant"], r["style"])
                  for r in members]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(f"incomplete family: {family}")
    counts = Counter((m[0]["arm"], m[0]["domain"]) for m in groups.values())
    expected_counts = {("reversal", d): config["reversal_families_per_domain"] for d in DOMAINS}
    expected_counts[("inheritance", "affine")] = config["inheritance_families"]
    if dict(counts) != expected_counts:
        raise ValueError("family roster mismatch")


def family_metrics(rows: list[dict], scores: dict[str, float]) -> list[dict]:
    """Compute paired contrasts in error-probability units, without fitting."""
    if set(scores) != {r["uid"] for r in rows}:
        raise ValueError("scores must cover every uid exactly")
    if any(not np.isfinite(s) or not 0 <= s <= 1 for s in scores.values()):
        raise ValueError("expected finite P(incorrect) scores in [0,1]")
    groups = defaultdict(list)
    for row in rows:
        groups[row["family_id"]].append(row)
    output = []
    for family, members in groups.items():
        first = members[0]
        lookup = {(r["role"], r["prefix_variant"], r["candidate_variant"], r["style"]):
                  scores[r["uid"]] for r in members}
        for style in sorted({r["style"] for r in members if r["role"] == "target"}):
            s = {(p, c): lookup["target", p, c, style] for p in (0, 1) for c in (0, 1)}
            d0, d1 = s[0, 1] - s[0, 0], s[1, 0] - s[1, 1]
            metrics = {"local_contrast_valid_prefix": d0,
                       "local_contrast_other_prefix": d1,
                       "local_contrast_mean": (d0 + d1) / 2,
                       "both_preferences_correct": float(d0 > 0 and d1 > 0),
                       "any_tied_preference": float(d0 == 0 or d1 == 0)}
            if first["arm"] == "inheritance":
                before0 = lookup["before", 0, -1, "plain"]
                before1 = lookup["before", 1, -1, "plain"]
                inherited = (s[1, 1] + s[1, 0] - s[0, 0] - s[0, 1]) / 2
                metrics.update({"prefix_error_contrast": inherited,
                                "before_error_contrast": before1 - before0,
                                "prefix_contrast_after_minus_before": inherited - (before1 - before0),
                                "global_conclusion_contrast":
                                (s[0, 1] + s[1, 1] - s[0, 0] - s[1, 0]) / 2})
            if style != "plain":
                differences = [s[p, c] - lookup["target", p, c, "plain"]
                               for p in (0, 1) for c in (0, 1)]
                plain_d0 = lookup["target", 0, 1, "plain"] - lookup["target", 0, 0, "plain"]
                plain_d1 = lookup["target", 1, 0, "plain"] - lookup["target", 1, 1, "plain"]
                metrics.update({"style_mean_shift": float(np.mean(differences)),
                                "style_mean_absolute_shift": float(np.mean(np.abs(differences))),
                                "style_change_in_local_contrast": (d0+d1-plain_d0-plain_d1)/2})
            output.append({"family_id": family, "arm": first["arm"],
                           "domain": first["domain"], "partition": first["partition"],
                           "style": style, "metrics": metrics})
    return output


def summarize(metrics: list[dict], replicates: int, seed: int) -> list[dict]:
    """Bootstrap whole families within domain; report each partition separately."""
    groups = defaultdict(list)
    for row in metrics:
        groups[row["partition"], row["arm"], row["domain"], row["style"]].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(rows), size=(replicates, len(rows)))
        for metric in sorted(rows[0]["metrics"]):
            values = np.array([r["metrics"][metric] for r in rows])
            boot = values[indices].mean(axis=1)
            result.append(dict(zip(("partition", "arm", "domain", "style"), key),
                               metric=metric, n_families=len(rows), mean=float(values.mean()),
                               ci95=np.quantile(boot, [0.025, 0.975]).tolist()))
    return result
