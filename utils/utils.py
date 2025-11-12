import json
from pathlib import Path
import pandas as pd
from enum import Enum

class OutputFormat(Enum):
    FLAT = 1
    ORIGINAL = 2


def serialize_itemsets(itemsets: dict) -> dict:
    out = []
    for length, level in itemsets.items():
        for items, support in level.items():
            out.append({
                "length": length,
                "items": list(items),
                "support": support
            })
    return out

def save_itemsets(itemsets: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(serialize_itemsets(itemsets), f)





def load_itemsets(path: Path, output_format=OutputFormat.FLAT) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    match output_format:
        case OutputFormat.FLAT:
            return data

        case OutputFormat.ORIGINAL:
            # rebuild {length: {tuple(items): support}}
            out = {}
            for rec in data:
                length = rec["length"]
                items = tuple(rec["items"])
                support = rec["support"]
                out.setdefault(length, {})[items] = support
            return out

        case _:
            raise ValueError("output_format must be 'flat' or 'original'")
    

import json
from pathlib import Path


def serialize_rules(rules):
    out = []
    for r in rules:
        antecedent = list(getattr(r, "antecedent", getattr(r, "lhs", []))) if hasattr(r, "antecedent") or hasattr(r, "lhs") else list(r[0]) if isinstance(r, (list, tuple)) else []
        consequent = list(getattr(r, "consequent", getattr(r, "rhs", []))) if hasattr(r, "consequent") or hasattr(r, "rhs") else list(r[1]) if isinstance(r, (list, tuple)) else []
        support = float(getattr(r, "support", 0))
        confidence = float(getattr(r, "confidence", 0))
        lift = float(getattr(r, "lift", getattr(r, "lift_", 0)))
        out.append({
            "antecedent": antecedent,
            "consequent": consequent,
            "support": support,
            "confidence": confidence,
            "lift": lift
        })
    return out


def save_rules(rules, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialize_rules(rules), f, indent=2)


def load_rules(path: Path, output_format=OutputFormat.FLAT) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    match output_format:
        case OutputFormat.FLAT:
            return data

        case "tuple":
            return [
                (
                    tuple(rec["antecedent"]),
                    tuple(rec["consequent"]),
                    rec.get("support", 0),
                    rec.get("confidence", 0),
                    rec.get("lift", 0),
                )
                for rec in data
            ]

        case _:
            raise ValueError("output_format must be 'flat' or 'tuple'")
