"""Prepare binary MultiFC splits for training and final evaluation.

Keep only the original true/false labels, remove duplicate and conflicting
claims, and reserve the original dev split for final evaluation.

    poetry run python experiments/prepare_data.py
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "multi_fc"
OUTPUT_DIR = DATA_DIR / "prepared"
LABELS = ("true", "false")
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    binary = frame["label"].isin(LABELS)
    stats = {
        "raw_rows": len(frame),
        "removed_non_binary": int((~binary).sum()),
        "excluded_labels": frame.loc[~binary, "label"].fillna("<missing>").value_counts().to_dict(),
    }
    frame = frame.loc[binary, ["claim", "label"]].rename(columns={"claim": "text"}).copy()
    frame["text"] = frame["text"].fillna("").str.strip()
    empty = frame["text"].eq("")
    stats["removed_empty_text"] = int(empty.sum())
    frame = frame.loc[~empty].copy()
    # Preserve the text for inference; use a normalized key only for deduplication.
    frame["text_key"] = (
        frame["text"].str.normalize("NFKC").str.replace(r"\s+", " ", regex=True).str.casefold()
    )
    return frame, stats


def prepare_splits(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    cleaned = {}
    cleaning = {}
    for name in ("train", "dev"):
        cleaned[name], cleaning[name] = clean_split(frames[name])

    combined = pd.concat(cleaned.values(), ignore_index=True)
    label_counts = combined.groupby("text_key")["label"].nunique()
    conflicts = label_counts[label_counts > 1].index

    for name, frame in cleaned.items():
        conflicting = frame["text_key"].isin(conflicts)
        cleaning[name]["removed_conflicting_labels"] = int(conflicting.sum())
        frame = frame.loc[~conflicting]
        duplicated = frame["text_key"].duplicated()
        cleaning[name]["removed_duplicates"] = int(duplicated.sum())
        cleaned[name] = frame.loc[~duplicated].copy()

    # Preserve the final evaluation set; remove shared claims from training.
    overlap = cleaned["train"]["text_key"].isin(cleaned["dev"]["text_key"])
    cleaning["train"]["removed_overlap_with_dev"] = int(overlap.sum())
    cleaned["train"] = cleaned["train"].loc[~overlap]

    for name, frame in cleaned.items():
        if set(frame["label"]) != set(LABELS):
            raise ValueError(f"Cleaned {name} must contain both true and false labels")

    train, validation = train_test_split(
        cleaned["train"],
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=cleaned["train"]["label"],
    )
    splits = {
        name: frame[["text", "label"]].reset_index(drop=True)
        for name, frame in {
            "train": train,
            "validation": validation,
            "test": cleaned["dev"],
        }.items()
    }
    report = {
        "label_rule": {"keep": list(LABELS), "other_labels": "exclude"},
        "text_key": "NFKC normalization, collapsed whitespace, casefold",
        "conflicting_texts": len(conflicts),
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "cleaning": cleaning,
        "splits": {
            name: {
                "source_split": "dev" if name == "test" else "train",
                "rows": len(frame),
                "class_counts": {label: int(frame["label"].eq(label).sum()) for label in LABELS},
            }
            for name, frame in splits.items()
        },
    }
    return splits, report


def main() -> None:
    # The original test split has no labels and is not used here.
    frames = {
        name: pd.read_csv(
            DATA_DIR / f"{name}.csv",
            usecols=["claim", "label"],
            dtype={"claim": "str", "label": "str"},
            na_values=["None"],
        )
        for name in ("train", "dev")
    }
    splits, report = prepare_splits(frames)
    report["source_files"] = {
        name: {"path": f"{name}.csv", "sha256": file_sha256(DATA_DIR / f"{name}.csv")}
        for name in frames
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        counts = report["splits"][name]["class_counts"]
        print(f"{name}: {len(frame)} rows, {counts}")

    report_path = OUTPUT_DIR / "summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Preparation report: {report_path}")


if __name__ == "__main__":
    main()
