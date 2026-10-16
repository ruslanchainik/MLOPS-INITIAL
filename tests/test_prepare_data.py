import pandas as pd
import pytest
from experiments.prepare_data import prepare_splits
from pandas.testing import assert_frame_equal


def source_frame(prefix: str, rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "claim": [f"{prefix} claim {index}" for index in range(rows)],
            "label": ["true" if index % 2 == 0 else "false" for index in range(rows)],
        }
    )


def test_preparation_removes_conflicts_duplicates_and_overlap():
    train = pd.concat(
        [
            source_frame("train"),
            pd.DataFrame(
                {
                    "claim": [
                        "TRAIN   claim 2",
                        "\uff34\uff32\uff21\uff29\uff2e claim 2",  # Fullwidth uppercase TRAIN.
                        "  ",
                        None,
                        "Unknown",
                    ],
                    "label": ["true", "true", "false", "true", "half-true"],
                }
            ),
        ],
        ignore_index=True,
    )
    dev = pd.concat(
        [
            source_frame("dev", 6),
            pd.DataFrame(
                {
                    "claim": ["train claim 0", "train claim 3", "DEV claim 0"],
                    "label": ["false", "false", "true"],
                }
            ),
        ],
        ignore_index=True,
    )

    splits, report = prepare_splits({"train": train, "dev": dev})

    assert report["conflicting_texts"] == 1
    assert report["cleaning"]["train"]["removed_conflicting_labels"] == 1
    assert report["cleaning"]["dev"]["removed_conflicting_labels"] == 1
    assert report["cleaning"]["train"]["removed_duplicates"] == 2
    assert report["cleaning"]["dev"]["removed_duplicates"] == 1
    assert report["cleaning"]["train"]["removed_empty_text"] == 2
    assert report["cleaning"]["train"]["removed_non_binary"] == 1
    assert report["cleaning"]["train"]["removed_overlap_with_dev"] == 1

    texts = {name: set(frame["text"]) for name, frame in splits.items()}
    assert texts["train"].isdisjoint(texts["validation"])
    assert texts["train"].isdisjoint(texts["test"])
    assert texts["validation"].isdisjoint(texts["test"])
    assert "train claim 0" not in set.union(*texts.values())
    assert "train claim 3" in texts["test"]
    assert "train claim 3" not in texts["train"] | texts["validation"]
    assert len(splits["train"]) + len(splits["validation"]) == 18
    assert len(splits["test"]) == 7
    for frame in splits.values():
        assert list(frame.columns) == ["text", "label"]
        assert set(frame["label"]) == {"true", "false"}


def test_preparation_is_reproducible_and_keeps_holdout_separate():
    frames = {"train": source_frame("train", 40), "dev": source_frame("dev", 10)}

    first, report = prepare_splits(frames)
    second, _ = prepare_splits(frames)

    for name in first:
        assert_frame_equal(first[name], second[name])
    assert len(first["train"]) == 32
    assert first["validation"]["label"].value_counts().to_dict() == {"true": 4, "false": 4}
    assert_frame_equal(first["test"], frames["dev"].rename(columns={"claim": "text"}))
    assert report["splits"]["test"]["source_split"] == "dev"


@pytest.mark.parametrize("source", ["train", "dev"])
def test_preparation_rejects_source_with_only_one_binary_class(source):
    frames = {"train": source_frame("train"), "dev": source_frame("dev")}
    frames[source]["label"] = "false"

    with pytest.raises(ValueError, match=f"Cleaned {source} must contain both"):
        prepare_splits(frames)
