"""EDA of the Webis Clickbait Corpus 2017, tracked in MLflow.

Registers the corpus as an MLflow dataset and logs plots, tables and a
summary as run artifacts.

    poetry run python experiments/eda_clickbait.py
"""

import hashlib
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "clickbait17"
FILES = ("instances.jsonl", "truth.jsonl")

ZENODO_RECORD = "5530410"
ZENODO_FILE = "clickbait17-train-170630.zip"
ZENODO_MD5 = "8f074f3f035393b81a559bfc8bd9a467"
SOURCE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{ZENODO_FILE}"

EXPERIMENT = "clickbait17_eda"
CLASSES = ("no-clickbait", "clickbait")

CUES = {
    "question mark": r"\?",
    "exclamation mark": r"!",
    "starts with a number": r"^\d",
    "you / your": r"\byou(?:r)?\b",
    "this / these": r"\b(?:this|these)\b",
    "hashtag or mention": r"[#@]",
}

CLASS_COLORS = {"no-clickbait": "#2a6fbb", "clickbait": "#c2410c"}
BAR_COLOR = "#2a6fbb"
ACCENT_COLOR = "#c2410c"
GRID_COLOR = "#d9d9d9"


def load_corpus() -> pd.DataFrame:
    instances = pd.read_json(DATA_DIR / "instances.jsonl", lines=True, dtype={"id": str})
    truth = pd.read_json(DATA_DIR / "truth.jsonl", lines=True, dtype={"id": str})
    raw = instances.merge(truth, on="id", validate="one_to_one")

    judgments = np.vstack(raw["truthJudgments"].to_numpy()).round(2)
    judges_agree = [np.unique(row, return_counts=True)[1].max() for row in judgments]
    # MLflow cannot infer a schema for list columns.
    return pd.DataFrame(
        {
            "id": raw["id"],
            "post_text": raw["postText"].str[0].fillna(""),
            "post_timestamp": pd.to_datetime(
                raw["postTimestamp"], format="%a %b %d %H:%M:%S %z %Y"
            ),
            "has_media": raw["postMedia"].str.len() > 0,
            "target_title": raw["targetTitle"],
            "target_description": raw["targetDescription"],
            "target_keywords": raw["targetKeywords"],
            "target_paragraphs": raw["targetParagraphs"].str.len(),
            "target_captions": raw["targetCaptions"].str.len(),
            "judges_agree": judges_agree,
            "truth_mean": raw["truthMean"],
            "truth_median": raw["truthMedian"],
            "truth_class": raw["truthClass"],
        }
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_dataset(df: pd.DataFrame) -> str:
    """Register the corpus in MLflow and link it to the active run."""
    dataset = mlflow.data.from_pandas(
        df,
        source=SOURCE_URL,
        name="clickbait17_train",
        targets="truth_class",
    )
    mlflow.log_input(dataset, context="eda", tags={"split": "train"})
    mlflow.log_param("zenodo_md5", ZENODO_MD5)
    for name in FILES:
        mlflow.log_param(f"sha256_{Path(name).stem}", file_sha256(DATA_DIR / name))
    return dataset.digest


def style_axes(ax: plt.Axes, grid_axis: str) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_class_balance(df: pd.DataFrame) -> Figure:
    counts = df["truth_class"].value_counts().reindex(CLASSES)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.to_numpy(), color=[CLASS_COLORS[c] for c in CLASSES])
    ax.bar_label(bars, labels=[f"{n} ({n / counts.sum():.0%})" for n in counts])
    ax.set_title("Class balance")
    ax.set_ylabel("posts")
    style_axes(ax, "y")
    fig.tight_layout()
    return fig


def plot_truth_mean(df: pd.DataFrame) -> Figure:
    # Mean of 5 ratings from {0, 1/3, 2/3, 1} takes 16 values.
    edges = np.arange(17) / 15 - 1 / 30

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        [df.loc[df["truth_class"] == cls, "truth_mean"] for cls in CLASSES],
        bins=edges,
        stacked=True,
        color=[CLASS_COLORS[cls] for cls in CLASSES],
        label=list(CLASSES),
        rwidth=0.9,
    )
    ax.axvline(0.5, color="black", linewidth=1, linestyle="--")
    ax.set_title("Mean clickbait score of the 5 judges, stacked by class")
    ax.set_xlabel("truthMean")
    ax.set_ylabel("posts")
    ax.legend(frameon=False)
    style_axes(ax, "y")
    fig.tight_layout()
    return fig


def plot_agreement(df: pd.DataFrame) -> Figure:
    shares = pd.crosstab(df["judges_agree"], df["truth_class"], normalize="columns")
    shares = shares.reindex(index=[2, 3, 4, 5], columns=list(CLASSES), fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(shares))
    width = 0.38
    for offset, cls in zip((-width / 2, width / 2), CLASSES, strict=True):
        ax.bar(x + offset, shares[cls], width=width, color=CLASS_COLORS[cls], label=cls)
    ax.set_xticks(x, [f"{n} of 5" for n in shares.index])
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_title("Judges who gave the most common rating")
    ax.set_xlabel("5 of 5 = unanimous")
    ax.set_ylabel("share of the class")
    ax.legend(frameon=False)
    style_axes(ax, "y")
    fig.tight_layout()
    return fig


def plot_post_length(df: pd.DataFrame) -> tuple[Figure, pd.Series]:
    words = df["post_text"].str.split().str.len()
    p99 = int(words.quantile(0.99))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cls in CLASSES:
        values = words[df["truth_class"] == cls].clip(upper=p99)
        weights = np.full(len(values), 1 / len(values))
        ax.hist(
            values,
            bins=range(0, p99 + 2),
            weights=weights,
            histtype="step",
            linewidth=2,
            color=CLASS_COLORS[cls],
            label=f"{cls}, median {words[df['truth_class'] == cls].median():.0f}",
        )
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_title("Post length in words (clipped at the 99th percentile)")
    ax.set_xlabel("words")
    ax.set_ylabel("share of the class")
    ax.legend(frameon=False)
    style_axes(ax, "y")
    fig.tight_layout()
    return fig, words


def plot_cues(df: pd.DataFrame) -> tuple[Figure, pd.DataFrame]:
    flags = pd.DataFrame(
        {
            name: df["post_text"].str.contains(pattern, case=False, regex=True)
            for name, pattern in CUES.items()
        }
    )
    flags["has media"] = df["has_media"]
    shares = flags.groupby(df["truth_class"]).mean().T[list(CLASSES)]
    shares = shares.sort_values("clickbait")

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(shares))
    height = 0.38
    for offset, cls in zip((-height / 2, height / 2), CLASSES, strict=True):
        ax.barh(y + offset, shares[cls], height=height, color=CLASS_COLORS[cls], label=cls)
    ax.set_yticks(y, shares.index)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_title("Share of posts with a surface cue, by class")
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax, "x")
    fig.tight_layout()
    return fig, shares


def plot_empty_fields(df: pd.DataFrame) -> tuple[Figure, pd.Series]:
    empty = pd.Series(
        {
            "post_text": (df["post_text"].str.strip() == "").mean(),
            "post_media": (~df["has_media"]).mean(),
            "target_title": (df["target_title"].fillna("") == "").mean(),
            "target_description": (df["target_description"].fillna("") == "").mean(),
            "target_keywords": (df["target_keywords"].fillna("") == "").mean(),
            "target_paragraphs": (df["target_paragraphs"] == 0).mean(),
            "target_captions": (df["target_captions"] == 0).mean(),
        }
    ).sort_values()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(empty.index, empty.to_numpy(), color=BAR_COLOR)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_title("Share of empty fields")
    style_axes(ax, "x")
    fig.tight_layout()
    return fig, empty


def build_summary(metrics: dict[str, float], digest: str) -> str:
    return f"""# Webis-Clickbait-17 EDA summary

Source: {SOURCE_URL} (Zenodo record {ZENODO_RECORD}, CC BY 4.0), files `instances.jsonl`
and `truth.jsonl` from the archive. MLflow digest: `{digest}`.

## Findings

- **Size.** {metrics["rows"]:.0f} posts, posted between December 2016 and April 2017. The
  archive is the only labelled split here, so train/validation/test must be cut from it.
- **Imbalance.** {metrics["clickbait_share"]:.1%} of posts are clickbait. A constant
  "no-clickbait" model scores {1 - metrics["clickbait_share"]:.0%} accuracy, so report F1 of
  the clickbait class or macro F1.
- **The class is the median vote.** `truthClass` matches `truthMedian >= 0.5` on every post,
  while `truthMean >= 0.5` disagrees on {metrics["mean_rule_mismatches"]:.0f} posts. The scores
  around 0.5 are a grey zone; `truthMean` can be used as a regression target.
- **Judges often disagree.** Only {metrics["unanimous_share"]:.1%} of posts got the same rating
  from all five judges ({metrics["unanimous_share_clickbait"]:.1%} of clickbait posts). Label
  noise caps achievable quality.
- **Clickbait posts are shorter.** Median {metrics["words_median_clickbait"]:.0f} words versus
  {metrics["words_median_no_clickbait"]:.0f}.
- **Surface cues are strong signals.** "you / your" appears in
  {metrics["cue_you_clickbait"]:.1%} of clickbait posts versus
  {metrics["cue_you_no_clickbait"]:.1%} of the others; a leading number, a question mark and
  "this / these" show the same pattern. A bag-of-words baseline should already work.
- **Data quality.** {metrics["empty_posts"]:.0f} posts have empty text,
  {metrics["duplicate_posts"]:.0f} post texts are repeated, and
  {metrics["duplicate_groups_conflicting"]:.0f} groups of identical texts carry different
  classes. Deduplicate before splitting so that copies do not leak between train and test.
"""


def main() -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT)

    df = load_corpus()
    is_clickbait = df["truth_class"] == "clickbait"

    with mlflow.start_run(run_name="eda"):
        digest = log_dataset(df)
        mlflow.log_params({"zenodo_record": ZENODO_RECORD, "zenodo_file": ZENODO_FILE})

        length_fig, words = plot_post_length(df)
        cues_fig, cue_shares = plot_cues(df)
        empty_fig, empty = plot_empty_fields(df)
        figures = {
            "class_balance": plot_class_balance(df),
            "truth_mean": plot_truth_mean(df),
            "judge_agreement": plot_agreement(df),
            "post_length": length_fig,
            "surface_cues": cues_fig,
            "empty_fields": empty_fig,
        }
        for name, figure in figures.items():
            mlflow.log_figure(figure, f"eda/{name}.png")
            plt.close(figure)

        texts = df.loc[df["post_text"].str.strip() != "", ["post_text", "truth_class"]]
        duplicated = texts[texts["post_text"].duplicated(keep=False)]
        unanimous = df["judges_agree"] == 5
        metrics = {
            "rows": float(len(df)),
            "clickbait_share": float(is_clickbait.mean()),
            "mean_rule_mismatches": float(((df["truth_mean"] >= 0.5) != is_clickbait).sum()),
            "unanimous_share": float(unanimous.mean()),
            "unanimous_share_clickbait": float(unanimous[is_clickbait].mean()),
            "words_median_clickbait": float(words[is_clickbait].median()),
            "words_median_no_clickbait": float(words[~is_clickbait].median()),
            "cue_you_clickbait": float(cue_shares.loc["you / your", "clickbait"]),
            "cue_you_no_clickbait": float(cue_shares.loc["you / your", "no-clickbait"]),
            "empty_posts": float((df["post_text"].str.strip() == "").sum()),
            "duplicate_posts": float(texts["post_text"].duplicated().sum()),
            "duplicate_groups_conflicting": float(
                (duplicated.groupby("post_text")["truth_class"].nunique() > 1).sum()
            ),
        }
        mlflow.log_metrics(metrics)

        mlflow.log_table(
            cue_shares.rename_axis("cue").reset_index(),
            "eda/surface_cues.json",
        )
        mlflow.log_table(
            empty.rename_axis("field").reset_index(name="empty_share"),
            "eda/empty_fields.json",
        )
        mlflow.log_text(build_summary(metrics, digest), "eda/summary.md")


if __name__ == "__main__":
    main()
