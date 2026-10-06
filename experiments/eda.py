"""EDA of the MultiFC dataset, tracked in MLflow.

Registers the three splits as MLflow datasets and logs plots, tables and a
summary as run artifacts.

    poetry run python experiments/eda.py
"""

import hashlib
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from matplotlib.figure import Figure

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "multi_fc"
SPLITS = ("train", "dev", "test")

# Pinned commit of the Hugging Face repo, so the source always points at the same bytes.
HF_REPO = "pszemraj/multi_fc"
HF_REVISION = "210f476c07dd5f486a9a51311e193a51c1ff8aec"
HF_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{split}.csv"

EXPERIMENT = "multi_fc_eda"
TOP_LABELS = 20
TOP_SITES = 8
HEATMAP_LABELS = 12

BAR_COLOR = "#2a6fbb"
ACCENT_COLOR = "#c2410c"
GRID_COLOR = "#d9d9d9"


def load_split(split: str) -> pd.DataFrame:
    # Fields that could not be crawled are stored as the literal string "None".
    df = pd.read_csv(DATA_DIR / f"{split}.csv", na_values=["None"])
    # claimID looks like "pomt-03627": the prefix is the fact-checking site.
    df["site"] = df["claimID"].str.split("-").str[0]
    return df


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_datasets(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Register every split in MLflow and link it to the active run."""
    digests = {}
    for split, df in frames.items():
        dataset = mlflow.data.from_pandas(
            df.drop(columns="site"),
            source=HF_URL.format(repo=HF_REPO, revision=HF_REVISION, split=split),
            name=f"multi_fc_{split}",
            # Test labels were never published, so the test split has no target.
            targets="label" if split != "test" else None,
        )
        mlflow.log_input(dataset, context="eda", tags={"split": split})
        mlflow.log_param(f"sha256_{split}", file_sha256(DATA_DIR / f"{split}.csv"))
        digests[split] = dataset.digest
    return digests


def style_axes(ax: plt.Axes, grid_axis: str) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_label_distribution(train: pd.DataFrame) -> Figure:
    counts = train["label"].value_counts()
    top = counts.head(TOP_LABELS).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top.index, top.to_numpy(), color=BAR_COLOR)
    ax.set_title(f"Top {TOP_LABELS} of {counts.size} labels in train")
    ax.set_xlabel("claims")
    style_axes(ax, "x")
    fig.tight_layout()
    return fig


def plot_sites(train: pd.DataFrame) -> Figure:
    per_site = train.groupby("site").agg(claims=("claimID", "size"), labels=("label", "nunique"))
    per_site = per_site.sort_values("claims")

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 7), sharey=True)
    left.barh(per_site.index, per_site["claims"], color=BAR_COLOR)
    left.set_title("Claims per site")
    left.set_xlabel("claims")
    right.barh(per_site.index, per_site["labels"], color=BAR_COLOR)
    right.set_title("Distinct labels per site")
    right.set_xlabel("labels")
    for ax in (left, right):
        style_axes(ax, "x")
    fig.tight_layout()
    return fig


def plot_label_by_site(train: pd.DataFrame) -> Figure:
    sites = train["site"].value_counts().head(TOP_SITES).index
    labels = train["label"].value_counts().head(HEATMAP_LABELS).index
    subset = train[train["site"].isin(sites)]
    # Share of each label inside a site: rows sum to at most 1.
    share = pd.crosstab(subset["site"], subset["label"], normalize="index")
    share = share.reindex(index=sites, columns=labels, fill_value=0)
    # Everything outside the global top labels, so that every row sums to 100%.
    share["other labels"] = 1 - share.sum(axis=1)
    labels = share.columns

    fig, ax = plt.subplots(figsize=(11, 5.5))
    image = ax.imshow(share.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(sites)), sites)
    for row in range(share.shape[0]):
        for col in range(share.shape[1]):
            value = share.iat[row, col]
            if value >= 0.05:
                ax.text(
                    col,
                    row,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > 0.5 else "black",
                )
    ax.set_title(f"Label share inside the {TOP_SITES} largest sites")
    fig.colorbar(image, ax=ax, label="share of the site's claims")
    fig.tight_layout()
    return fig


def plot_claim_length(train: pd.DataFrame) -> tuple[Figure, pd.Series]:
    words = train["claim"].dropna().str.split().str.len()
    median, p95, p99 = words.quantile([0.5, 0.95, 0.99])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(words.clip(upper=p99), bins=range(1, int(p99) + 2), color=BAR_COLOR)
    ax.axvline(median, color=ACCENT_COLOR, linewidth=2, label=f"median = {median:.0f}")
    ax.axvline(p95, color=ACCENT_COLOR, linewidth=2, linestyle="--", label=f"95th pct = {p95:.0f}")
    ax.set_title("Claim length in words (train, clipped at the 99th percentile)")
    ax.set_xlabel("words")
    ax.set_ylabel("claims")
    ax.legend(frameon=False)
    style_axes(ax, "y")
    fig.tight_layout()
    return fig, words


def plot_missing(train: pd.DataFrame) -> tuple[Figure, pd.Series]:
    missing = train.drop(columns="site").isna().mean().sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(missing.index, missing.to_numpy(), color=BAR_COLOR)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_title("Share of missing values per column (train)")
    style_axes(ax, "x")
    fig.tight_layout()
    return fig, missing


def build_summary(metrics: dict[str, float], digests: dict[str, str]) -> str:
    return f"""# MultiFC EDA summary

Source: https://huggingface.co/datasets/{HF_REPO} at commit `{HF_REVISION}`.
MLflow digests: {", ".join(f"{split} `{digest}`" for split, digest in digests.items())}.

## Findings

- **Size.** train {metrics["rows_train"]:.0f}, dev {metrics["rows_dev"]:.0f}, test {metrics["rows_test"]:.0f} rows.
- **Labels are not unified.** train has {metrics["labels_train"]:.0f} distinct labels from
  {metrics["sites_train"]:.0f} fact-checking sites, each with its own rating scale. The most
  frequent label covers {metrics["majority_share_train"]:.1%} of train, so accuracy is
  misleading and macro F1 is the right metric.
- **Labels depend on the site.** The heatmap shows that sites barely share labels: a model
  can guess the scale from the site alone. Either map labels to a common scale or train on
  one site (PolitiFact, `pomt`, is the largest).
- **Test has no labels.** {metrics["labelled_share_test"]:.0%} of test rows are labelled, so
  evaluation is only possible on dev.
- **Claims are short.** Median {metrics["claim_words_median"]:.0f} words, 95th percentile
  {metrics["claim_words_p95"]:.0f}, 99th percentile {metrics["claim_words_p99"]:.0f}. Words are not tokens,
  but a max length of 128 subword tokens should cover nearly all claims.
- **Metadata is sparse.** {metrics["columns_over_half_missing"]:.0f} of 13 columns are more
  than half empty in train, so they are weak features.
- **`reason` leaks the target.** It holds the fact-check article with the verdict
  ({metrics["reason_missing_share"]:.0%} missing). Do not use it as a feature.
- **Duplicates.** {metrics["duplicate_claims_train"]:.0f} repeated claims in train,
  {metrics["claims_shared_train_dev"]:.0f} claims shared between train and dev.
"""


def main() -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT)

    frames = {split: load_split(split) for split in SPLITS}
    train, dev, test = (frames[split] for split in SPLITS)

    with mlflow.start_run(run_name="eda"):
        digests = log_datasets(frames)
        mlflow.log_params({"hf_repo": HF_REPO, "hf_revision": HF_REVISION})

        label_counts = train["label"].value_counts()
        length_fig, claim_words = plot_claim_length(train)
        missing_fig, missing = plot_missing(train)
        figures = {
            "label_distribution": plot_label_distribution(train),
            "sites": plot_sites(train),
            "label_by_site": plot_label_by_site(train),
            "claim_length": length_fig,
            "missing_values": missing_fig,
        }
        for name, figure in figures.items():
            mlflow.log_figure(figure, f"eda/{name}.png")
            plt.close(figure)

        metrics = {
            **{f"rows_{split}": float(len(df)) for split, df in frames.items()},
            "labels_train": float(label_counts.size),
            "labels_dev": float(dev["label"].nunique()),
            "sites_train": float(train["site"].nunique()),
            "majority_share_train": float(label_counts.iloc[0] / label_counts.sum()),
            "labelled_share_test": float(test["label"].notna().mean()),
            "claim_words_median": float(claim_words.median()),
            "claim_words_p95": float(claim_words.quantile(0.95)),
            "claim_words_p99": float(claim_words.quantile(0.99)),
            "columns_over_half_missing": float((missing > 0.5).sum()),
            "reason_missing_share": float(missing["reason"]),
            "duplicate_claims_train": float(train["claim"].duplicated().sum()),
            "claims_shared_train_dev": float(len(set(train["claim"]) & set(dev["claim"]))),
        }
        mlflow.log_metrics(metrics)

        mlflow.log_table(
            label_counts.rename_axis("label").reset_index(name="claims"),
            "eda/label_counts.json",
        )
        mlflow.log_table(
            missing.rename_axis("column").reset_index(name="missing_share"),
            "eda/missing_values.json",
        )
        mlflow.log_text(build_summary(metrics, digests), "eda/summary.md")


if __name__ == "__main__":
    main()
