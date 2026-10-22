import argparse
import csv
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment CSV: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Empty experiment CSV: {path}")
    return rows


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except KeyError, ValueError, TypeError:
        return float("nan")


def plot(experiment_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "baseline": "Unbounded / 1 synthetic consumer",
        "bounded1": "Backpressure / 1 synthetic consumer",
        "bounded2": "Backpressure / 2 synthetic consumers",
    }
    data = {name: read_csv(experiment_dir / f"{name}_producer.csv") for name in labels}
    for name, rows in data.items():
        elapsed = [number(row, "elapsed_seconds") for row in rows]
        if any(not math.isfinite(value) for value in elapsed) or elapsed != sorted(elapsed):
            raise ValueError(f"Invalid sample times: {name}")
    figures = [
        ("queue_depth.png", "queue_ready", "Ready messages in RabbitMQ"),
        ("producer_rate.png", "publish_per_second", "Confirmed publishes / second"),
        ("generation_rate.png", "generation_per_second", "Generated tasks / second"),
        ("producer_memory.png", "producer_peak_rss_mb", "Producer peak RSS (MB)"),
        ("broker_memory.png", "broker_memory_mb", "RabbitMQ node memory (MB)"),
        ("publish_latency.png", "mean_publish_latency_seconds", "Mean time to broker confirm (s)"),
    ]
    for filename, column, ylabel in figures:
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, label in labels.items():
            rows = data[name]
            measured_column = column
            if column == "queue_ready" and any(row.get("queue_ready_amqp") for row in rows):
                measured_column = "queue_ready_amqp"
            values = [number(row, measured_column) for row in rows]
            if not any(math.isfinite(value) for value in values):
                raise ValueError(f"No measurements for {name}: {measured_column}")
            ax.plot(
                [number(row, "elapsed_seconds") for row in rows],
                values,
                label=label,
                drawstyle="steps-post" if column == "queue_ready" else "default",
            )
        ax.set(xlabel="Elapsed seconds", ylabel=ylabel)
        ax.set_title("Synthetic consumer + real RabbitMQ (not ML inference)")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(experiment_dir / filename, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, label in labels.items():
        worker_files = sorted(experiment_dir.glob(f"{name}_worker*.csv"))
        observations = []
        for worker_file in worker_files:
            observations.extend(read_csv(worker_file))
        observations.sort(key=lambda row: row["timestamp_utc"])
        if observations:
            ax.plot(
                range(1, len(observations) + 1),
                [number(row, "task_latency_seconds") for row in observations],
                label=label,
                alpha=0.8,
            )
    ax.set(
        xlabel="Completed task index",
        ylabel="Created-to-completion task latency (s)",
        title="Synthetic consumer + real RabbitMQ (not ML inference)",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(experiment_dir / "task_latency.png", dpi=150)
    plt.close(fig)
    for column, filename, ylabel in (
        ("worker_peak_rss_mb", "worker_memory.png", "Demo worker peak RSS (MB)"),
        ("processing_seconds", "processing_time.png", "Actual demo processing time (s)"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, label in labels.items():
            observations = []
            for worker_file in sorted(experiment_dir.glob(f"{name}_worker*.csv")):
                observations.extend(read_csv(worker_file))
            observations.sort(key=lambda row: row["timestamp_utc"])
            if observations:
                ax.plot(
                    range(1, len(observations) + 1),
                    [number(row, column) for row in observations],
                    label=label,
                )
        ax.set(xlabel="Completed task index", ylabel=ylabel)
        ax.set_title("Synthetic consumer + real RabbitMQ (not ML inference)")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(experiment_dir / filename, dpi=150)
        plt.close(fig)
    print(f"Figures saved to {experiment_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.experiment_dir)


if __name__ == "__main__":
    main()
