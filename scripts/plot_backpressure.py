import argparse
import csv
import math
from datetime import datetime
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty experiment CSV: {path}")
    return rows


def plot(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "baseline": "Unbounded / 1 worker",
        "bounded1": "Bounded / 1 worker",
        "bounded2": "Bounded / 2 workers",
    }
    data = {}
    for name in labels:
        producer = read_csv(directory / f"{name}_producer.csv")
        workers = [
            row for path in sorted(directory.glob(f"{name}_worker*.csv")) for row in read_csv(path)
        ]
        workers.sort(key=lambda row: row["timestamp_utc"])
        if not workers:
            raise ValueError(f"Missing worker measurements: {name}")
        data[name] = producer, workers
    groups = {
        "queue_and_rate.png": [
            ("queue_ready_amqp", "Ready messages", False),
            ("generation_per_second", "Generated tasks / second", False),
            ("publish_per_second", "Confirmed publishes / second", False),
        ],
        "memory.png": [
            ("producer_peak_rss_mb", "Producer peak RSS (MiB)", False),
            ("broker_memory_mb", "RabbitMQ node memory (MiB)", False),
            ("worker_peak_rss_mb", "Worker peak RSS (MiB)", True),
        ],
        "latency.png": [
            ("mean_publish_latency_seconds", "Mean creation-to-confirm time (s)", False),
            ("task_latency_seconds", "Creation-to-completion time (s)", True),
            ("processing_seconds", "ML processing time incl. configured delay (s)", True),
        ],
    }
    for filename, columns in groups.items():
        figure, axes = plt.subplots(3, 1, figsize=(11, 10))
        for axis, (column, title, use_worker) in zip(axes, columns, strict=True):
            for name, label in labels.items():
                producer, workers = data[name]
                rows = workers if use_worker else producer
                start = datetime.fromisoformat(producer[0]["timestamp_utc"])
                elapsed = [
                    (datetime.fromisoformat(row["timestamp_utc"]) - start).total_seconds()
                    for row in rows
                ]
                values = [float(row[column]) if row.get(column) else math.nan for row in rows]
                if not any(math.isfinite(value) for value in values):
                    raise ValueError(f"Missing measurements: {name}/{column}")
                axis.plot(elapsed, values, label=label, alpha=0.8)
            axis.set(xlabel="Elapsed seconds", ylabel=title)
            axis.grid(alpha=0.2)
            axis.legend()
        figure.suptitle("RabbitMQ → FastStream → MLflow → PostgreSQL")
        figure.tight_layout()
        figure.savefig(directory / filename, dpi=130)
        plt.close(figure)
    print(f"Figures saved to {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    plot(parser.parse_args().experiment_dir)
