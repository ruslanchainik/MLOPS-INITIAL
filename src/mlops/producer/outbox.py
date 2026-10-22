import fcntl
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from mlops.messaging import TaskMessage


@dataclass(frozen=True, slots=True)
class PendingTask:
    message: TaskMessage
    created_at_epoch: float
    sequence: int


class SQLiteOutbox:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            self.path = str(Path(self.path).expanduser().resolve())
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db: sqlite3.Connection | None = None
        self._lock = None

    def __enter__(self) -> SQLiteOutbox:
        if self.path != ":memory:":
            self._lock = open(self.path + ".lock", "a")
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self._lock.close()
                self._lock = None
                raise RuntimeError("Outbox is already in use by another producer") from exc
        try:
            return self._open()
        except BaseException:
            self.__exit__()
            raise

    def _open(self) -> SQLiteOutbox:
        self._db = sqlite3.connect(self.path)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS producer_batches (
                batch_id TEXT PRIMARY KEY,
                target_count INTEGER NOT NULL,
                text TEXT NOT NULL,
                queue_name TEXT NOT NULL,
                max_length INTEGER,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS producer_tasks (
                batch_id TEXT NOT NULL REFERENCES producer_batches(batch_id),
                sequence INTEGER NOT NULL,
                task_id TEXT NOT NULL UNIQUE,
                created_at REAL NOT NULL,
                confirmed_at REAL,
                PRIMARY KEY (batch_id, sequence)
            );
            CREATE INDEX IF NOT EXISTS producer_pending
                ON producer_tasks(batch_id, sequence) WHERE confirmed_at IS NULL;
            """
        )
        return self

    def __exit__(self, *_: object) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    @property
    def db(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError("Outbox must be used as a context manager")
        return self._db

    def open_batch(
        self, *, batch_id: str, count: int, text: str, queue_name: str, max_length: int | None
    ) -> None:
        if not batch_id.strip():
            raise ValueError("batch_id must not be empty")
        if count <= 0:
            raise ValueError("count must be positive")
        if not text.strip():
            raise ValueError("text must not be empty")
        with self.db:
            self.db.execute(
                """INSERT OR IGNORE INTO producer_batches
                   (batch_id, target_count, text, queue_name, max_length, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (batch_id, count, text, queue_name, max_length, time.time()),
            )
        row = self.db.execute(
            "SELECT target_count, text, queue_name, max_length FROM producer_batches WHERE batch_id=?",
            (batch_id,),
        ).fetchone()
        if row != (count, text, queue_name, max_length):
            raise ValueError(
                f"Batch {batch_id!r} already exists with different parameters; "
                "resume with the original arguments or use another --batch-id"
            )

    def next_task(self, batch_id: str) -> tuple[PendingTask | None, bool]:
        row = self.db.execute(
            """SELECT t.sequence, t.task_id, t.created_at, b.text
               FROM producer_tasks t JOIN producer_batches b USING (batch_id)
               WHERE t.batch_id=? AND t.confirmed_at IS NULL
               ORDER BY t.sequence LIMIT 1""",
            (batch_id,),
        ).fetchone()
        if row:
            return PendingTask(TaskMessage(task_id=row[1], text=row[3]), row[2], row[0]), False

        batch = self.db.execute(
            "SELECT target_count, text FROM producer_batches WHERE batch_id=?", (batch_id,)
        ).fetchone()
        if batch is None:
            raise ValueError(f"Unknown batch: {batch_id}")
        last = self.db.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM producer_tasks WHERE batch_id=?", (batch_id,)
        ).fetchone()[0]
        if last >= batch[0]:
            return None, False

        task = TaskMessage(task_id=str(uuid4()), text=batch[1])
        created_at = time.time()
        with self.db:
            self.db.execute(
                """INSERT INTO producer_tasks (batch_id, sequence, task_id, created_at)
                   VALUES (?, ?, ?, ?)""",
                (batch_id, last + 1, task.task_id, created_at),
            )
        return PendingTask(task, created_at, last + 1), True

    def mark_confirmed(self, batch_id: str, task_id: str) -> None:
        with self.db:
            cursor = self.db.execute(
                """UPDATE producer_tasks SET confirmed_at=?
                   WHERE batch_id=? AND task_id=? AND confirmed_at IS NULL""",
                (time.time(), batch_id, task_id),
            )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Unknown or already confirmed task: {task_id}")

    def count_confirmed(self, batch_id: str) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM producer_tasks WHERE batch_id=? AND confirmed_at IS NOT NULL",
            (batch_id,),
        ).fetchone()[0]
