import json
import shutil
import socket
import sqlite3
from pathlib import Path

from ._names import generate_run_name

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    name       TEXT PRIMARY KEY,
    machine_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS config (
    run_name   TEXT NOT NULL REFERENCES runs(name),
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(run_name, key)
);
CREATE TABLE IF NOT EXISTS tags (
    run_name   TEXT NOT NULL REFERENCES runs(name),
    tag        TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(run_name, tag)
);
CREATE TABLE IF NOT EXISTS steps (
    run_name   TEXT    NOT NULL REFERENCES runs(name),
    step       INTEGER NOT NULL,
    key        TEXT    NOT NULL,
    value      REAL    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS epochs (
    run_name   TEXT    NOT NULL REFERENCES runs(name),
    epoch      INTEGER NOT NULL,
    key        TEXT    NOT NULL,
    value      REAL    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS eval (
    run_name   TEXT    NOT NULL REFERENCES runs(name),
    step       INTEGER,
    epoch      INTEGER,
    key        TEXT    NOT NULL,
    value      REAL    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS test (
    run_name   TEXT NOT NULL REFERENCES runs(name),
    key        TEXT NOT NULL,
    value      REAL NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(run_name, key)
);
"""

_DATA_TABLES = ("config", "tags", "steps", "epochs", "eval", "test")


class MetricsLogger:
    def __init__(
        self,
        root_dir: str | Path,
        run_name: str | None = None,
        machine_id: str | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.machine_id = machine_id or socket.gethostname()

        self._db_path = self.root_dir / "runs.db"
        self._conn = sqlite3.connect(self._db_path)
        self._conn.executescript(_SCHEMA)

        if run_name is None:
            run_name = generate_run_name(self._conn)
        self.run_name = run_name

        self._conn.execute(
            "INSERT INTO runs (name, machine_id) VALUES (?, ?)"
            " ON CONFLICT(name) DO UPDATE SET machine_id = excluded.machine_id",
            (self.run_name, self.machine_id),
        )
        self._conn.commit()

        self._checkpoint_dir = self.root_dir / self.run_name / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        return self.root_dir / self.run_name

    def set_config(self, config: dict) -> None:
        self._conn.executemany(
            "INSERT INTO config (run_name, key, value) VALUES (?, ?, ?)"
            " ON CONFLICT(run_name, key) DO UPDATE SET value = excluded.value,"
            " updated_at = datetime('now')",
            [(self.run_name, k, json.dumps(v)) for k, v in config.items()],
        )
        self._conn.commit()

    def add_tags(self, tags: list[str]) -> None:
        self._conn.executemany(
            "INSERT OR IGNORE INTO tags (run_name, tag) VALUES (?, ?)",
            [(self.run_name, tag) for tag in tags],
        )
        self._conn.commit()

    def log_step(self, step: int, **metrics) -> None:
        self._conn.executemany(
            "INSERT INTO steps (run_name, step, key, value) VALUES (?, ?, ?, ?)",
            [(self.run_name, step, k, v) for k, v in metrics.items()],
        )
        self._conn.commit()

    def log_epoch(self, epoch: int, **metrics) -> None:
        self._conn.executemany(
            "INSERT INTO epochs (run_name, epoch, key, value) VALUES (?, ?, ?, ?)",
            [(self.run_name, epoch, k, v) for k, v in metrics.items()],
        )
        self._conn.commit()

    def log_eval(
        self, *, step: int | None = None, epoch: int | None = None, **metrics
    ) -> None:
        if step is None and epoch is None:
            raise ValueError("At least one of 'step' or 'epoch' must be provided.")
        self._conn.executemany(
            "INSERT INTO eval (run_name, step, epoch, key, value) VALUES (?, ?, ?, ?, ?)",
            [(self.run_name, step, epoch, k, v) for k, v in metrics.items()],
        )
        self._conn.commit()

    def log_test(self, **metrics) -> None:
        self._conn.executemany(
            "INSERT INTO test (run_name, key, value) VALUES (?, ?, ?)"
            " ON CONFLICT(run_name, key) DO UPDATE SET value = excluded.value,"
            " updated_at = datetime('now')",
            [(self.run_name, k, v) for k, v in metrics.items()],
        )
        self._conn.commit()

    def checkpoint_path(
        self, step: int | None = None, epoch: int | None = None
    ) -> Path:
        if (step is None) == (epoch is None):
            raise ValueError("Exactly one of 'step' or 'epoch' must be provided.")
        if step is not None:
            return self._checkpoint_dir / f"step_{step}.pt"
        return self._checkpoint_dir / f"epoch_{epoch}.pt"

    @property
    def checkpoint_dir(self) -> Path:
        return self._checkpoint_dir

    def delete_run(self, run_name: str) -> None:
        self._conn.execute("BEGIN")
        try:
            for table in _DATA_TABLES:
                self._conn.execute(
                    f"DELETE FROM {table} WHERE run_name = ?", (run_name,)
                )
            self._conn.execute("DELETE FROM runs WHERE name = ?", (run_name,))
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

        run_dir = self.root_dir / run_name
        if run_dir.exists():
            shutil.rmtree(run_dir)

    @staticmethod
    def merge(target_dir: str | Path, source_dir: str | Path) -> None:
        target_dir = Path(target_dir)
        source_dir = Path(source_dir)
        target_db = target_dir / "runs.db"
        source_db = source_dir / "runs.db"

        if not source_db.exists():
            raise FileNotFoundError(f"No runs.db found in {source_dir}")

        target_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(target_db)
        conn.executescript(_SCHEMA)
        conn.execute("ATTACH DATABASE ? AS other", (str(source_db),))

        # Check for name conflicts
        conflicts = conn.execute(
            "SELECT o.name, o.machine_id, m.machine_id"
            " FROM other.runs o INNER JOIN main.runs m ON o.name = m.name"
        ).fetchall()
        if conflicts:
            conn.execute("DETACH DATABASE other")
            conn.close()
            details = ", ".join(
                f"'{name}' (source: {src or '?'}, target: {tgt or '?'})"
                for name, src, tgt in conflicts
            )
            raise ValueError(
                f"Run name conflicts: {details}. "
                f"Rename the conflicting runs before merging."
            )

        try:
            conn.execute("BEGIN")
            conn.execute("INSERT INTO main.runs SELECT * FROM other.runs")
            for table in _DATA_TABLES:
                conn.execute(f"INSERT INTO main.{table} SELECT * FROM other.{table}")
            source_runs = [
                row[0] for row in conn.execute("SELECT name FROM other.runs").fetchall()
            ]
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.execute("DETACH DATABASE other")
            conn.close()

        # Copy run directories (checkpoints, etc.)
        for run_name in source_runs:
            run_dir = source_dir / run_name
            if run_dir.is_dir():
                shutil.copytree(run_dir, target_dir / run_name)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
