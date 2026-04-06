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
CREATE TABLE IF NOT EXISTS tags (
    run_name   TEXT NOT NULL REFERENCES runs(name),
    tag        TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(run_name, tag)
);
CREATE TABLE IF NOT EXISTS config (
    run_name   TEXT PRIMARY KEY REFERENCES runs(name),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS train (
    run_name   TEXT    NOT NULL REFERENCES runs(name),
    step       INTEGER,
    epoch      INTEGER,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS eval (
    run_name   TEXT    NOT NULL REFERENCES runs(name),
    step       INTEGER,
    epoch      INTEGER,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS test (
    run_name   TEXT PRIMARY KEY REFERENCES runs(name),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_TYPE_MAP = {float: "REAL", int: "INTEGER", str: "TEXT"}

_DATA_TABLES = ("config", "tags", "train", "eval", "test")


class MetricsLogger:
    def __init__(
        self,
        root_dir: str | Path,
        schema: dict[str, dict[str, type]],
        run_name: str | None = None,
        machine_id: str | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.machine_id = machine_id or socket.gethostname()
        self._schema = schema

        self._db_path = self.root_dir / "runs.db"
        self._conn = sqlite3.connect(self._db_path)
        self._conn.executescript(_SCHEMA)

        for table, columns in schema.items():
            existing = {
                row[1]
                for row in self._conn.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            }
            for col_name, col_type in columns.items():
                if col_name not in existing:
                    sql_type = _TYPE_MAP[col_type]
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col_name} {sql_type}"
                    )
        self._conn.commit()

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

    def _validate_keys(self, stage: str, keys: set[str]) -> None:
        allowed = set(self._schema.get(stage, {}))
        unknown = keys - allowed
        if unknown:
            raise ValueError(
                f"Unknown keys for '{stage}': {unknown}. Allowed: {allowed}"
            )

    def set_config(self, config: dict) -> None:
        if not config:
            return
        self._validate_keys("config", set(config.keys()))
        cols = list(config.keys())
        col_names = ", ".join(["run_name"] + cols)
        placeholders = ", ".join(["?"] * (len(cols) + 1))
        update_clause = ", ".join(f"{c} = excluded.{c}" for c in cols)
        sql = (
            f"INSERT INTO config ({col_names}) VALUES ({placeholders})"
            f" ON CONFLICT(run_name) DO UPDATE SET {update_clause},"
            " updated_at = datetime('now')"
        )
        self._conn.execute(sql, [self.run_name] + [config[c] for c in cols])
        self._conn.commit()

    def add_tags(self, tags: list[str]) -> None:
        self._conn.executemany(
            "INSERT OR IGNORE INTO tags (run_name, tag) VALUES (?, ?)",
            [(self.run_name, tag) for tag in tags],
        )
        self._conn.commit()

    def _log_train(self, step: int | None, epoch: int | None, metrics: dict) -> None:
        if not metrics:
            return
        self._validate_keys("train", set(metrics.keys()))
        cols = list(metrics.keys())
        col_names = ", ".join(["run_name", "step", "epoch"] + cols)
        placeholders = ", ".join(["?"] * (len(cols) + 3))
        sql = f"INSERT INTO train ({col_names}) VALUES ({placeholders})"
        self._conn.execute(
            sql, [self.run_name, step, epoch] + [metrics[c] for c in cols]
        )
        self._conn.commit()

    def log_step(self, step: int, **metrics) -> None:
        self._log_train(step=step, epoch=None, metrics=metrics)

    def log_epoch(self, epoch: int, **metrics) -> None:
        self._log_train(step=None, epoch=epoch, metrics=metrics)

    def log_eval(
        self, *, step: int | None = None, epoch: int | None = None, **metrics
    ) -> None:
        if step is None and epoch is None:
            raise ValueError("At least one of 'step' or 'epoch' must be provided.")
        if not metrics:
            return
        self._validate_keys("eval", set(metrics.keys()))
        cols = list(metrics.keys())
        col_names = ", ".join(["run_name", "step", "epoch"] + cols)
        placeholders = ", ".join(["?"] * (len(cols) + 3))
        sql = f"INSERT INTO eval ({col_names}) VALUES ({placeholders})"
        self._conn.execute(
            sql, [self.run_name, step, epoch] + [metrics[c] for c in cols]
        )
        self._conn.commit()

    def log_test(self, **metrics) -> None:
        if not metrics:
            return
        self._validate_keys("test", set(metrics.keys()))
        cols = list(metrics.keys())
        col_names = ", ".join(["run_name"] + cols)
        placeholders = ", ".join(["?"] * (len(cols) + 1))
        update_clause = ", ".join(f"{c} = excluded.{c}" for c in cols)
        sql = (
            f"INSERT INTO test ({col_names}) VALUES ({placeholders})"
            f" ON CONFLICT(run_name) DO UPDATE SET {update_clause},"
            " updated_at = datetime('now')"
        )
        self._conn.execute(sql, [self.run_name] + [metrics[c] for c in cols])
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
            # Add any source columns missing from target
            for table in _DATA_TABLES:
                if table == "tags":
                    continue
                source_cols = {
                    row[1]: row[2]
                    for row in conn.execute(
                        f"PRAGMA other.table_info({table})"
                    ).fetchall()
                }
                target_cols = {
                    row[1]
                    for row in conn.execute(
                        f"PRAGMA main.table_info({table})"
                    ).fetchall()
                }
                for col_name, col_type in source_cols.items():
                    if col_name not in target_cols:
                        conn.execute(
                            f"ALTER TABLE main.{table} ADD COLUMN"
                            f" {col_name} {col_type}"
                        )

            conn.execute("BEGIN")
            try:
                conn.execute("INSERT INTO main.runs SELECT * FROM other.runs")
                for table in _DATA_TABLES:
                    source_col_names = [
                        row[1]
                        for row in conn.execute(
                            f"PRAGMA other.table_info({table})"
                        ).fetchall()
                    ]
                    cols = ", ".join(source_col_names)
                    conn.execute(
                        f"INSERT INTO main.{table} ({cols})"
                        f" SELECT {cols} FROM other.{table}"
                    )
                source_runs = [
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM other.runs"
                    ).fetchall()
                ]
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.execute("DETACH DATABASE other")
            conn.close()

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
