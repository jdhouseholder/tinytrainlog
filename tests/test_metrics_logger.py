import re
import sqlite3

import pytest

from tinytrainlog import MetricsLogger

_SCHEMA = {
    "config": {"lr": float, "batch_size": int, "model": str, "dropout": float, "layers": int},
    "train": {"loss": float, "lr": float, "val_loss": float, "val_acc": float},
    "eval": {"val_loss": float, "val_acc": float},
    "test": {"test_loss": float, "test_acc": float},
}


def _query(db_path, sql, params=()):
    con = sqlite3.connect(db_path)
    rows = con.execute(sql, params).fetchall()
    con.close()
    return rows


def test_auto_generated_run_name(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA)
    assert re.match(r"^[a-z]+-[a-z]+$", logger.run_name)
    logger.close()


def test_auto_name_avoids_collision(tmp_path):
    names = set()
    for _ in range(10):
        logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA)
        names.add(logger.run_name)
        logger.close()
    assert len(names) == 10


def test_explicit_run_name(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="my-run")
    assert logger.run_name == "my-run"
    assert logger.run_dir == tmp_path / "my-run"
    logger.close()


def test_run_registered_in_db(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    logger.close()
    rows = _query(tmp_path / "runs.db", "SELECT name FROM runs")
    assert rows == [("r",)]


def test_set_config(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    logger.set_config({"lr": 0.001, "batch_size": 32})
    logger.close()

    rows = _query(
        tmp_path / "runs.db",
        "SELECT lr, batch_size FROM config WHERE run_name = 'r'",
    )
    assert rows == [(0.001, 32)]


def test_set_config_overwrites(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    logger.set_config({"lr": 0.001})
    logger.set_config({"lr": 0.01})
    logger.close()

    rows = _query(
        tmp_path / "runs.db",
        "SELECT lr FROM config WHERE run_name = 'r'",
    )
    assert rows == [(0.01,)]


def test_set_config_preserves_types(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    logger.set_config({"model": "resnet50", "layers": 50, "dropout": 0.1})
    logger.close()

    rows = _query(
        tmp_path / "runs.db",
        "SELECT model, layers, dropout FROM config WHERE run_name = 'r'",
    )
    assert rows == [("resnet50", 50, 0.1)]


def test_set_config_partial_update(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    logger.set_config({"lr": 0.001})
    logger.set_config({"batch_size": 32})
    logger.close()

    rows = _query(
        tmp_path / "runs.db",
        "SELECT lr, batch_size FROM config WHERE run_name = 'r'",
    )
    assert rows == [(0.001, 32)]


def test_add_tags(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    logger.add_tags(["a", "b"])
    logger.add_tags(["b", "c"])
    logger.close()

    rows = _query(
        tmp_path / "runs.db", "SELECT tag FROM tags WHERE run_name = 'r' ORDER BY tag"
    )
    assert [r[0] for r in rows] == ["a", "b", "c"]


def test_log_step(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    logger.log_step(step=0, loss=0.5, lr=0.001)
    logger.log_step(step=1, loss=0.4, lr=0.001)
    logger.close()

    rows = _query(
        tmp_path / "runs.db",
        "SELECT step, loss, lr FROM train WHERE run_name = 'r' ORDER BY step",
    )
    assert rows == [(0, 0.5, 0.001), (1, 0.4, 0.001)]


def test_log_epoch(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    logger.log_epoch(epoch=0, val_loss=0.3, val_acc=0.92)
    logger.close()

    rows = _query(
        tmp_path / "runs.db",
        "SELECT epoch, val_loss, val_acc FROM train WHERE run_name = 'r'",
    )
    assert rows == [(0, 0.3, 0.92)]


def test_log_eval_by_epoch(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    logger.log_eval(epoch=0, val_loss=0.3, val_acc=0.92)
    logger.log_eval(epoch=1, val_loss=0.2, val_acc=0.95)
    logger.close()

    rows = _query(
        tmp_path / "runs.db",
        "SELECT step, epoch, val_loss, val_acc FROM eval"
        " WHERE run_name = 'r' ORDER BY epoch",
    )
    assert rows == [(None, 0, 0.3, 0.92), (None, 1, 0.2, 0.95)]


def test_log_eval_by_step(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    logger.log_eval(step=500, val_loss=0.3)
    logger.log_eval(step=1000, epoch=1, val_loss=0.2)
    logger.close()

    rows = _query(
        tmp_path / "runs.db",
        "SELECT step, epoch, val_loss FROM eval WHERE run_name = 'r' ORDER BY step",
    )
    assert rows == [(500, None, 0.3), (1000, 1, 0.2)]


def test_log_eval_requires_step_or_epoch(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    with pytest.raises(ValueError):
        logger.log_eval(val_loss=0.3)
    logger.close()


def test_log_test(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    logger.log_test(test_loss=0.25, test_acc=0.94)
    logger.close()

    rows = _query(
        tmp_path / "runs.db",
        "SELECT test_loss, test_acc FROM test WHERE run_name = 'r'",
    )
    assert rows == [(0.25, 0.94)]


def test_log_test_overwrites(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    logger.log_test(test_acc=0.90)
    logger.log_test(test_acc=0.94)
    logger.close()

    rows = _query(
        tmp_path / "runs.db",
        "SELECT test_acc FROM test WHERE run_name = 'r'",
    )
    assert rows == [(0.94,)]


def test_checkpoint_path_step(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    path = logger.checkpoint_path(step=100)
    assert path == tmp_path / "r" / "checkpoints" / "step_100.pt"
    logger.close()


def test_checkpoint_path_epoch(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    path = logger.checkpoint_path(epoch=2)
    assert path == tmp_path / "r" / "checkpoints" / "epoch_2.pt"
    logger.close()


def test_checkpoint_path_requires_exactly_one(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    with pytest.raises(ValueError):
        logger.checkpoint_path()
    with pytest.raises(ValueError):
        logger.checkpoint_path(step=1, epoch=1)
    logger.close()


def test_checkpoint_dir(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, schema=_SCHEMA, run_name="r")
    assert logger.checkpoint_dir == tmp_path / "r" / "checkpoints"
    assert logger.checkpoint_dir.exists()
    logger.close()


def test_context_manager(tmp_path):
    schema = {"train": {"loss": float}}
    with MetricsLogger(root_dir=tmp_path, schema=schema, run_name="r") as logger:
        logger.log_step(step=0, loss=1.0)
    rows = _query(
        tmp_path / "runs.db", "SELECT step, loss FROM train WHERE run_name = 'r'"
    )
    assert rows == [(0, 1.0)]


def test_multiple_runs_share_db(tmp_path):
    schema = {"train": {"loss": float}}
    l1 = MetricsLogger(root_dir=tmp_path, schema=schema, run_name="a")
    l1.log_step(step=0, loss=1.0)
    l1.close()

    l2 = MetricsLogger(root_dir=tmp_path, schema=schema, run_name="b")
    l2.log_step(step=0, loss=2.0)
    l2.close()

    rows = _query(tmp_path / "runs.db", "SELECT name FROM runs ORDER BY name")
    assert [r[0] for r in rows] == ["a", "b"]


def test_filter_runs_by_config(tmp_path):
    schema = {"config": {"lr": float}}
    for name, lr in [("a", 0.001), ("b", 0.01), ("c", 0.001)]:
        l = MetricsLogger(root_dir=tmp_path, schema=schema, run_name=name)
        l.set_config({"lr": lr})
        l.close()

    rows = _query(
        tmp_path / "runs.db",
        "SELECT run_name FROM config WHERE lr = 0.001 ORDER BY run_name",
    )
    assert [r[0] for r in rows] == ["a", "c"]


def test_merge(tmp_path):
    machine_a = tmp_path / "machine_a"
    machine_b = tmp_path / "machine_b"
    target = tmp_path / "merged"

    schema = {
        "config": {"lr": float},
        "train": {"loss": float},
    }

    with MetricsLogger(root_dir=machine_a, schema=schema, run_name="run-a") as l:
        l.set_config({"lr": 0.001})
        l.add_tags(["baseline"])
        l.log_step(step=0, loss=0.5)
        l.log_epoch(epoch=0, loss=0.3)
        l.checkpoint_path(step=0).write_text("fake-ckpt-a")

    with MetricsLogger(root_dir=machine_b, schema=schema, run_name="run-b") as l:
        l.set_config({"lr": 0.01})
        l.add_tags(["experiment"])
        l.log_step(step=0, loss=0.8)
        l.log_epoch(epoch=0, loss=0.6)
        l.checkpoint_path(step=0).write_text("fake-ckpt-b")

    MetricsLogger.merge(target, machine_a)
    MetricsLogger.merge(target, machine_b)

    rows = _query(target / "runs.db", "SELECT name FROM runs ORDER BY name")
    assert [r[0] for r in rows] == ["run-a", "run-b"]

    rows = _query(target / "runs.db", "SELECT run_name, lr FROM config ORDER BY run_name")
    assert len(rows) == 2

    rows = _query(target / "runs.db", "SELECT run_name, tag FROM tags ORDER BY run_name")
    assert len(rows) == 2

    rows = _query(
        target / "runs.db",
        "SELECT run_name, step, loss FROM train WHERE step IS NOT NULL ORDER BY run_name",
    )
    assert len(rows) == 2

    assert (target / "run-a" / "checkpoints" / "step_0.pt").read_text() == "fake-ckpt-a"
    assert (target / "run-b" / "checkpoints" / "step_0.pt").read_text() == "fake-ckpt-b"


def test_merge_conflict(tmp_path):
    machine_a = tmp_path / "machine_a"
    machine_b = tmp_path / "machine_b"

    schema = {"train": {"loss": float}}

    with MetricsLogger(root_dir=machine_a, schema=schema, run_name="same-name") as l:
        l.log_step(step=0, loss=1.0)

    with MetricsLogger(root_dir=machine_b, schema=schema, run_name="same-name") as l:
        l.log_step(step=0, loss=2.0)

    with pytest.raises(ValueError, match="same-name"):
        MetricsLogger.merge(machine_a, machine_b)


def test_merge_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        MetricsLogger.merge(tmp_path / "target", tmp_path / "nonexistent")


def test_delete_run(tmp_path):
    schema = {
        "config": {"lr": float},
        "train": {"loss": float},
        "eval": {"val_loss": float},
        "test": {"test_acc": float},
    }
    logger = MetricsLogger(root_dir=tmp_path, schema=schema, run_name="doomed")
    logger.set_config({"lr": 0.001})
    logger.add_tags(["test"])
    logger.log_step(step=0, loss=1.0)
    logger.log_epoch(epoch=0, loss=0.5)
    logger.log_eval(epoch=0, val_loss=0.4)
    logger.log_test(test_acc=0.9)
    logger.checkpoint_path(step=0).write_text("fake")

    logger.delete_run("doomed")
    logger.close()

    for table in ("runs", "config", "tags", "train", "eval", "test"):
        rows = _query(tmp_path / "runs.db", f"SELECT * FROM {table}")
        assert rows == [], f"expected {table} to be empty"

    assert not (tmp_path / "doomed").exists()


def test_delete_run_preserves_others(tmp_path):
    schema = {"train": {"loss": float}}
    l1 = MetricsLogger(root_dir=tmp_path, schema=schema, run_name="keep")
    l1.log_step(step=0, loss=1.0)
    l1.close()

    l2 = MetricsLogger(root_dir=tmp_path, schema=schema, run_name="remove")
    l2.log_step(step=0, loss=2.0)
    l2.delete_run("remove")
    l2.close()

    rows = _query(tmp_path / "runs.db", "SELECT name FROM runs")
    assert rows == [("keep",)]
    rows = _query(tmp_path / "runs.db", "SELECT run_name FROM train")
    assert rows == [("keep",)]


def test_timestamps_on_rows(tmp_path):
    schema = {
        "config": {"lr": float},
        "train": {"loss": float},
        "eval": {"val_loss": float},
        "test": {"test_acc": float},
    }
    logger = MetricsLogger(root_dir=tmp_path, schema=schema, run_name="r")
    logger.set_config({"lr": 0.001})
    logger.add_tags(["a"])
    logger.log_step(step=0, loss=1.0)
    logger.log_eval(epoch=0, val_loss=0.4)
    logger.log_test(test_acc=0.9)
    logger.close()

    db = tmp_path / "runs.db"

    rows = _query(db, "SELECT created_at FROM runs WHERE name = 'r'")
    assert rows[0][0] is not None

    rows = _query(db, "SELECT updated_at FROM config WHERE run_name = 'r'")
    assert rows[0][0] is not None

    rows = _query(db, "SELECT created_at FROM tags WHERE run_name = 'r'")
    assert rows[0][0] is not None

    rows = _query(db, "SELECT created_at FROM train WHERE run_name = 'r'")
    assert rows[0][0] is not None

    rows = _query(db, "SELECT created_at FROM eval WHERE run_name = 'r'")
    assert rows[0][0] is not None

    rows = _query(db, "SELECT updated_at FROM test WHERE run_name = 'r'")
    assert rows[0][0] is not None


def test_schema_validation_rejects_unknown_keys(tmp_path):
    schema = {"train": {"loss": float}}
    logger = MetricsLogger(root_dir=tmp_path, schema=schema, run_name="r")
    with pytest.raises(ValueError, match="Unknown keys"):
        logger.log_step(step=0, unknown_metric=1.0)
    logger.close()


def test_schema_migration_adds_columns(tmp_path):
    schema1 = {"train": {"loss": float}}
    l1 = MetricsLogger(root_dir=tmp_path, schema=schema1, run_name="r1")
    l1.log_step(step=0, loss=0.5)
    l1.close()

    schema2 = {"train": {"loss": float, "acc": float}}
    l2 = MetricsLogger(root_dir=tmp_path, schema=schema2, run_name="r2")
    l2.log_step(step=0, loss=0.4, acc=0.9)
    l2.close()

    rows = _query(
        tmp_path / "runs.db",
        "SELECT run_name, step, loss, acc FROM train ORDER BY run_name",
    )
    assert rows == [("r1", 0, 0.5, None), ("r2", 0, 0.4, 0.9)]


def test_merge_different_schemas(tmp_path):
    machine_a = tmp_path / "machine_a"
    machine_b = tmp_path / "machine_b"
    target = tmp_path / "merged"

    with MetricsLogger(
        root_dir=machine_a,
        schema={"train": {"loss": float}},
        run_name="run-a",
    ) as l:
        l.log_step(step=0, loss=0.5)

    with MetricsLogger(
        root_dir=machine_b,
        schema={"train": {"loss": float, "acc": float}},
        run_name="run-b",
    ) as l:
        l.log_step(step=0, loss=0.4, acc=0.9)

    MetricsLogger.merge(target, machine_a)
    MetricsLogger.merge(target, machine_b)

    rows = _query(
        target / "runs.db",
        "SELECT run_name, loss, acc FROM train ORDER BY run_name",
    )
    assert rows == [("run-a", 0.5, None), ("run-b", 0.4, 0.9)]
