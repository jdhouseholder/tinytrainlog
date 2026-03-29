import json
import re

import pytest

from tinytrainlog import MetricsLogger


def test_auto_generated_run_name(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path)
    assert re.match(r"^[a-z]+-[a-z]+$", logger.run_name)
    assert logger.run_dir.exists()


def test_auto_name_avoids_collision(tmp_path):
    names = set()
    for _ in range(10):
        logger = MetricsLogger(root_dir=tmp_path)
        names.add(logger.run_name)
    assert len(names) == 10


def test_explicit_run_name(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, run_name="my-run")
    assert logger.run_name == "my-run"
    assert logger.run_dir == tmp_path / "my-run"
    assert logger.run_dir.exists()


def test_set_config(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, run_name="r")
    config = {"lr": 0.001, "batch_size": 32}
    logger.set_config(config)

    saved = json.loads((logger.run_dir / "config.json").read_text())
    assert saved == config


def test_set_config_overwrites(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, run_name="r")
    logger.set_config({"lr": 0.001})
    logger.set_config({"lr": 0.01})

    saved = json.loads((logger.run_dir / "config.json").read_text())
    assert saved == {"lr": 0.01}


def test_add_tags(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, run_name="r")
    logger.add_tags(["a", "b"])
    logger.add_tags(["b", "c"])

    saved = json.loads((logger.run_dir / "tags.json").read_text())
    assert saved == ["a", "b", "c"]


def test_log_step(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, run_name="r")
    logger.log_step(step=0, loss=0.5, lr=0.001)
    logger.log_step(step=1, loss=0.4, lr=0.001)

    lines = (logger.run_dir / "steps.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"step": 0, "loss": 0.5, "lr": 0.001}
    assert json.loads(lines[1]) == {"step": 1, "loss": 0.4, "lr": 0.001}


def test_log_epoch(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, run_name="r")
    logger.log_epoch(epoch=0, val_loss=0.3, val_acc=0.92)

    lines = (logger.run_dir / "epochs.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"epoch": 0, "val_loss": 0.3, "val_acc": 0.92}


def test_checkpoint_path_step(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, run_name="r")
    path = logger.checkpoint_path(step=100)
    assert path == logger.run_dir / "checkpoints" / "step_100.pt"


def test_checkpoint_path_epoch(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, run_name="r")
    path = logger.checkpoint_path(epoch=2)
    assert path == logger.run_dir / "checkpoints" / "epoch_2.pt"


def test_checkpoint_path_requires_exactly_one(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, run_name="r")
    with pytest.raises(ValueError):
        logger.checkpoint_path()
    with pytest.raises(ValueError):
        logger.checkpoint_path(step=1, epoch=1)


def test_checkpoint_dir(tmp_path):
    logger = MetricsLogger(root_dir=tmp_path, run_name="r")
    assert logger.checkpoint_dir == logger.run_dir / "checkpoints"
    assert logger.checkpoint_dir.exists()
