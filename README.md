# 🚅🪵 Tiny Train Log 🚅🪵

[![PyPI](https://img.shields.io/pypi/v/tinytrainlog)](https://pypi.org/project/tinytrainlog/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Structured, queryable logging for ML experiments. Track metrics across machines,
merge results into one SQLite database, and analyze everything with plain SQL.

## Quick start

```bash
pip install tinytrainlog
```

```python
from tinytrainlog import MetricsLogger

schema = {
    "config": {"model": str, "lr": float, "epochs": int},
    "train":  {"train_loss": float},
    "eval":   {"val_loss": float, "val_acc": float},
    "test":   {"test_acc": float, "test_loss": float},
}

with MetricsLogger("./runs", schema=schema, run_name="lr-sweep-3e4") as log:
    log.set_config({"model": "resnet50", "lr": 3e-4, "epochs": 10})
    log.add_tags(["sweep", "baseline"])

    for epoch in range(10):
        train_loss = train(model, loader)
        log.log_epoch(epoch=epoch, train_loss=train_loss)

        val_loss, val_acc = evaluate(model, val_loader)
        log.log_eval(epoch=epoch, val_loss=val_loss, val_acc=val_acc)

        torch.save(model.state_dict(), log.checkpoint_path(epoch=epoch))

    log.log_test(test_acc=0.94, test_loss=0.21)
```

Everything lands in a single `runs.db` SQLite file — query it however you like:

```python
import sqlite3

conn = sqlite3.connect("./runs/runs.db")

# Compare learning rates across runs
conn.execute("""
    SELECT c.run_name, c.lr, e.val_acc
    FROM config c
    JOIN eval e USING (run_name)
    ORDER BY e.val_acc DESC
""").fetchall()
```

## API

**Initialization:**

```python
schema = {
    "config": {"model": str, "lr": float, "epochs": int},
    "train":  {"train_loss": float},
    "eval":   {"val_loss": float, "val_acc": float},
    "test":   {"test_acc": float, "test_loss": float},
}

# Auto-generated run name (e.g. "bold-falcon")
log = MetricsLogger("./runs", schema=schema)

# Explicit name
log = MetricsLogger("./runs", schema=schema, run_name="lr-sweep-3e4")

# Override machine ID (defaults to hostname)
log = MetricsLogger("./runs", schema=schema, machine_id="gpu-box-1")
```

The `schema` dict defines typed columns for each stage. Supported types:
`float` (REAL), `int` (INTEGER), `str` (TEXT). New columns are added
automatically via `ALTER TABLE` on init — schemas can evolve over time.

**Logging:**

| Method | Purpose |
|--------|---------|
| `set_config(dict)` | Hyperparameters and run metadata (one row per run, upserts) |
| `add_tags(list)` | Labels for filtering (e.g. `["ablation", "v2"]`) |
| `log_step(step, **metrics)` | Per-batch metrics to `train` table |
| `log_epoch(epoch, **metrics)` | Per-epoch metrics to `train` table |
| `log_eval(step=, epoch=, **metrics)` | Validation metrics (requires at least one of step/epoch) |
| `log_test(**metrics)` | Final test results (one row per run, upserts) |

Metric keys are validated against the schema — unknown keys raise `ValueError`.

**Checkpoints and paths:**

```python
log.run_name                  # "bold-falcon"
log.run_dir                   # Path("./runs/bold-falcon")
log.checkpoint_dir            # Path("./runs/bold-falcon/checkpoints")
log.checkpoint_path(epoch=5)  # Path("./runs/bold-falcon/checkpoints/epoch_5.pt")
log.checkpoint_path(step=100) # Path("./runs/bold-falcon/checkpoints/step_100.pt")
```

Use `run_dir` to save extra artifacts (plots, predictions, etc.) alongside the run.

### Multi-server merging

Ran experiments on multiple machines? Merge them into one database:

```python
MetricsLogger.merge(target_dir="./all_runs", source_dir="/mnt/gpu-box-1/runs")
MetricsLogger.merge(target_dir="./all_runs", source_dir="/mnt/gpu-box-2/runs")
```

Merge auto-discovers columns from the source and adds any missing ones to
the target via `ALTER TABLE`, so databases with different schemas merge cleanly.

### Deleting a run

Remove a run and all its data (config, metrics, checkpoints):

```python
logger.delete_run("old-run")
```

## Recipes

Save queries in a `.sql` file and run them directly from the terminal:

```bash
sqlite3 ./runs/runs.db < analysis.sql
```

```sql
-- analysis.sql
.headers on
.mode column

SELECT r.name, c.lr, c.model, t.test_acc
FROM runs r
JOIN config c ON c.run_name = r.name
LEFT JOIN test t ON t.run_name = r.name
ORDER BY t.test_acc DESC;
```

All data lives in a single SQLite file:

```python
import sqlite3
conn = sqlite3.connect("./runs/runs.db")
```

**List all runs with tags:**
```sql
SELECT r.name, r.machine_id, r.created_at, GROUP_CONCAT(t.tag)
FROM runs r LEFT JOIN tags t ON t.run_name = r.name
GROUP BY r.name
```

**Best run by test accuracy:**
```sql
SELECT run_name, test_acc FROM test
ORDER BY test_acc DESC LIMIT 1
```

**Compare hyperparameters across runs:**
```sql
SELECT c.run_name, c.lr, c.epochs, MIN(e.val_loss) AS best_val_loss
FROM config c
JOIN eval e USING (run_name)
GROUP BY c.run_name
ORDER BY best_val_loss
```

**Training curve for a run (for plotting):**
```sql
SELECT epoch, train_loss FROM train
WHERE run_name = 'lr-sweep-3e4' ORDER BY epoch
```

**Filter runs by tag:**
```sql
SELECT run_name FROM tags WHERE tag = 'ablation'
```

**Side-by-side eval comparison:**
```sql
SELECT a.epoch, a.val_acc AS model_a, b.val_acc AS model_b
FROM eval a JOIN eval b USING (epoch)
WHERE a.run_name = 'model-a' AND b.run_name = 'model-b'
ORDER BY a.epoch
```

**Latest runs:**
```sql
SELECT name, created_at FROM runs ORDER BY created_at DESC LIMIT 10
```

**Runs from a specific machine:**
```sql
SELECT name FROM runs WHERE machine_id = 'gpu-box-1'
```

**Pareto frontier (accuracy vs. training budget):**
```sql
SELECT r.name, c.epochs, t.test_acc
FROM runs r
JOIN config c ON c.run_name = r.name
JOIN test t ON t.run_name = r.name
WHERE NOT EXISTS (
    SELECT 1 FROM config c2 JOIN test t2 ON t2.run_name = c2.run_name
    WHERE c2.epochs <= c.epochs
      AND t2.test_acc >= t.test_acc
      AND (c2.epochs < c.epochs OR t2.test_acc > t.test_acc)
)
ORDER BY c.epochs
```
