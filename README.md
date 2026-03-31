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

with MetricsLogger("./runs", run_name="lr-sweep-3e4") as log:
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
    SELECT c.run_name, c.value AS lr, e.value AS val_acc
    FROM config c
    JOIN eval e USING (run_name)
    WHERE c.key = 'lr' AND e.key = 'val_acc'
    ORDER BY CAST(e.value AS REAL) DESC
""").fetchall()
```

## API

**Initialization:**

```python
# Auto-generated run name (e.g. "bold-falcon")
log = MetricsLogger("./runs")

# Explicit name
log = MetricsLogger("./runs", run_name="lr-sweep-3e4")

# Override machine ID (defaults to hostname)
log = MetricsLogger("./runs", machine_id="gpu-box-1")
```

**Logging:**

| Method | Purpose |
|--------|---------|
| `set_config(dict)` | Hyperparameters and run metadata (JSON, upserts per key) |
| `add_tags(list)` | Labels for filtering (e.g. `["ablation", "v2"]`) |
| `log_step(step, **metrics)` | Per-batch metrics (loss, lr, throughput) |
| `log_epoch(epoch, **metrics)` | Per-epoch metrics |
| `log_eval(step=, epoch=, **metrics)` | Validation metrics (requires at least one of step/epoch) |
| `log_test(**metrics)` | Final test results (upserts) |

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

SELECT r.name,
       MAX(CASE WHEN c.key = 'lr' THEN c.value END) AS lr,
       MAX(CASE WHEN c.key = 'model' THEN c.value END) AS model,
       t.value AS test_acc
FROM runs r
JOIN config c ON c.run_name = r.name
LEFT JOIN test t ON t.run_name = r.name AND t.key = 'test_acc'
GROUP BY r.name
ORDER BY t.value DESC;
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
SELECT run_name, value FROM test
WHERE key = 'test_acc' ORDER BY value DESC LIMIT 1
```

**Compare hyperparameters across runs:**
```sql
SELECT r.name,
       MAX(CASE WHEN c.key = 'lr' THEN c.value END) AS lr,
       MAX(CASE WHEN c.key = 'model' THEN c.value END) AS model,
       t.value AS test_acc
FROM runs r
JOIN config c ON c.run_name = r.name
LEFT JOIN test t ON t.run_name = r.name AND t.key = 'test_acc'
GROUP BY r.name
ORDER BY t.value DESC
```

**Training curve for a run (for plotting):**
```sql
SELECT step, key, value FROM steps
WHERE run_name = 'lr-sweep-3e4' ORDER BY step
```

**Filter runs by tag:**
```sql
SELECT run_name FROM tags WHERE tag = 'ablation'
```

**Side-by-side eval comparison:**
```sql
SELECT a.epoch, a.value AS model_a, b.value AS model_b
FROM eval a JOIN eval b USING (epoch, key)
WHERE a.run_name = 'model-a' AND b.run_name = 'model-b' AND a.key = 'val_acc'
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

**Pareto frontier (accuracy vs. parameter count):**
```sql
SELECT r.name, c.value AS param_count, t.value AS test_acc
FROM runs r
JOIN config c ON c.run_name = r.name AND c.key = 'param_count'
JOIN test t ON t.run_name = r.name AND t.key = 'test_acc'
WHERE NOT EXISTS (
    SELECT 1 FROM config c2 JOIN test t2 ON t2.run_name = c2.run_name
    WHERE c2.key = 'param_count' AND t2.key = 'test_acc'
      AND CAST(c2.value AS REAL) <= CAST(c.value AS REAL)
      AND t2.value >= t.value
      AND (CAST(c2.value AS REAL) < CAST(c.value AS REAL) OR t2.value > t.value)
)
ORDER BY CAST(c.value AS REAL)
```
