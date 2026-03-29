# 🚅🪵 Tiny Train Log 🚅🪵

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

### Multi-server merging

Ran experiments on multiple machines? Merge them into one database:

```python
MetricsLogger.merge(target_dir="./all_runs", source_dir="/mnt/gpu-box-1/runs")
MetricsLogger.merge(target_dir="./all_runs", source_dir="/mnt/gpu-box-2/runs")
```
