import json
from pathlib import Path

from ._names import generate_run_name


class MetricsLogger:
    def __init__(self, root_dir: str | Path, run_name: str | None = None):
        self.root_dir = Path(root_dir)
        if run_name is None:
            run_name = generate_run_name(self.root_dir)
        self.run_name = run_name
        self.run_dir = self.root_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = self.run_dir / "checkpoints"
        self._checkpoint_dir.mkdir(exist_ok=True)

    def set_config(self, config: dict) -> None:
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def add_tags(self, tags: list[str]) -> None:
        tags_path = self.run_dir / "tags.json"
        if tags_path.exists():
            existing = json.loads(tags_path.read_text())
        else:
            existing = []
        seen = set(existing)
        for tag in tags:
            if tag not in seen:
                existing.append(tag)
                seen.add(tag)
        tags_path.write_text(json.dumps(existing, indent=2) + "\n")

    def log_step(self, step: int, **metrics) -> None:
        with open(self.run_dir / "steps.jsonl", "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")

    def log_epoch(self, epoch: int, **metrics) -> None:
        with open(self.run_dir / "epochs.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}) + "\n")

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
