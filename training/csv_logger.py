import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from lightning.pytorch.loggers import CSVLogger


class TimestampedCSVLogger(CSVLogger):
    """CSVLogger whose run folder is named after the start time instead of version_<n>.

    Each run writes to <save_dir>/<name>/<YYYY-MM-DD_HH-MM-SS>/, so repeated runs of the
    same regime never overwrite each other and are easy to tell apart. When a run is resumed
    into its existing folder (by passing its folder name as ``version``), the metrics of the
    earlier sessions are kept and new ones are appended.
    """

    def __init__(
        self,
        save_dir: str,
        name: str = "run",
        version: Optional[str] = None,
        **kwargs,
    ):
        if version is None:
            version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        super().__init__(save_dir, name=name, version=version, **kwargs)

    @property
    def experiment(self):
        if self._experiment is not None:
            return self._experiment

        # Lightning's writer deletes an existing metrics.csv when it is created
        metrics_file = Path(self.log_dir, "metrics.csv")
        previous = metrics_file.read_text() if metrics_file.exists() else None

        experiment = CSVLogger.experiment.fget(self)
        if previous:
            metrics_file.write_text(previous)
            with open(metrics_file, newline="") as f:
                experiment.metrics_keys = sorted(csv.DictReader(f).fieldnames or [])
        return experiment
