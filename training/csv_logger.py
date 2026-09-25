from datetime import datetime
from typing import Optional

from lightning.pytorch.loggers import CSVLogger


class TimestampedCSVLogger(CSVLogger):
    """CSVLogger whose run folder is named after the start time instead of version_<n>.

    Each run writes to <save_dir>/<name>/<YYYY-MM-DD_HH-MM-SS>/, so repeated runs of the
    same regime never overwrite each other and are easy to tell apart.
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
