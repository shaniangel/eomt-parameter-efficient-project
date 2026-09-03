# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from PyTorch Lightning,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------


import jsonargparse._typehints as _t
from types import MethodType
from gitignore_parser import parse_gitignore
import logging
import torch
import warnings
from lightning.pytorch import cli
from lightning.pytorch.callbacks import ModelSummary, LearningRateMonitor
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher

from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule

# Suppress PyTorch FX warnings for DINOv3 models
import os
os.environ["TORCH_LOGS"] = "-dynamo"


_orig_single = _t.raise_unexpected_value


def _raise_single(*args, exception=None, **kwargs):
    if isinstance(exception, Exception):
        raise exception
    return _orig_single(*args, exception=exception, **kwargs)


_orig_union = _t.raise_union_unexpected_value


def _raise_union(subtypes, val, vals):
    for e in reversed(vals):
        if isinstance(e, Exception):
            raise e
    return _orig_union(subtypes, val, vals)


_t.raise_unexpected_value = _raise_single
_t.raise_union_unexpected_value = _raise_union


def _should_check_val_fx(self: _TrainingEpochLoop, data_fetcher: _DataFetcher) -> bool:
    if not self._should_check_val_epoch():
        return False

    is_infinite_dataset = self.trainer.val_check_batch == float("inf")
    is_last_batch = self.batch_progress.is_last_batch
    if is_last_batch and (
        is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)
    ):
        return True

    if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
        return True

    is_val_check_batch = is_last_batch
    if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
        is_val_check_batch = (
            self.batch_idx + 1
        ) % self.trainer.limit_train_batches == 0
    elif self.trainer.val_check_batch != float("inf"):
        if self.trainer.check_val_every_n_epoch is not None:
            is_val_check_batch = (
                self.batch_idx + 1
            ) % self.trainer.val_check_batch == 0
        else:
            # added below to check val based on global steps instead of batches in case of iteration based val check and gradient accumulation
            is_val_check_batch = (
                self.global_step
            ) % self.trainer.val_check_batch == 0 and not self._should_accumulate()

    return is_val_check_batch


class LightningCLI(cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        warnings.filterwarnings(
            "ignore",
            message=r".*It is recommended to use .* when logging on epoch level in distributed setting to accumulate the metric across devices.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"^The ``compute`` method of metric PanopticQuality was called before the ``update`` method.*",
        )
        warnings.filterwarnings(
            "ignore", message=r"^Grad strides do not match bucket view strides.*"
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*functools.partial will be a method descriptor in future Python versions*",
        )

        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--compile_disabled", action="store_true")

        parser.link_arguments(
            "data.init_args.num_classes", "model.init_args.num_classes"
        )
        parser.link_arguments(
            "data.init_args.num_classes",
            "model.init_args.network.init_args.num_classes",
        )

        parser.link_arguments(
            "data.init_args.stuff_classes", "model.init_args.stuff_classes"
        )

        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments(
            "data.init_args.img_size", "model.init_args.network.init_args.img_size"
        )
        parser.link_arguments(
            "data.init_args.img_size",
            "model.init_args.network.init_args.encoder.init_args.img_size",
        )

        parser.link_arguments(
            "model.init_args.ckpt_path",
            "model.init_args.network.init_args.encoder.init_args.ckpt_path",
        )

    def fit(self, model, **kwargs):
        # Create a unique run directory under the logger save_dir so each run has its own folder.
        from pathlib import Path
        import datetime
        import json

        try:
            logger_save_dir = Path(getattr(self.trainer.logger, "save_dir", "pilot_logs"))
            logger_name = getattr(self.trainer.logger, "name", "run")
        except Exception:
            logger_save_dir = Path("pilot_logs")
            logger_name = "run"

        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = logger_save_dir / f"{logger_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Try to steer the logger to write under this run dir where possible
        try:
            if hasattr(self.trainer.logger, "save_dir"):
                self.trainer.logger.save_dir = str(run_dir)
        except Exception:
            pass

        # Ensure a CSVLogger is present so per-epoch metrics.csv files are written
        # (this enables plotting per-epoch curves). We prefer adding a CSV logger
        # rather than replacing the configured logger.
        try:
            from lightning.pytorch.loggers import CSVLogger

            # build list of existing loggers
            try:
                existing = list(self.trainer.loggers) if isinstance(self.trainer.loggers, (list, tuple)) else [self.trainer.logger]
            except Exception:
                existing = [self.trainer.logger] if getattr(self.trainer, "logger", None) else []

            has_csv = any(getattr(l, "__class__", type(None)).__name__ == "CSVLogger" for l in existing if l)
            if not has_csv:
                # create CSVLogger that writes under the same run_dir
                csv_logger = CSVLogger(save_dir=str(run_dir.parent), name=run_dir.name)
                try:
                    # attach to trainer.loggers (works whether it was a list or single)
                    if isinstance(self.trainer.loggers, list):
                        self.trainer.loggers.append(csv_logger)
                    else:
                        self.trainer.loggers = existing + [csv_logger]
                except Exception:
                    try:
                        self.trainer.loggers = existing + [csv_logger]
                    except Exception:
                        pass
        except Exception:
            # don't fail training if logger plumbing fails
            pass

        # Save the intended run_dir for later use
        self._run_dir = run_dir

        # Optional: log code (if logger/expt supports it)
        if hasattr(self.trainer.logger, "experiment") and hasattr(
            self.trainer.logger.experiment, "log_code"
        ):
            is_gitignored = parse_gitignore(".gitignore")
            include_fn = lambda path: path.endswith(".py") or path.endswith(".yaml")
            try:
                self.trainer.logger.experiment.log_code(
                    ".", include_fn=include_fn, exclude_fn=is_gitignored
                )
            except Exception:
                # don't fail training for logging errors
                pass

        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )

        if not self.config[self.config["subcommand"]]["compile_disabled"]:
            model = torch.compile(model)

        # Run training
        self.trainer.fit(model, **kwargs)

        # After training, run validation and persist metrics to the run directory
        try:
            dm = getattr(self, "datamodule", None)
            # Run a validation pass (no grad)
            val_results = self.trainer.validate(model, datamodule=dm)

            # Write validation results into a JSON file in the run dir
            metrics_path = Path(getattr(self, "_run_dir", "pilot_logs")) / "validation_metrics.json"
            try:
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(val_results, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

            # Also push metrics to the logger if available
            if hasattr(self.trainer, "logger") and hasattr(self.trainer.logger, "log_metrics"):
                # If val_results is a list of dicts (as Lightning returns), log the first
                to_log = val_results[0] if isinstance(val_results, (list, tuple)) and val_results else val_results
                try:
                    # flatten nested metrics if necessary
                    if isinstance(to_log, dict):
                        self.trainer.logger.log_metrics({k: float(v) for k, v in to_log.items()})
                except Exception:
                    pass

        except Exception:
            # don't fail the whole run if validation/logging fails
            logging.exception("Validation after training failed")


def cli_main():
    LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "enable_model_summary": False,
            "callbacks": [
                ModelSummary(max_depth=3),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            "devices": 1,
            "gradient_clip_val": 0.01,
            "gradient_clip_algorithm": "norm",
        },
    )


if __name__ == "__main__":
    cli_main()
