# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms

CLASS_MAPPING = {i: i - 1 for i in range(1, 151)}


class ADE20KSemantic(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (512, 512),
        num_classes: int = 150,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
        check_empty_targets=True,
        debug_overfit: bool = False,  # Enable to force train == val for debugging
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        self.debug_overfit = debug_overfit

        # Disable random augmentations when debugging/overfitting single images
        if self.debug_overfit:
            color_jitter_enabled = False
            scale_range = (1.0, 1.0)

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(target, **kwargs):
        masks, labels = [], []

        for label_id in target[0].unique():
            cls_id = label_id.item()

            if cls_id not in CLASS_MAPPING:
                continue

            masks.append(target[0] == label_id)
            labels.append(CLASS_MAPPING[cls_id])

        return masks, labels, [False for _ in range(len(masks))]

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        import zipfile

        dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "zip_path": Path(self.path, "ADEChallengeData2016.zip"),
            "target_zip_path": Path(self.path, "ADEChallengeData2016.zip"),
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
        }

        # Detect whether the zip contains a top-level ADEChallengeData2016 folder
        zip_path = dataset_kwargs["zip_path"]
        candidate_with_root_imgs = Path("ADEChallengeData2016/images/training").as_posix()
        candidate_without_root_imgs = Path("images/training").as_posix()
        try:
            with zipfile.ZipFile(zip_path) as z:
                names = z.namelist()
                if any(n.startswith(candidate_with_root_imgs) for n in names):
                    img_train_path = Path("ADEChallengeData2016/images/training")
                    img_val_path = Path("ADEChallengeData2016/images/validation")
                    ann_train_path = Path("ADEChallengeData2016/annotations/training")
                    ann_val_path = Path("ADEChallengeData2016/annotations/validation")
                else:
                    img_train_path = Path("images/training")
                    img_val_path = Path("images/validation")
                    ann_train_path = Path("annotations/training")
                    ann_val_path = Path("annotations/validation")
        except FileNotFoundError:
            # Fall back to default expected paths; Dataset will handle missing files gracefully
            img_train_path = Path("ADEChallengeData2016/images/training")
            img_val_path = Path("ADEChallengeData2016/images/validation")
            ann_train_path = Path("ADEChallengeData2016/annotations/training")
            ann_val_path = Path("ADEChallengeData2016/annotations/validation")

        self.train_dataset = Dataset(
            img_folder_path_in_zip=img_train_path,
            target_folder_path_in_zip=ann_train_path,
            transforms=self.transforms,
            **dataset_kwargs,
        )
        if self.debug_overfit:
            self.val_dataset = self.train_dataset
        else:
            self.val_dataset = Dataset(
                img_folder_path_in_zip=img_val_path,
                target_folder_path_in_zip=ann_val_path,
                transforms=self.transforms,
                **dataset_kwargs,
            )

        return self

    def train_dataloader(self):
        dataset = self.train_dataset

        return DataLoader(
            dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
