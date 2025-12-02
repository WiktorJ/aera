"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import aera.autonomous.openpi.training_config as _training_config
import openpi.training.config as _openpi_config
import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {
            k: v
            for k, v in x.items()
            if not np.issubdtype(np.asarray(v).dtype, np.str_)
        }


def create_torch_dataloader(
    data_config: _openpi_config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.TorchDataLoader, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str, max_frames: int | None = None, push_to_hub: bool = False
):
    config = _training_config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    data_loader, num_batches = create_torch_dataloader(
        data_config,
        config.model.action_horizon,
        config.batch_size,
        config.model,
        config.num_workers,
        max_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / (
        data_config.repo_id if data_config.repo_id else ""
    )
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    if push_to_hub and data_config.repo_id:
        print(f"Pushing stats to Hugging Face Hub: {data_config.repo_id}")
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path=output_path,
            repo_id=data_config.repo_id,
            repo_type="dataset",
            path_in_repo="assets",
        )


if __name__ == "__main__":
    tyro.cli(main)
