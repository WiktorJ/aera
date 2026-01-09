import dataclasses
import pathlib

import tyro
from typing_extensions import override

import aera.autonomous.openpi.data_transform as data_transform
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.training.config as openpi_config
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class Ar4Mk3DataConfig(openpi_config.DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    """

    extra_delta_transform: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> openpi_config.DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.

        # repack_transform = _transforms.Group(
        #     inputs=[
        #         _transforms.RepackTransform(
        #             {
        #                 "image": "image",
        #                 "state": "state",
        #                 "actions": "actions",
        #                 "prompt": "prompt",
        #             }
        #         )
        #     ]
        # )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        data_transforms = _transforms.Group(
            inputs=[data_transform.Ar4Mk3Inputs(model_type=model_config.model_type)],
            outputs=[data_transform.Ar4Mk3Outputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.

        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = openpi_config.ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


_CONFIGS = [
    openpi_config.TrainConfig(
        name="pi0_ar4_mk3_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_horizon=10,
        ),
        data=Ar4Mk3DataConfig(
            repo_id="Purple69/aera_semi_pnp_dr_08_01_2026",
            base_config=openpi_config.DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            assets=openpi_config.AssetsConfig(
                assets_dir="hf://datasets/Purple69/aera_semi_pnp_dr_08_01_2026/",
                asset_id="assets",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
        num_train_steps=8_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        batch_size=16,
        log_interval=100,
        keep_period=500,
    ),
    openpi_config.TrainConfig(
        name="pi0_fast_ar4_mk3_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=100,
            paligemma_variant="gemma_2b_lora",
        ),
        data=Ar4Mk3DataConfig(
            repo_id="Purple69/aera_semi_pnp_dr_08_01_2026",
            base_config=openpi_config.DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            assets=openpi_config.AssetsConfig(
                assets_dir="hf://datasets/Purple69/aera_semi_pnp_dr_08_01_2026/",
                asset_id="assets",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
        num_train_steps=8_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        batch_size=16,
        log_interval=100,
        keep_period=1000,
    ),
    openpi_config.TrainConfig(
        name="pi0_fast_ar4_mk3_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=100,
        ),
        data=Ar4Mk3DataConfig(
            repo_id="Purple69/aera_semi_pnp_dr_08_01_2026",
            base_config=openpi_config.DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            assets=openpi_config.AssetsConfig(
                assets_dir="hf://datasets/Purple69/aera_semi_pnp_dr_08_01_2026/",
                asset_id="assets",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
        num_train_steps=8_000,
        log_interval=100,
        keep_period=1000,
    ),
    openpi_config.TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"
        ),
        data=openpi_config.FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=True,
    ),
]


_CONFIGS_DICT = {c.name: c for c in _CONFIGS}


def cli() -> openpi_config.TrainConfig:
    return tyro.extras.overridable_config_cli(
        {k: (k, v) for k, v in _CONFIGS_DICT.items()}
    )


def get_config(config_name: str) -> openpi_config.TrainConfig:
    """Get a config by name."""
    return _CONFIGS_DICT[config_name]
