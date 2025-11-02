import dataclasses
import pathlib

from typing_extensions import override

import openpi.models.model as _model
import openpi.transforms as _transforms
import openpi.training.config as openpi_config
import aera.autonomous.openpi.data_transform as ar4mk3_data_transform


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
            inputs=[
                ar4mk3_data_transform.Ar4mk3Inputs(model_type=model_config.model_type)
            ],
            outputs=[ar4mk3_data_transform.Ar4mk3Outputs()],
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
