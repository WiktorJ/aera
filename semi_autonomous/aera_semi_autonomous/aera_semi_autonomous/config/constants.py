import os
import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GroundingDINO config and checkpoint
GSA_PATH = "./Grounded-Segment-Anything/Grounded-Segment-Anything"
GROUNDING_DINO_CONFIG_PATH = os.path.join(
    GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "sam_vit_h_4b8939.pth")

# Predict classes and hyper-param for GroundingDINO
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

# Action constants
_PICK_OBJECT = "pick_object"
_MOVE_ABOVE_OBJECT_AND_RELEASE = "move_above_object_and_release"
_RELEASE_GRIPPER = "release_gripper"

# Mapping between actions and verbose descriptions in natural language
ACTION_DESCRIPTIONS = {
    _PICK_OBJECT: "Pick up the {object_name} by grasping it with the robot gripper",
    _MOVE_ABOVE_OBJECT_AND_RELEASE: "Move above the {object_name} and release the currently held object",
    _RELEASE_GRIPPER: "Release the gripper to drop any currently held object",
}

AVAILABLE_ACTIONS = (
    _PICK_OBJECT,
    _MOVE_ABOVE_OBJECT_AND_RELEASE,
    _RELEASE_GRIPPER,
)

_TF_PREFIX = "camera"
BASE_LINK_NAME = f"{_TF_PREFIX}base_link"
