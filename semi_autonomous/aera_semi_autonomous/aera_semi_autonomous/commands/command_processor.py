import yaml
import logging
from typing import List, Tuple, Optional
import numpy as np
from sensor_msgs.msg import Image

from aera_semi_autonomous.vision.object_detector import ObjectDetector
from aera_semi_autonomous.control.robot_interface import RobotInterface
from aera_semi_autonomous.manipulation.manipulation_handler import ManipulationHandler

from aera_semi_autonomous.config.constants import (
    AVAILABLE_ACTIONS,
    _PICK_OBJECT,
    _MOVE_ABOVE_OBJECT_AND_RELEASE,
    _RELEASE_GRIPPER,
    _OBJECT_DETECTION_INDEX,
)


class CommandProcessor:
    def __init__(self, logger):
        self.logger = logger

    def parse_prompt_message(
        self, msg_data: str
    ) -> Optional[Tuple[List[Tuple[str, str]], dict]]:
        """Parse YAML/JSON prompt message into list of (action, object) tuples and offsets."""
        try:
            data = yaml.safe_load(msg_data)

            # Handle backward compatibility: if data is a list, treat as old format
            if isinstance(data, list):
                commands = []
                for command_data in data:
                    if not isinstance(command_data, dict):
                        self.logger.error(
                            f"Command item is not a dictionary: {command_data}"
                        )
                        return None
                    action = command_data.get("action")
                    object_to_detect = command_data.get("object", "")

                    if not action:
                        self.logger.error(
                            f"No 'action' found in command: {command_data}"
                        )
                        return None

                    if action not in AVAILABLE_ACTIONS:
                        self.logger.warn(
                            f"Action: {action} is not valid. Valid actions: {AVAILABLE_ACTIONS}"
                        )
                        return None
                    commands.append((action, object_to_detect))
                return commands, {}

            # Handle new format: dictionary with commands and optional offsets
            if not isinstance(data, dict):
                self.logger.error(
                    f"The top level is not a dictionary or list: {msg_data}"
                )
                return None

            # Extract commands
            commands_data = data.get("commands", [])
            if not isinstance(commands_data, list):
                self.logger.error(f"'commands' is not a list: {commands_data}")
                return None

            commands = []
            for command_data in commands_data:
                if not isinstance(command_data, dict):
                    self.logger.error(
                        f"Command item is not a dictionary: {command_data}"
                    )
                    return None
                action = command_data.get("action")
                object_to_detect = command_data.get("object", "")

                if not action:
                    self.logger.error(f"No 'action' found in command: {command_data}")
                    return None

                if action not in AVAILABLE_ACTIONS:
                    self.logger.warn(
                        f"Action: {action} is not valid. Valid actions: {AVAILABLE_ACTIONS}"
                    )
                    return None
                commands.append((action, object_to_detect))

            # Extract offsets (optional)
            offsets = {}
            if "offsets" in data:
                offset_data = data["offsets"]
                if isinstance(offset_data, dict):
                    offsets["offset_x"] = offset_data.get("offset_x")
                    offsets["offset_y"] = offset_data.get("offset_y")
                    offsets["offset_z"] = offset_data.get("offset_z")

            return commands, offsets
        except yaml.YAMLError:
            self.logger.error(f"Failed to parse YAML/JSON from prompt: {msg_data}")
            return None

    def handle_tool_call(
        self,
        tool_call: str,
        object_to_detect: str,
        object_detector: ObjectDetector,
        robot: RobotInterface,
        manipulation_handler: ManipulationHandler,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        object_in_gripper: bool,
    ) -> bool:
        """Handle a single tool call and return updated object_in_gripper status."""
        if tool_call == _PICK_OBJECT:
            detections = object_detector.detect_objects(rgb_image, [object_to_detect])
            if detections.class_id is None or len(detections.class_id) == 0:
                self.logger.info(
                    f"No {object_to_detect} detected. Got the following detection: {detections.class_id}"
                )
                return object_in_gripper
            if object_in_gripper:
                logging.error("Object in gripper")
                return object_in_gripper

            success = manipulation_handler.pick_object(
                _OBJECT_DETECTION_INDEX, detections, depth_image, rgb_image
            )
            return True if success else object_in_gripper

        elif tool_call == _MOVE_ABOVE_OBJECT_AND_RELEASE:
            detections = object_detector.detect_objects(rgb_image, [object_to_detect])
            if detections.class_id is None or len(detections.class_id) == 0:
                self.logger.info(
                    f"No {object_to_detect} detected. Got the following detection: {detections.class_id}"
                )
                return object_in_gripper
            success = manipulation_handler.release_above(
                _OBJECT_DETECTION_INDEX, detections, depth_image, rgb_image
            )
            return False if success else object_in_gripper

        elif tool_call == _RELEASE_GRIPPER:
            success = robot.release_gripper()
            return False if success else object_in_gripper

        return object_in_gripper
