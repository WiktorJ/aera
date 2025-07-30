import yaml
import logging
from typing import List, Tuple, Optional

from aera_semi_autonomous.config.constants import (
    _AVAILABLE_ACTIONS,
    _PICK_OBJECT,
    _MOVE_ABOVE_OBJECT_AND_RELEASE,
    _RELEASE_GRIPPER,
    _OBJECT_DETECTION_INDEX,
)


class CommandProcessor:
    def __init__(self, logger):
        self.logger = logger

    def parse_prompt_message(self, msg_data: str) -> Optional[List[Tuple[str, str]]]:
        """Parse YAML/JSON prompt message into list of (action, object) tuples."""
        try:
            data = yaml.safe_load(msg_data)

            if not isinstance(data, list):
                self.logger.error(f"The top level is not a list: {msg_data}")
                return None

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
                    self.logger.error(f"No 'action' found in command: {command_data}")
                    return None

                if action not in _AVAILABLE_ACTIONS:
                    self.logger.warn(
                        f"Action: {action} is not valid. Valid actions: {_AVAILABLE_ACTIONS}"
                    )
                    return None
                commands.append((action, object_to_detect))
            return commands
        except yaml.YAMLError:
            self.logger.error(f"Failed to parse YAML/JSON from prompt: {msg_data}")
            return None

    def handle_tool_call(
        self,
        tool_call: str,
        object_to_detect: str,
        object_detector,
        robot_controller,
        manipulation_handler,
        rgb_image,
        depth_image,
        object_in_gripper: bool,
        last_rgb_msg=None,
    ) -> bool:
        """Handle a single tool call and return updated object_in_gripper status."""
        if tool_call == _PICK_OBJECT:
            detections = object_detector.detect_objects(rgb_image, [object_to_detect])
            if len(detections.class_id) == 0:
                self.logger.info(
                    f"No {object_to_detect} detected. Got the following detection: {detections.class_id}"
                )
                return object_in_gripper
            if object_in_gripper:
                logging.error("Object in gripper")
                return object_in_gripper

            manipulation_handler.pick_object(_OBJECT_DETECTION_INDEX, detections, depth_image, last_rgb_msg)
            return True
            
        elif tool_call == _MOVE_ABOVE_OBJECT_AND_RELEASE:
            detections = object_detector.detect_objects(rgb_image, [object_to_detect])
            if len(detections.class_id) == 0:
                self.logger.info(
                    f"No {object_to_detect} detected. Got the following detection: {detections.class_id}"
                )
                return object_in_gripper
            manipulation_handler.release_above(_OBJECT_DETECTION_INDEX, detections, depth_image, last_rgb_msg)
            return False
            
        elif tool_call == _RELEASE_GRIPPER:
            robot_controller.release_gripper()
            return False
            
        return object_in_gripper
