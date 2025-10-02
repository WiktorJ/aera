import unittest
import json
from unittest.mock import Mock

# NOTE: This test assumes 'aera_semi_autonomous.commands.command_processor' can be imported
# and that the CommandProcessor class is initialized with its dependencies.
from aera_semi_autonomous.commands.command_processor import CommandProcessor


class TestCommandProcessor(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, including mocks for dependencies."""
        self.mock_logger = Mock()
        
        # Based on the actual CommandProcessor constructor signature
        self.processor = CommandProcessor(self.mock_logger)

    def test_parse_prompt_message_pick(self):
        """Test parsing a valid 'pick_object' prompt."""
        prompt = '{"tool_calls": [{"name": "pick_object", "args": {"object_name": "red_cube"}}]}'
        result = self.processor.parse_prompt_message(prompt)
        self.assertIsNotNone(result)
        tool_calls, args = result
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0][0], "pick_object")
        self.assertEqual(tool_calls[0][1], "red_cube")

    def test_parse_prompt_message_release_above(self):
        """Test parsing a valid 'move_above_object_and_release' prompt."""
        prompt = '{"tool_calls": [{"name": "move_above_object_and_release", "args": {"object_name": "blue_bowl"}}]}'
        result = self.processor.parse_prompt_message(prompt)
        self.assertIsNotNone(result)
        tool_calls, args = result
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0][0], "move_above_object_and_release")
        self.assertEqual(tool_calls[0][1], "blue_bowl")

    def test_parse_prompt_message_release_gripper(self):
        """Test parsing a valid 'release_gripper' prompt."""
        prompt = '{"tool_calls": [{"name": "release_gripper", "args": {}}]}'
        result = self.processor.parse_prompt_message(prompt)
        self.assertIsNotNone(result)
        tool_calls, args = result
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0][0], "release_gripper")
        self.assertEqual(tool_calls[0][1], "")

    def test_parse_prompt_message_malformed(self):
        """Test parsing a malformed (non-JSON) prompt string."""
        prompt = 'this is not json'
        result = self.processor.parse_prompt_message(prompt)
        self.assertIsNone(result)

    def test_parse_prompt_message_unknown_tool(self):
        """Test parsing a prompt with an unknown tool call."""
        prompt = '{"tool_calls": [{"name": "unknown_tool", "args": {}}]}'
        result = self.processor.parse_prompt_message(prompt)
        self.assertIsNone(result)

    def test_handle_tool_call_pick_object(self):
        """Test handling a 'pick_object' tool call."""
        # Create mock dependencies
        mock_object_detector = Mock()
        mock_robot = Mock()
        mock_manipulation_handler = Mock()
        mock_rgb_image = Mock()
        mock_depth_image = Mock()
        
        # Mock the detections object properly
        mock_detections = Mock()
        mock_detections.class_id = [0]  # Mock as list with one detection
        mock_object_detector.detect_objects.return_value = mock_detections
        mock_manipulation_handler.pick_object.return_value = True
        
        result = self.processor.handle_tool_call(
            "pick_object",
            "red_cube", 
            mock_object_detector,
            mock_robot,
            mock_manipulation_handler,
            mock_rgb_image,
            mock_depth_image,
            False
        )

        # Verify the method returns True for successful pick
        self.assertTrue(result)

    def test_handle_tool_call_move_above_object_and_release(self):
        """Test handling a 'move_above_object_and_release' tool call."""
        # Create mock dependencies
        mock_object_detector = Mock()
        mock_robot = Mock()
        mock_manipulation_handler = Mock()
        mock_rgb_image = Mock()
        mock_depth_image = Mock()
        
        # Mock the detections object properly
        mock_detections = Mock()
        mock_detections.class_id = [0]  # Mock as list with one detection
        mock_object_detector.detect_objects.return_value = mock_detections
        mock_manipulation_handler.release_above.return_value = True
        
        result = self.processor.handle_tool_call(
            "move_above_object_and_release",
            "blue_bowl",
            mock_object_detector,
            mock_robot,
            mock_manipulation_handler,
            mock_rgb_image,
            mock_depth_image,
            True  # object_in_gripper = True
        )

        # Verify the method returns False for successful release
        self.assertFalse(result)

    def test_handle_tool_call_release_gripper(self):
        """Test handling a 'release_gripper' tool call."""
        # Create mock dependencies
        mock_object_detector = Mock()
        mock_robot = Mock()
        mock_manipulation_handler = Mock()
        mock_rgb_image = Mock()
        mock_depth_image = Mock()
        
        result = self.processor.handle_tool_call(
            "release_gripper",
            "",
            mock_object_detector,
            mock_robot,
            mock_manipulation_handler,
            mock_rgb_image,
            mock_depth_image,
            False
        )

        # Just verify the method can be called without error
        self.assertIsNotNone(result)
