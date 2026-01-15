import unittest
from unittest.mock import patch, MagicMock, call, ANY
import sys
import os

# Add the parent directory to the sys.path to allow for correct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plan_customization')))

# Import the functions and classes to be tested/mocked
import plan_customization
from plan_customization import PlanningStep

class TestPlanCustomization(unittest.TestCase):
    """Tests the plan customization functionality."""

    @patch('plan_customization.get_user_choice', return_value=1) # 1 = Approve
    def test_interrupt_approve_plan(self, mock_get_user_choice):
        """Test the interrupt callback when the user approves the plan."""
        mock_agent = MagicMock()
        mock_plan_step = MagicMock(spec=PlanningStep)
        mock_plan_step.plan = "Original Plan"

        plan_customization.interrupt_after_plan(mock_plan_step, mock_agent)

        mock_get_user_choice.assert_called_once()
        mock_agent.interrupt.assert_not_called()
        self.assertEqual(mock_plan_step.plan, "Original Plan") # Plan should not be modified

    @patch('plan_customization.get_modified_plan', return_value="Modified Plan")
    @patch('plan_customization.get_user_choice', return_value=2) # 2 = Modify
    def test_interrupt_modify_plan(self, mock_get_user_choice, mock_get_modified_plan):
        """Test the interrupt callback when the user modifies the plan."""
        mock_agent = MagicMock()
        mock_plan_step = MagicMock(spec=PlanningStep)
        mock_plan_step.plan = "Original Plan"

        plan_customization.interrupt_after_plan(mock_plan_step, mock_agent)

        mock_get_user_choice.assert_called_once()
        mock_get_modified_plan.assert_called_once_with("Original Plan")
        mock_agent.interrupt.assert_not_called()
        self.assertEqual(mock_plan_step.plan, "Modified Plan") # Plan should be updated

    @patch('plan_customization.get_user_choice', return_value=3) # 3 = Cancel
    def test_interrupt_cancel_plan(self, mock_get_user_choice):
        """Test the interrupt callback when the user cancels the execution."""
        mock_agent = MagicMock()
        mock_plan_step = MagicMock(spec=PlanningStep)
        mock_plan_step.plan = "Original Plan"

        plan_customization.interrupt_after_plan(mock_plan_step, mock_agent)

        mock_get_user_choice.assert_called_once()
        mock_agent.interrupt.assert_called_once()

    @patch('plan_customization.CodeAgent')
    @patch('plan_customization.InferenceClientModel')
    @patch('plan_customization.DuckDuckGoSearchTool')
    def test_plan_customization_main_flow(self, MockDuckDuckGoSearchTool, MockInferenceClientModel, MockCodeAgent):
        """Test the main plan_customization function."""
        # --- Mock Initialization ---
        mock_agent_instance = MagicMock()
        mock_agent_instance.run.return_value = "Final agent result."
        MockCodeAgent.return_value = mock_agent_instance

        # --- Execute the function ---
        # We patch the input function within the main function's scope to avoid test prompts
        with patch('builtins.input', return_value='n'): # Simulate 'n' for resume demo
            result = plan_customization.plan_customization()

        # --- Assertions ---
        MockCodeAgent.assert_called_once_with(
            model=ANY,
            tools=ANY,
            planning_interval=5,
            step_callbacks={PlanningStep: plan_customization.interrupt_after_plan},
            max_steps=10,
            verbosity_level=1,
        )
        mock_agent_instance.run.assert_called_once()
        self.assertEqual(result, "Final agent result.")

if __name__ == '__main__':
    unittest.main()