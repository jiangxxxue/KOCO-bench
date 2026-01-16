"""
Plan Customization Example

This example demonstrates how to use step callbacks to interrupt the agent after
plan creation, allow user interaction to approve or modify the plan, and then
resume execution while preserving agent memory.

Key concepts demonstrated:
1. Step callbacks to interrupt after PlanningStep
2. Extracting and modifying the current plan
3. Resuming execution with reset=False to preserve memory
4. User interaction for plan approval/modification
"""

from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, PlanningStep


def display_plan(plan_content):
    """Display the plan in a formatted way"""
    print("\n" + "=" * 60)
    print("🤖 AGENT PLAN CREATED")
    print("=" * 60)
    print(plan_content)
    print("=" * 60)


def get_user_choice():
    """Get user's choice for plan approval"""
    while True:
        choice = input("\nChoose an option:\n1. Approve plan\n2. Modify plan\n3. Cancel\nYour choice (1-3): ").strip()
        if choice in ["1", "2", "3"]:
            return int(choice)
        print("Invalid choice. Please enter 1, 2, or 3.")


def get_modified_plan(original_plan):
    """Allow user to modify the plan"""
    print("\n" + "-" * 40)
    print("MODIFY PLAN")
    print("-" * 40)
    print("Current plan:")
    print(original_plan)
    print("-" * 40)
    print("Enter your modified plan (press Enter twice to finish):")

    lines = []
    empty_line_count = 0

    while empty_line_count < 2:
        line = input()
        if line.strip() == "":
            empty_line_count += 1
        else:
            empty_line_count = 0
        lines.append(line)

    # Remove the last two empty lines
    modified_plan = "\n".join(lines[:-2])
    return modified_plan if modified_plan.strip() else original_plan


def interrupt_after_plan(memory_step, agent):
    """
    Step callback that interrupts the agent after a planning step is created.
    This allows for user interaction to review and potentially modify the plan.
    """
    if isinstance(memory_step, PlanningStep):
        print("\n🛑 Agent interrupted after plan creation...")

        # Display the created plan
        display_plan(memory_step.plan)

        # Get user choice
        choice = get_user_choice()

        if choice == 1:  # Approve plan
            print("✅ Plan approved! Continuing execution...")
            # Don't interrupt - let the agent continue
            return

        elif choice == 2:  # Modify plan
            # Get modified plan from user
            modified_plan = get_modified_plan(memory_step.plan)

            # Update the plan in the memory step
            memory_step.plan = modified_plan

            print("\nPlan updated!")
            display_plan(modified_plan)
            print("✅ Continuing with modified plan...")
            # Don't interrupt - let the agent continue with modified plan
            return

        elif choice == 3:  # Cancel
            print("❌ Execution cancelled by user.")
            agent.interrupt()
            return


