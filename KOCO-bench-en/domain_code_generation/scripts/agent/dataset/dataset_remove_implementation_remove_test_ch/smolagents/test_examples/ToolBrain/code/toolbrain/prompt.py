from __future__ import annotations

import inspect
from textwrap import dedent
from typing import Any, Dict, Sequence
from smolagents import Tool, tool, ChatMessage


def _extract_tool_meta(tool: Tool) -> Dict[str, Any]:
    """
    Extracts and returns a uniform metadata dictionary for a smolagents Tool object.

    The returned metadata dictionary provides key information about the Tool, including its name,
    description, and optional attributes such as input/output types and initialization status.

    Args:
        tool (Tool): The smolagents Tool object from which to extract metadata.

    Returns:
        Dict[str, Any]: A dictionary containing the following keys:
            - name (str): Name of the tool.
            - description (str): Description of the tool.
            - inputs (Optional[Any]): Additional inputs attribute if present.
            - output_type (Optional[Any]): Additional output_type attribute if present.
            - is_initialized (Optional[bool]): Additional is_initialized attribute if present.
            - parameters (Dict[str, Any]): Parameters dict extracted from tool.spec if available.
            - returns (Optional[str]): Short returns hint parsed from docstring.
            - examples (Optional[str]): Examples snippet parsed from docstring.
    """
    # TODO: Implement _extract_tool_meta function
    raise NotImplementedError("_extract_tool_meta function needs to be implemented")

def tool_to_card(tool: Tool) -> str:
    """
    Converts a smolagents Tool object into a formatted string card ready for prompt insertion.

    This function extracts key metadata from the Tool and organizes it into a human-readable
    multiline string that can be used for display or documentation purposes.

    Args:
        tool (Tool): The smolagents Tool object to convert.

    Returns:
        str: A formatted string representing the tool, including its name, description,
             inputs, output_type, and is_initialized status.
    """
    meta = _extract_tool_meta(tool)
    lines = []
    lines.append(f"Tool: {meta['name']}")
    lines.append(f"Purpose: {meta['description'] if meta['description'] else 'No description provided.'}")
    lines.append("Args (JSON):")

    if meta["parameters"]:
        for pname, pinfo in meta["parameters"].items():
            ptype = pinfo.get("type", "unknown") if isinstance(pinfo, dict) else "unknown"
            pdesc = pinfo.get("description", "") if isinstance(pinfo, dict) else ""
            pdefault = pinfo.get("default") if isinstance(pinfo, dict) else None
            default_str = f" (default={pdefault})" if pdefault is not None else ""
            lines.append(f"- {pname}: {ptype}, {pdesc}{default_str}".rstrip(", "))
    elif meta["inputs"]:
        # Try to normalize inputs into a {name: schema} dict and render per-arg lines
        input_map = None
        if isinstance(meta["inputs"], dict):
            input_map = meta["inputs"]
        elif isinstance(meta["inputs"], str):
            try:
                import json, ast
                parsed = None
                try:
                    parsed = json.loads(meta["inputs"])
                except Exception:
                    parsed = ast.literal_eval(meta["inputs"])
                if isinstance(parsed, dict):
                    input_map = parsed
            except Exception:
                input_map = None
        if isinstance(input_map, dict):
            for pname, pinfo in input_map.items():
                ptype = pinfo.get("type", "unknown") if isinstance(pinfo, dict) else "unknown"
                pdesc = pinfo.get("description", "") if isinstance(pinfo, dict) else ""
                pdefault = pinfo.get("default") if isinstance(pinfo, dict) else None
                default_str = f" (default={pdefault})" if pdefault is not None else ""
                lines.append(f"- {pname}: {ptype}, {pdesc}{default_str}".rstrip(", "))
        else:
            # fallback to raw repr if we cannot normalize
            lines.append(f"- {meta['inputs']}")
    else:
        lines.append("- None")

    if meta["returns"]:
        lines.append(f"Returns: {meta['returns']}")
    elif meta["output_type"]:
        lines.append(f"Output type: {meta['output_type']}")

    if meta["examples"]:
        lines.append("\nExample:")
        lines.append(meta["examples"])

    card = "\n".join(lines)
    return card


def tools_to_card(tools: Sequence[Any]) -> str:
    """
    Render multiple Tool objects into a prompt-ready string.

    The output begins with a header 'Available tools (N):', where N is the number of tools.
    Each tool is enumerated (1-based) with its card, prefixed by its index (e.g., '1) ').
    Tool cards are separated by a clear separator line '\n---\n', with no trailing separator after the last.
    If the input list is empty, returns 'No tools provided.'

    Args:
        tools (Sequence[Tool]): Sequence of smolagent Tool to render as tool cards.

    Returns:
        str: A formatted string listing all provided tools, or 'No tools provided.' if empty.

    Raises:
        TypeError: If any item in the input is not an instance of Tool.
    """
    # TODO: Implement tools_to_card function
    raise NotImplementedError("tools_to_card function needs to be implemented")



def validate_model(model: Any) -> None:
    """
    Validate the provided LLM model.

    Requirements:
    - model must not be None
    - model must expose a callable `.generate` method
    - `.generate` should accept at least one positional argument (e.g., a list[ChatMessage])
    """
    # TODO: Implement validate_model function
    raise NotImplementedError("validate_model function needs to be implemented")


def validate_tools(tools: Sequence[Tool]) -> None:
    """
    Validate the sequence of tools.

    Requirements:
    - `tools` must be a non-empty sequence (not a string/bytes)
    - every item must be an instance of `smolagents.Tool`
    - each tool must be callable and have a non-empty string `name`
    - tool names must be unique (case-insensitive)
    """
    # TODO: Implement validate_tools function
    raise NotImplementedError("validate_tools function needs to be implemented")


def build_prompt_to_generate_training_examples(
    tools_description: str,
    task_description: str | None = None,
    min_tool_calls: int = 1,
    max_words: int = 80,
    guidance_example: str | None = None
) -> str:
    """
    Constructs a prompt string for generating training examples using the provided tools.

    The prompt includes a task description, a list of available tools with their details,
    and specific instructions for generating examples. It can enforce the use of multiple
    tools in each example if required.

    Args:
        task_description (str): A brief description of the task for which examples are to be generated.
        tools_description (str): A detailed description of tools.
        min_tool_calls (int, optional): Minimum number of distinct tools that must be used in each example. Defaults to 1.
        max_words (int, optional): Maximum word count for the generated example. Defaults to 80.
        guidance_example (Optional[str], optional): A few-shot, non-binding hint shown above the instructions to steer style and difficulty. This is NOT the output to return.

    Returns:
        str: A formatted prompt string ready for use in example generation.

    Raises:
        TypeError: If any item in the tools sequence is not an instance of smolagent Tool.
    """
    # TODO: Implement build_prompt_to_generate_training_examples function
    raise NotImplementedError("build_prompt_to_generate_training_examples function needs to be implemented")
    


if __name__ == "__main__":
    @tool
    def add(x: int, y: int) -> int:
        """Add two numbers.

        Args:
            x (int): First number.
            y (int, optional): Second number.

        Returns:
            int: The sum of x and y.
        """
        return x + y
    
    @tool
    def multiply(a: int, b: int) -> int:
        """
        Multiply two integers.

        Args:
            a (int): First factor.
            b (int): Second factor.

        Returns:
            int: Product of a and b.
        """
        return a * b

    meta = _extract_tool_meta(add)
    card = tool_to_card(add)
    full_card = tools_to_card([add, multiply])
    prompt = build_prompt_to_generate_training_examples(
        task_description="Perform basic arithmetic operations.",
        tools_description=full_card,
        guidance_example=None,
    )
    print(prompt)
