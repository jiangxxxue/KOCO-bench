"""
Tool output formatting classes for SmolCC.

This module provides classes for tool output formatting with rich display capabilities,
styled to match the desired appearance.
"""
from typing import Any, Optional, List, Dict, Union
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.box import ROUNDED


class ToolOutput:
    """
    Container for tool execution results with display information.
    
    Base class for all tool output formats, provides a consistent
    interface for displaying different types of output.
    """
    
    def __init__(self, data: Any):
        """
        Initialize a ToolOutput object.
        
        Args:
            data: The raw data returned by the tool
        """
        self.data = data
    
    def __str__(self) -> str:
        """String representation for backward compatibility."""
        return str(self.data)
    
    def display(self, console: Console) -> None:
        """
        Display the output in the console using styling.
        
        Args:
            console: The Rich console to display the output in
        """
        # Default display with styling
        console.print(f"  ⎿  {str(self.data)}", style="bright_black")


class ToolCallOutput(ToolOutput):
    """Output format for displaying a tool call."""
    
    def __init__(self, tool_name: str, params: Union[Dict, str, Any]):
        """
        Initialize a ToolCallOutput object.
        
        Args:
            tool_name: Name of the tool being called
            params: Parameters being passed to the tool
        """
        # Format the tool call string 
        if isinstance(params, dict):
            param_str = ", ".join([f"{k}: {repr(v)}" for k, v in params.items()])
            tool_call = f"{tool_name}({param_str})"
        else:
            tool_call = f"{tool_name}({repr(params)})"
        
        # Truncate if too long
        if len(tool_call) > 80:
            tool_call = tool_call[:77] + "..."
            
        super().__init__(tool_call)
        self.tool_name = tool_name
        self.params = params
    


class TextOutput(ToolOutput):
    """Simple text output with UI styling."""
    



class TableOutput(ToolOutput):

