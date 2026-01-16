"""
Tool Agent for SmolCC with improved output formatting.

This module provides an agent implementation that uses Rich for
better terminal output, including spinners for long-running operations
and formatted tool outputs.
"""
import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

from smolagents import ToolCallingAgent, Tool, LogLevel, AgentLogger
from smolcc.utils import strip_quotes

from smolcc.tool_output import ToolOutput, ToolCallOutput, AssistantOutput, convert_to_tool_output


class RichConsoleLogger(AgentLogger):
    """
    Custom logger that suppresses default AgentLogger output but provides
    a Rich console for enhanced display.
    """
    def __init__(self, level: LogLevel = LogLevel.INFO, log_file: Optional[str] = None):
        # Create the Rich console for output
        self.console = Console()
        super().__init__(level=level)
        
        # Set up file logging if requested
        self.log_file = log_file
        if log_file:
            # Create or truncate the log file
            with open(log_file, "w") as f:
                f.write("RichConsoleLogger initialized\n")
    
    def log(self, *args, level: int = LogLevel.INFO, **kwargs) -> None:
        """
        Override to suppress default AgentLogger output.
        
        We'll handle our own logging through the Rich console elsewhere.
        """
        if self.log_file and level <= self.level:
            with open(self.log_file, "a") as f:
                f.write(f"LOG: {args}\n")
    
    
    def log_task(self, content: str, subtitle: str, title: str = None, level: LogLevel = LogLevel.INFO) -> None:
        """Suppress task headers."""
        if self.log_file and level <= self.level:
            with open(self.log_file, "a") as f:
                f.write(f"TASK: {content}\nSubtitle: {subtitle}\nTitle: {title}\n")
    
    def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
        """Suppress rule dividers."""
        if self.log_file and level <= self.level:
            with open(self.log_file, "a") as f:
                f.write(f"RULE: {title}\n")
    
    def log_markdown(self, content: str, title: str = None, level=LogLevel.INFO, style=None) -> None:
        """Suppress markdown output."""
        if self.log_file and level <= self.level:
            with open(self.log_file, "a") as f:
                f.write(f"MARKDOWN: {title}\n{content}\n")


class ToolAgent(ToolCallingAgent):
    """
    Tool Agent with improved terminal output.
    
    This agent extends the ToolCallingAgent with better terminal output,
    including spinners for long-running operations and formatted tool outputs.
    """
    def __init__(self, *args, **kwargs):
        # Extract log file if provided
        log_file = kwargs.pop("log_file", None)
        
        # Create our custom logger
        if 'logger' not in kwargs:
            kwargs['logger'] = RichConsoleLogger(level=LogLevel.INFO, log_file=log_file)
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Get the console from the logger
        self.console = self.logger.console
    
    
        
    




def main():
    """
    Run the EnhancedToolAgent with a sample query.
    """
    # Example query
    query = "What files are in the current directory?"
    
    # Create the agent
    agent = create_enhanced_agent()
    
    # Run the query
    agent.run(query)


if __name__ == "__main__":
    main()