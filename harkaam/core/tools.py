"""
Tools module for the Harkaam framework.

This module provides the foundation for creating and managing tools
that agents can use to interact with external systems or perform
specific operations.
"""

from typing import Any, Callable, Dict, List, Optional, Union
from pydantic import BaseModel, Field, create_model

class ToolParameter(BaseModel):
    """A parameter for a tool."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None

class Tool:
    """
    A tool that can be used by an agent.
    
    Tools provide agents with the ability to interact with external systems
    or perform specific operations.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: List[ToolParameter] = None,
    ):
        """
        Initialize a new tool.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            func: The function to call when the tool is executed
            parameters: A list of parameters for the tool
        """
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or []
        
        # Create a Pydantic model for validating parameters
        param_fields = {}
        for param in self.parameters:
            field_type = self._get_type_from_string(param.type)
            field_default = ...  # Required
            if not param.required:
                field_default = param.default
            
            param_fields[param.name] = (field_type, Field(default=field_default, description=param.description))
        
        self.ParamModel = create_model(f"{name}Params", **param_fields)
    
    def _get_type_from_string(self, type_str: str) -> Any:
        """Convert a string type to a Python type."""
        type_map = {
            "string": str,
            "integer": int,
            "float": float,
            "boolean": bool,
            "array": List,
            "object": Dict,
        }
        return type_map.get(type_str.lower(), Any)
    
    def execute(self, parameters: Dict[str, Any]) -> Any:
        """
        Execute the tool with the given parameters.
        
        Args:
            parameters: The parameters for the tool
            
        Returns:
            The result of the tool execution
        """
        # Validate parameters using the Pydantic model
        validated_params = self.ParamModel(**parameters)
        
        # Execute the function with the validated parameters
        return self.func(**validated_params.dict())

class ToolRegistry:
    """
    A registry for tools.
    
    The tool registry keeps track of available tools and provides
    a way to register and retrieve them.
    """
    
    def __init__(self):
        """Initialize a new tool registry."""
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """
        Register a tool.
        
        Args:
            tool: The tool to register
        """
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            name: The name of the tool
            
        Returns:
            The tool if found, None otherwise
        """
        return self.tools.get(name)
    
    def list(self) -> List[Tool]:
        """
        List all registered tools.
        
        Returns:
            A list of all registered tools
        """
        return list(self.tools.values())