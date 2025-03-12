"""
Base agent class that provides the foundation for all agent architectures.
With added support for verbose mode during execution and formatted output.
"""

from typing import Any, Dict, List, Optional, Union
import uuid
from abc import ABC, abstractmethod
import json
import sys
import time
from datetime import datetime
import textwrap

from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    """Configuration for an agent."""
    name: str
    description: str
    llm: str  # Format: "provider:model" (e.g., "openai:gpt-4")
    temperature: float = 0.7
    max_tokens: int = 1000
    system_prompt: Optional[str] = None
    verbose: bool = False  # Flag for verbose output

class AgentState(BaseModel):
    """State of an agent during execution."""
    stage: str = "idle"
    step_count: int = 0
    context: Dict[str, Any] = Field(default_factory=dict)
    working_memory: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)

class AgentResult(BaseModel):
    """Result of an agent execution."""
    agent_id: str
    output: Any
    intermediate_steps: List[Dict[str, Any]] = Field(default_factory=list)
    final_state: Optional[AgentState] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def format_output(self, verbose: bool = False) -> str:
        """
        Format the agent result in a user-friendly way.
        
        Args:
            verbose: Whether to include detailed thinking steps
            
        Returns:
            A formatted string representation of the result
        """
        # Start with the basic output
        formatted = f"Result from {self.metadata.get('architecture', 'Agent').upper()} Agent:\n"
        formatted += "=" * 80 + "\n\n"
        
        # Add the final output
        formatted += f"ANSWER:\n{self.output}\n\n"
        
        # If verbose, add thinking steps
        if verbose:
            formatted += "THINKING PROCESS:\n" + "-" * 40 + "\n"
            
            for step in self.intermediate_steps:
                step_type = step.get('type', '').upper()
                content = step.get('content', '')
                
                if step_type and content:
                    formatted += f"{step_type}:\n"
                    if isinstance(content, list):
                        content = '\n'.join(content)
                    # Indent content for readability
                    indented_content = textwrap.indent(content, '  ')
                    formatted += f"{indented_content}\n\n"
        
        # Add metadata
        formatted += "-" * 40 + "\n"
        formatted += f"Iterations: {self.metadata.get('iterations', 0)}\n"
        
        return formatted
    
    def __str__(self) -> str:
        """String representation that calls format_output."""
        return self.format_output()

class BaseAgent(ABC):
    """
    Base agent class that all architecture-specific agents inherit from.
    
    This class defines the common interface and shared functionality
    for all agent architectures in the Harkaam framework.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        llm: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        memory: Optional[Any] = None,
        verbose: bool = False,
        **kwargs  # Add this to handle additional architecture-specific parameters
    ):
        """
        Initialize a new agent.
        
        Args:
            name: The name of the agent
            description: A description of the agent's purpose
            llm: The LLM to use, in format "provider:model" (e.g., "openai:gpt-4")
            temperature: The temperature for the LLM
            max_tokens: The maximum number of tokens for the LLM response
            system_prompt: An optional system prompt to override the default
            tools: A list of tools the agent can use
            memory: An optional memory system for the agent
            verbose: Whether to display verbose output during execution
        """
        self.id = str(uuid.uuid4())
        self.config = AgentConfig(
            name=name,
            description=description,
            llm=llm,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt or self._default_system_prompt(name, description),
            verbose=verbose
        )
        
        self.tools = tools or []
        self.memory = memory
        self.state = AgentState()
        
        # For tracking execution time in verbose mode
        self._start_time = None
        
        # LLM client will be initialized by the subclass
        
        # Print initialization in verbose mode
        if self.config.verbose:
            architecture = self.__class__.__name__.replace('Agent', '')
            self.log(f"Initialized {architecture} agent: {self.config.name}")

    def _default_system_prompt(self, name: str, description: str) -> str:
        """Generate a default system prompt for the agent."""
        return f"""You are {name}, {description}.
        
Your goal is to complete the assigned task to the best of your ability.
Think step by step about the task and provide a clear, accurate response.
"""

    @classmethod
    def create(cls, architecture: str, **kwargs):
        """
        Factory method to create an agent with the specified architecture.
        
        Args:
            architecture: The agent architecture to use
            **kwargs: Arguments to pass to the agent constructor
            
        Returns:
            An instance of the specified agent architecture
        """
        from harkaam.agents.react import ReActAgent
        from harkaam.agents.ooda import OODAAgent
        from harkaam.agents.bdi import BDIAgent
        from harkaam.agents.lat import LATAgent
        from harkaam.agents.raise_agent import RAISEAgent
        from harkaam.agents.rewoo import ReWOOAgent
        
        architecture_map = {
            "react": ReActAgent,
            "ooda": OODAAgent,
            "bdi": BDIAgent,
            "lat": LATAgent,
            "raise": RAISEAgent,
            "rewoo": ReWOOAgent,
        }
        
        if architecture.lower() not in architecture_map:
            raise ValueError(f"Unknown agent architecture: {architecture}")
        
        agent_class = architecture_map[architecture.lower()]
        return agent_class(**kwargs)

    def run(self, task: str, **kwargs) -> Union[AgentResult, str]:
        """
        Run the agent on a task.
        
        This is a convenience method that calls the architecture-specific
        execute method with the task.
        
        Args:
            task: The task for the agent to execute
            **kwargs: Additional arguments for execution
            
        Returns:
            The result of the execution, formatted if format_output is True
        """
        # Get formatting option
        format_output = kwargs.pop("format_output", True)
        
        # Reset the agent state
        self.state = AgentState()
        
        # Start timing if in verbose mode
        if self.config.verbose:
            self._start_time = time.time()
            self.log(f"Starting task: {task}")
        
        # Call the architecture-specific execute method
        result = self.execute(task, **kwargs)
        
        # Log completion if in verbose mode
        if self.config.verbose:
            elapsed = time.time() - self._start_time
            self.log(f"Task completed in {elapsed:.2f} seconds")
        
        # Format result if requested
        if format_output:
            return result.format_output(verbose=self.config.verbose)
        else:
            return result

    @abstractmethod
    def execute(self, task: str, **kwargs) -> AgentResult:
        """
        Execute a task using the agent's architecture.
        
        This method must be implemented by each architecture-specific
        agent subclass.
        
        Args:
            task: The task for the agent to execute
            **kwargs: Additional arguments for execution
            
        Returns:
            The result of the execution
        """
        pass

    def _update_state(self, **kwargs) -> None:
        """
        Update the agent's state and log in verbose mode.
        
        Args:
            **kwargs: State attributes to update
        """
        # Extract information for logging
        stage = kwargs.get('stage', None)
        if stage and self.config.verbose:
            self.log(f"Stage: {stage}")
        
        # Update state first
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
            else:
                # For context and working_memory
                if key == "context_update" and isinstance(value, dict):
                    self.state.context.update(value)
                elif key == "memory_update" and isinstance(value, dict):
                    self.state.working_memory.update(value)
                elif key == "add_to_history" and isinstance(value, dict):
                    self.state.history.append(value)
                    
                    # Log in verbose mode
                    if self.config.verbose:
                        self._log_thinking(value)
    
    def _log_thinking(self, history_item: Dict[str, Any]) -> None:
        """
        Log thinking patterns in verbose mode.
        """
        if not self.config.verbose:
            return
            
        # Get architecture name from class
        architecture = self.__class__.__name__.lower()
        
        # Handle BDI
        if 'bdi' in architecture:
            if "beliefs" in history_item and history_item["beliefs"]:
                self.log(f"💭 BELIEFS: {self._truncate(history_item['beliefs'])}")
            if "desires" in history_item and history_item["desires"]:
                self.log(f"💭 DESIRES: {self._truncate(history_item['desires'])}")
            if "intentions" in history_item and history_item["intentions"]:
                self.log(f"💭 INTENTIONS: {self._truncate(history_item['intentions'])}")
            if "actions" in history_item and history_item["actions"]:
                self.log(f"🔧 ACTIONS: {self._truncate(history_item['actions'])}")
            if "results" in history_item and history_item["results"]:
                self.log(f"📊 RESULTS: {self._truncate(history_item['results'])}")
        
        # Handle ReAct
        elif 'react' in architecture:
            if "thought" in history_item and history_item["thought"]:
                self.log(f"💭 THOUGHT: {self._truncate(history_item['thought'])}")
            if "action" in history_item and history_item["action"]:
                self.log(f"🔧 ACTION: {self._truncate(history_item['action'])}")
            if "observation" in history_item and history_item["observation"]:
                self.log(f"👁️ OBSERVATION: {self._truncate(history_item['observation'])}")
        
        # Handle OODA
        elif 'ooda' in architecture:
            if "observe" in history_item and history_item["observe"]:
                self.log(f"👁️ OBSERVE: {self._truncate(history_item['observe'])}")
            if "orient" in history_item and history_item["orient"]:
                self.log(f"🧭 ORIENT: {self._truncate(history_item['orient'])}")
            if "decide" in history_item and history_item["decide"]:
                self.log(f"🤔 DECIDE: {self._truncate(history_item['decide'])}")
            if "act" in history_item and history_item["act"]:
                self.log(f"🔧 ACT: {self._truncate(history_item['act'])}")
                
        # Generic handling for other architectures or undefined patterns
        else:
            for key, value in history_item.items():
                if isinstance(value, str) and value.strip():
                    self.log(f"💭 {key.upper()}: {self._truncate(value)}")
    
    def log(self, message: str) -> None:
        """
        Log a message in verbose mode.
        
        Args:
            message: The message to log
        """
        if not self.config.verbose:
            return
            
        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        agent_name = self.config.name
        
        # Print log message
        print(f"[{timestamp}] 🤖 {agent_name}: {message}")
    
    def _truncate(self, text: str, ) -> str:
        """
        Truncate text for display in logs.
        
        Args:
            text: The text to truncate
            
        Returns:
            Truncated text
        """
        return text