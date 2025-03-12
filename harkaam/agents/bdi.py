"""
Simplified BDI (Belief, Desire, Intention) agent implementation.

A BDI agent architecture based on the philosophical theory of human practical reasoning,
consisting of three main components: beliefs, desires, and intentions.
"""

from typing import Any, Dict, List, Tuple
import re
import json

from harkaam.agents.base import BaseAgent, AgentResult
from harkaam.core.tools import Tool, ToolRegistry
from harkaam.core.llm import create_llm
from harkaam.core.prompt import get_prompt_for_architecture
from harkaam.core.parser import create_parser

class BDIAgent(BaseAgent):
    """
    BDI agent implementation with three main components:
    1. Beliefs: The agent's knowledge about the world
    2. Desires: The agent's goals or objectives
    3. Intentions: The agent's committed plans to achieve its desires
    """
    
    def __init__(self, **kwargs):
        """Initialize a new BDI agent."""
        super().__init__(**kwargs)
        
        # Initialize core components
        self.llm_client = create_llm(self.config.llm)
        self.max_iterations = kwargs.get("max_iterations", 10)
        self.tool_registry = ToolRegistry()
        
        # Register tools
        self.tools = kwargs.get("tools", [])
        for tool in self.tools:
            self.tool_registry.register(tool)
    
    def execute(self, task: str, **kwargs) -> AgentResult:
        """
        Execute a task using the BDI architecture cycle:
        1. Update beliefs based on task and environment
        2. Generate possible desires (goals) based on beliefs
        3. Filter desires to intentions
        4. Select and execute actions to achieve intentions
        5. Update beliefs based on results
        6. Repeat until task is complete
        """
        # Initialize context and tracking variables
        context = self._initialize_context(task, **kwargs)
        iterations = 0
        is_done = False
        final_answer = ""
        intermediate_steps = []
        
        # Main BDI cycle
        while not is_done and iterations < self.max_iterations:
            iterations += 1
            
            # Update beliefs based on current inputs
            beliefs = self._update_beliefs(context)
            context["beliefs"].append(beliefs)
            intermediate_steps.append({"type": "beliefs", "content": beliefs})
            
            # Generate desires based on beliefs
            desires = self._generate_desires(context)
            context["desires"].append(desires)
            intermediate_steps.append({"type": "desires", "content": desires})
            
            # Filter desires to intentions
            intentions = self._filter_to_intentions(context)
            context["intentions"].append(intentions)
            intermediate_steps.append({"type": "intentions", "content": intentions})
            
            # Select and execute actions
            actions = self._select_actions(context)
            context["selected_actions"].append(actions)
            intermediate_steps.append({"type": "actions", "content": actions})
            
            action_results = self._execute_actions(context)
            context["actions_results"].append(action_results)
            intermediate_steps.append({"type": "results", "content": action_results})
            
            # Check if task is complete
            is_done, final_answer = self._check_completion(context)
            
            # Update state
            self._update_state(
                stage=f"iteration_{iterations}", 
                step_count=iterations,
                add_to_history={
                    "beliefs": beliefs,
                    "desires": desires,
                    "intentions": intentions,
                    "actions": actions,
                    "results": action_results
                }
            )
        
        # Handle max iterations reached
        if not is_done:
            final_answer = self._generate_partial_answer(context)
        
        return AgentResult(
            agent_id=self.id,
            output=final_answer,
            intermediate_steps=intermediate_steps,
            final_state=self.state,
            metadata={"architecture": "bdi", "iterations": iterations}
        )
    
    def _initialize_context(self, task: str, **kwargs) -> Dict[str, Any]:
        """Initialize the BDI context with task and tools."""
        context = {
            "task": task,
            "beliefs": [],
            "desires": [],
            "intentions": [],
            "selected_actions": [],
            "actions_results": [],
            "available_tools": [tool.name for tool in self.tools],
            "tool_descriptions": {tool.name: tool.description for tool in self.tools},
        }
        
        # Add any additional context
        if "context" in kwargs and isinstance(kwargs["context"], dict):
            for key, value in kwargs["context"].items():
                if key not in context:
                    context[key] = value
        
        self._update_state(
            stage="initializing",
            context_update={"task": task},
            working_memory=context
        )
        
        return context
    
    def _update_beliefs(self, context: Dict[str, Any]) -> str:
        """Update the agent's beliefs based on the current context."""
        system_prompt = get_prompt_for_architecture(
            architecture="bdi",
            prompt_type="system",
            agent_name=self.config.name,
            agent_description=self.config.description,
            available_actions=", ".join([tool.name for tool in self.tools])
        )
        
        user_prompt = f"Task: {context['task']}\n\n"
        
        # Add context history
        if context["beliefs"]:
            user_prompt += f"Previous beliefs:\n{context['beliefs'][-1]}\n\n"
        
        if context["actions_results"]:
            user_prompt += f"Recent action results:\n{context['actions_results'][-1]}\n\n"
        
        user_prompt += "Update your beliefs based on the task and previous information. What do you know or believe about the current situation?"
        
        # Get response from LLM
        _, response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        return self._extract_section(response, "beliefs")
    
    def _generate_desires(self, context: Dict[str, Any]) -> str:
        """Generate desires based on the current beliefs."""
        system_prompt = get_prompt_for_architecture(
            architecture="bdi",
            prompt_type="system",
            agent_name=self.config.name,
            agent_description=self.config.description,
            available_actions=", ".join([tool.name for tool in self.tools])
        )
        
        user_prompt = f"Task: {context['task']}\n\n"
        
        if context["beliefs"]:
            user_prompt += f"Current beliefs:\n{context['beliefs'][-1]}\n\n"
        
        user_prompt += "Based on your current beliefs and the task, generate desires (goals). What do you want to achieve?"
        
        _, response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        return self._extract_section(response, "desires")
    
    def _filter_to_intentions(self, context: Dict[str, Any]) -> str:
        """Filter desires to intentions (specific plans)."""
        system_prompt = get_prompt_for_architecture(
            architecture="bdi",
            prompt_type="system",
            agent_name=self.config.name,
            agent_description=self.config.description,
            available_actions=", ".join([tool.name for tool in self.tools])
        )
        
        user_prompt = f"Task: {context['task']}\n\n"
        
        if context["beliefs"]:
            user_prompt += f"Current beliefs:\n{context['beliefs'][-1]}\n\n"
        
        if context["desires"]:
            user_prompt += f"Current desires:\n{context['desires'][-1]}\n\n"
        
        user_prompt += "Based on your beliefs and desires, what specific intentions do you commit to?"
        
        _, response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        return self._extract_section(response, "intentions")
    
    def _select_actions(self, context: Dict[str, Any]) -> str:
        """Select actions based on intentions."""
        system_prompt = get_prompt_for_architecture(
            architecture="bdi",
            prompt_type="system",
            agent_name=self.config.name,
            agent_description=self.config.description,
            available_actions=", ".join([tool.name for tool in self.tools])
        )
        
        user_prompt = f"Task: {context['task']}\n\n"
        
        if context["beliefs"]:
            user_prompt += f"Current beliefs:\n{context['beliefs'][-1]}\n\n"
        
        if context["intentions"]:
            user_prompt += f"Current intentions:\n{context['intentions'][-1]}\n\n"
        
        # Add tool information
        user_prompt += "Available tools:\n"
        for tool_name, tool_desc in context["tool_descriptions"].items():
            user_prompt += f"- {tool_name}: {tool_desc}\n"
        
        user_prompt += "\nBased on your intentions, which tool will you use and with what parameters?"
        
        _, response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        return self._extract_section(response, "actions")
    
    def _execute_actions(self, context: Dict[str, Any]) -> str:
        """Execute the selected actions."""
        actions = context["selected_actions"][-1] if context["selected_actions"] else ""
        
        # Extract tool usage from the actions
        tool_pattern = r"(?:use\s+)?(\w+)(?::|,|\s+with|,?\s+)?\s+(.*)"
        match = re.search(tool_pattern, actions, re.IGNORECASE)
        
        if match:
            tool_name = match.group(1).strip().lower()
            parameter = match.group(2).strip()
            
            # Find the tool (case-insensitive)
            tool = next((t for t in self.tools if t.name.lower() == tool_name), None)
            
            if tool:
                try:
                    result = tool.execute({"query": parameter})
                    return f"Action result: Used {tool.name} with parameter '{parameter}' and got: {json.dumps(result, indent=2)}"
                except Exception as e:
                    return f"Error executing tool {tool.name}: {str(e)}"
            else:
                return f"Could not find tool '{tool_name}'."
        else:
            return f"Actions performed: {actions} (no specific tool used)"
    
    def _check_completion(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if the task is complete."""
        system_prompt = "Determine if the agent has completed its task."
        
        user_prompt = f"Task: {context['task']}\n\n"
        
        if context["beliefs"]:
            user_prompt += f"Current beliefs:\n{context['beliefs'][-1]}\n\n"
        
        if context["actions_results"]:
            user_prompt += f"Latest action results:\n{context['actions_results'][-1]}\n\n"
        
        user_prompt += "Has the task been completed? If yes, provide a final answer."
        
        _, response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.5,
            max_tokens=500
        )
        
        is_complete = "yes" in response.lower()[:100] or "complete" in response.lower()[:100]
        
        # Extract final answer if complete
        final_answer = response.strip()
        if is_complete:
            final_answer_match = re.search(r"final answer:?(.*)", final_answer, re.IGNORECASE | re.DOTALL)
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()
        
        return is_complete, final_answer
    
    def _generate_partial_answer(self, context: Dict[str, Any]) -> str:
        """Generate a partial answer when max iterations are reached."""
        system_prompt = "Provide a partial answer based on information gathered so far."
        
        user_prompt = f"Task: {context['task']}\n\n"
        
        if context["beliefs"]:
            user_prompt += f"Current beliefs:\n{context['beliefs'][-1]}\n\n"
        
        if context["actions_results"]:
            user_prompt += f"Latest action results:\n{context['actions_results'][-1]}\n\n"
        
        user_prompt += "Please provide a partial answer based on the information gathered so far."
        
        _, response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=500
        )
        
        return "Task not completed within maximum iterations. " + response
    
    def _extract_section(self, response: str, section_type: str) -> str:
        """Helper method to extract sections from LLM responses."""
        parser = create_parser("bdi")
        parsed = parser.parse(response)
        
        # Try to extract from parsed response
        section_content = ""
        if "cycles" in parsed and parsed["cycles"]:
            for cycle in parsed["cycles"]:
                if section_type in cycle:
                    section_content = cycle[section_type]
                    break
        
        # If not found, try regex extraction
        if not section_content and "raw_response" in parsed:
            patterns = {
                "beliefs": r"Beliefs?:\s*(.*?)(?:Desire|Intention|Action|$)",
                "desires": r"Desires?:\s*(.*?)(?:Intention|Action|$)",
                "intentions": r"Intentions?:\s*(.*?)(?:Act|Execution|$)",
                "actions": r"Actions?:\s*(.*?)(?:$)"
            }
            
            if section_type in patterns:
                match = re.search(patterns[section_type], parsed["raw_response"], re.DOTALL)
                if match:
                    section_content = match.group(1).strip()
                else:
                    section_content = parsed["raw_response"]
        
        return section_content or response