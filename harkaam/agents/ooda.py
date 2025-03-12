"""
OODA (Observe, Orient, Decide, Act) agent implementation.

The OODA loop is a decision cycle developed by military strategist John Boyd.
It consists of four stages: Observe, Orient, Decide, and Act.
"""

from typing import Any, Dict, List, Tuple
import re
import json

from harkaam.agents.base import BaseAgent, AgentResult
from harkaam.core.tools import Tool, ToolRegistry
from harkaam.core.llm import create_llm
from harkaam.core.prompt import get_prompt_for_architecture
from harkaam.core.parser import create_parser

class OODAAgent(BaseAgent):
    """
    OODA agent implementation.
    
    The OODA architecture follows a four-stage decision cycle:
    1. Observe: Gather information from the environment
    2. Orient: Analyze the information and form a mental model
    3. Decide: Make a decision based on the mental model
    4. Act: Execute the decision
    """
    
    def __init__(self, **kwargs):
        """Initialize a new OODA agent."""
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
        Execute a task using the OODA loop:
        1. Observe: Gather information about the task and environment
        2. Orient: Analyze the information and form a mental model
        3. Decide: Make a decision based on the mental model
        4. Act: Execute the decision
        5. Repeat until task is complete or max iterations reached
        """
        # Initialize context and tracking
        context = self._initialize_context(task, **kwargs)
        iterations = 0
        is_done = False
        final_answer = ""
        intermediate_steps = []
        
        # Main OODA loop
        while not is_done and iterations < self.max_iterations:
            iterations += 1
            
            # Step 1: Observe
            observation = self._process_stage(context, "observation", 
                "Please observe the current situation. What information can you gather about the task?")
            context["observations"].append(observation)
            intermediate_steps.append({"type": "observation", "content": observation})
            
            # Step 2: Orient
            orientation = self._process_stage(context, "orientation", 
                "Based on your observation, analyze the information and form a mental model of the situation.")
            context["orientations"].append(orientation)
            intermediate_steps.append({"type": "orientation", "content": orientation})
            
            # Step 3: Decide
            decide_prompt = "Based on your orientation, make a decision. What action will you take to accomplish your task?"
            if context["available_tools"]:
                decide_prompt += " Which tool will you use, if any?\n\nAvailable tools:\n"
                for tool_name, tool_desc in context["tool_descriptions"].items():
                    decide_prompt += f"- {tool_name}: {tool_desc}\n"
            
            decision = self._process_stage(context, "decision", decide_prompt)
            context["decisions"].append(decision)
            intermediate_steps.append({"type": "decision", "content": decision})
            
            # Step 4: Act
            action_result = self._execute_action(context)
            context["actions"].append(action_result)
            intermediate_steps.append({"type": "action", "content": action_result})
            
            # Check if task is complete
            is_done, final_answer = self._check_completion(context)
            
            # Update state with all stages of this OODA cycle
            self._update_state(
                stage=f"iteration_{iterations}",
                step_count=iterations,
                add_to_history={
                    "observe": observation,
                    "orient": orientation,
                    "decide": decision,
                    "act": action_result
                }
            )
        
        # Handle max iterations reached
        if not is_done:
            final_answer = "Task not completed within maximum iterations. " + self._generate_partial_answer(context)
        
        # Create result
        return AgentResult(
            agent_id=self.id,
            output=final_answer,
            intermediate_steps=intermediate_steps,
            final_state=self.state,
            metadata={"architecture": "ooda", "iterations": iterations}
        )
    
    def _initialize_context(self, task: str, **kwargs) -> Dict[str, Any]:
        """Initialize the context for the OODA agent."""
        # Create initial context
        context = {
            "task": task,
            "observations": [],
            "orientations": [],
            "decisions": [],
            "actions": [],
            "available_tools": [tool.name for tool in self.tools],
            "tool_descriptions": {tool.name: tool.description for tool in self.tools},
        }
        
        # Add any additional context from kwargs
        if "context" in kwargs and isinstance(kwargs["context"], dict):
            for key, value in kwargs["context"].items():
                if key not in context:
                    context[key] = value
        
        # Update agent state
        self._update_state(
            stage="initializing",
            context_update={"task": task},
            working_memory=context
        )
        
        return context
    
    def _process_stage(self, context: Dict[str, Any], stage: str, prompt_addition: str) -> str:
        """
        Process a stage of the OODA loop.
        
        Args:
            context: The current context
            stage: The stage name (observation, orientation, decision)
            prompt_addition: The prompt for this stage
            
        Returns:
            The processed stage content
        """
        # Get system prompt
        system_prompt = get_prompt_for_architecture(
            architecture="ooda",
            prompt_type="system",
            agent_name=self.config.name,
            agent_description=self.config.description,
            available_actions=", ".join([tool.name for tool in self.tools])
        )
        
        # Create user prompt
        user_prompt = f"Task: {context['task']}\n\n"
        
        # Add relevant context based on stage
        if stage == "observation":
            if context["observations"]:
                user_prompt += "Previous observations:\n"
                for obs in context["observations"][-2:]:  # Last 2 observations
                    user_prompt += f"- {obs}\n\n"
            if context["actions"]:
                user_prompt += "Previous actions:\n"
                for action in context["actions"][-2:]:  # Last 2 actions
                    user_prompt += f"- {action}\n\n"
        elif stage == "orientation":
            if context["observations"]:
                user_prompt += f"Current observation:\n{context['observations'][-1]}\n\n"
        elif stage == "decision":
            if context["observations"]:
                user_prompt += f"Current observation:\n{context['observations'][-1]}\n\n"
            if context["orientations"]:
                user_prompt += f"Current orientation:\n{context['orientations'][-1]}\n\n"
        
        # Add prompt addition
        user_prompt += prompt_addition
        
        # Get response from LLM
        _, response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Extract content using parser
        extracted_content = self._extract_content(response, stage)
        
        return extracted_content
    
    def _extract_content(self, response: str, stage: str) -> str:
        """
        Extract content from LLM response for a specific stage.
        
        Args:
            response: The LLM response
            stage: The stage name
            
        Returns:
            The extracted content
        """
        # Try using the parser first
        parser = create_parser("ooda")
        parsed = parser.parse(response)
        
        content = ""
        
        # Check if parser successful
        if "loops" in parsed and parsed["loops"]:
            for loop in parsed["loops"]:
                if stage in loop:
                    content = loop[stage]
                    break
        
        # If parser fails, try regex
        if not content and "raw_response" in parsed:
            stage_patterns = {
                "observation": r"Observation(?:s)?:\s*(.*?)(?:Orientation:|$)",
                "orientation": r"Orientation(?:s)?:\s*(.*?)(?:Decision:|$)",
                "decision": r"Decision(?:s)?:\s*(.*?)(?:Action:|$)",
                "action": r"Action(?:s)?:\s*(.*?)(?:$)"
            }
            
            if stage in stage_patterns:
                match = re.search(stage_patterns[stage], parsed["raw_response"], re.DOTALL)
                if match:
                    content = match.group(1).strip()
        
        # If all else fails, use the full response
        if not content:
            content = response
        
        return content
    
    def _execute_action(self, context: Dict[str, Any]) -> str:
        """
        Execute the latest decision by taking an action.
        """
        # Get the latest decision
        decision = context["decisions"][-1] if context["decisions"] else ""
        
        # Extract tool usage from the decision
        tool_pattern = r"(?:use\s+)?(\w+)(?::|,|\s+with|,?\s+)?\s+(.*)"
        match = re.search(tool_pattern, decision, re.IGNORECASE)
        
        if match:
            tool_name = match.group(1).strip().lower()
            parameter = match.group(2).strip()
            
            # Find the tool (case-insensitive)
            tool = next((t for t in self.tools if t.name.lower() == tool_name), None)
            
            if tool:
                try:
                    # Execute the tool
                    result = tool.execute({"query": parameter})
                    return f"Action result: Used {tool.name} with parameter '{parameter}' and got: {json.dumps(result, indent=2)}"
                except Exception as e:
                    return f"Error executing tool {tool.name}: {str(e)}"
            else:
                return f"Could not find tool '{tool_name}'. Available tools: {', '.join(context['available_tools'])}"
        else:
            # If no explicit tool usage is found, interpret the decision as an action
            return f"Action taken based on decision: {decision}"
    
    def _check_completion(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if the task is complete.
        """
        # Create user prompt with latest OODA cycle
        user_prompt = f"Task: {context['task']}\n\n"
        
        if context["observations"]:
            user_prompt += f"Latest observation:\n{context['observations'][-1]}\n\n"
        
        if context["orientations"]:
            user_prompt += f"Latest orientation:\n{context['orientations'][-1]}\n\n"
        
        if context["decisions"]:
            user_prompt += f"Latest decision:\n{context['decisions'][-1]}\n\n"
        
        if context["actions"]:
            user_prompt += f"Latest action result:\n{context['actions'][-1]}\n\n"
        
        user_prompt += "Based on the above information, has the task been completed? If yes, provide a final answer. If no, explain what's still needed."
        
        # Get response from LLM
        _, response = self.llm_client.generate(
            system_prompt="Determine if the agent has completed its task and can provide a final answer.",
            user_prompt=user_prompt,
            temperature=0.5,
            max_tokens=500
        )
        
        # Check if complete
        is_complete = "yes" in response.lower()[:100] or "complete" in response.lower()[:100]
        
        # Clean up final answer
        final_answer = response.strip()
        if is_complete:
            match = re.search(r"final answer:?(.*)", final_answer, re.IGNORECASE | re.DOTALL)
            if match:
                final_answer = match.group(1).strip()
        
        return is_complete, final_answer
    
    def _generate_partial_answer(self, context: Dict[str, Any]) -> str:
        """
        Generate a partial answer when max iterations are reached.
        """
        user_prompt = f"Task: {context['task']}\n\n"
        
        if context["observations"]:
            user_prompt += f"Latest observation:\n{context['observations'][-1]}\n\n"
        
        if context["orientations"]:
            user_prompt += f"Latest orientation:\n{context['orientations'][-1]}\n\n"
        
        if context["actions"]:
            user_prompt += f"Latest action result:\n{context['actions'][-1]}\n\n"
        
        user_prompt += "Please provide a partial answer based on the information gathered so far."
        
        # Get response from LLM
        _, response = self.llm_client.generate(
            system_prompt="Based on the information gathered so far, provide a partial answer to the task.",
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=500
        )
        
        return response