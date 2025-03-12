"""
ReAct (Reasoning and Acting) agent implementation.

ReAct is an agent architecture that interleaves reasoning and acting steps.
The agent alternates between reasoning about the current situation and taking
actions to make progress towards solving the task.
"""

from typing import Any, Dict, List, Optional, Tuple
import re
import json

from harkaam.agents.base import BaseAgent, AgentResult, AgentState
from harkaam.core.tools import Tool, ToolRegistry
from harkaam.core.llm import create_llm
from harkaam.core.prompt import get_prompt_for_architecture
from harkaam.core.parser import create_parser

class ReActAgent(BaseAgent):
    """
    ReAct agent implementation.
    
    The ReAct architecture interleaves reasoning and acting steps. The agent
    first reasons about the current situation, then decides on an action,
    executes the action, and observes the result.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize a new ReAct agent.
        
        Args:
            **kwargs: Arguments to pass to the base agent constructor
        """
        super().__init__(**kwargs)
        
        # Initialize LLM client
        provider, model = self.config.llm.split(":", 1)
        self.llm_client = create_llm(self.config.llm)
        
        # ReAct-specific configuration
        self.max_iterations = kwargs.get("max_iterations", 10)
        self.tool_registry = ToolRegistry()
        
        # Register tools
        self.tools = kwargs.get("tools", [])
        for tool in self.tools:
            self.tool_registry.register(tool)
    
    def execute(self, task: str, **kwargs) -> AgentResult:
        """
        Execute a task using the ReAct architecture.
        
        The ReAct execution loop consists of:
        1. Set up initial context
        2. Decide whether to think or act
        3. If thinking: Generate a thought and add to context
        4. If acting: Do an action, get observation, add to context
        5. Check if task is done
        6. Repeat until task is complete or max iterations reached
        
        Args:
            task: The task to execute
            **kwargs: Additional execution arguments
            
        Returns:
            The result of the execution
        """
        # Step 1: Set up initial context
        context = self._set_up_initial_context(task, **kwargs)
        
        # Initialize variables for tracking
        iterations = 0
        is_done = False
        final_answer = ""
        intermediate_steps = []
        
        # Main execution loop
        while not is_done and iterations < self.max_iterations:
            iterations += 1
            
            # Step 2: Decide whether to think or act
            # For ReAct, we allow the LLM to decide implicitly based on the prompt format
            next_step = self._get_next_step(context)
            
            # Step 3-4: Think or Act based on the decision
            if "thought" in next_step and next_step["thought"]:
                # Think: Generate a thought and add to context
                thought = next_step["thought"]
                context["thoughts"].append(thought)
                intermediate_steps.append({"type": "thought", "content": thought})
                
                # Update agent state
                self._update_state(
                    stage="thinking",
                    step_count=iterations,
                    add_to_history={"type": "thought", "content": thought}
                )
            
            if "action" in next_step and next_step["action"]:
                # Act: Do an action, get observation, add to context
                action = next_step["action"]
                observation = self._execute_action(action)
                
                context["actions"].append(action)
                context["observations"].append(observation)
                
                intermediate_steps.append({
                    "type": "action", 
                    "content": action,
                    "observation": observation
                })
                
                # Update agent state
                self._update_state(
                    stage="acting",
                    step_count=iterations,
                    add_to_history={"type": "action_observation", "action": action, "observation": observation}
                )
            
            # Step 5: Check if task is done
            if "final_answer" in next_step and next_step["final_answer"]:
                is_done = True
                final_answer = next_step["final_answer"]
            
        # Handle case where max iterations reached without completion
        if not is_done:
            final_answer = "Task not completed within maximum iterations. " + self._generate_partial_answer(context)
        
        # Update final state
        self._update_state(
            stage="completed",
            add_to_history={"type": "final_answer", "content": final_answer}
        )
        
        # Create result
        return AgentResult(
            agent_id=self.id,
            output=final_answer,
            intermediate_steps=intermediate_steps,
            final_state=self.state,
            metadata={"architecture": "react", "iterations": iterations}
        )
    
    def _set_up_initial_context(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Set up the initial context for the ReAct agent.
        
        Args:
            task: The task to execute
            **kwargs: Additional context
            
        Returns:
            The initial context
        """
        # Create initial context
        context = {
            "task": task,
            "thoughts": [],
            "actions": [],
            "observations": [],
            "available_tools": [tool.name for tool in self.tools],
            "tool_descriptions": {tool.name: tool.description for tool in self.tools},
        }
        
        # Add any additional context from kwargs
        if "context" in kwargs and isinstance(kwargs["context"], dict):
            for key, value in kwargs["context"].items():
                if key not in context:  # Don't overwrite existing keys
                    context[key] = value
        
        # Update agent state
        self._update_state(
            stage="initializing",
            context_update={"task": task},
            working_memory=context
        )
        
        return context
    
    def _get_next_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the next step in the ReAct process (think or act).
        
        Args:
            context: The current context
            
        Returns:
            A dictionary containing the next thought, action, or final answer
        """
        # Prepare system prompt
        system_prompt = get_prompt_for_architecture(
            architecture="react",
            prompt_type="system",
            agent_name=self.config.name,
            agent_description=self.config.description,
            available_actions=", ".join(["search", "use tool"] + [f"use {tool}" for tool in context["available_tools"]])
        )
        
        # Prepare user prompt with context
        user_prompt = self._prepare_user_prompt(context)
        
        # Generate response from LLM
        thinking, response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Parse the response
        parser = create_parser("react")
        parsed_response = parser.parse(response)
        
        # Extract the next step
        result = {}
        
        # Get the most recent thought-action pair if available
        if parsed_response["cycles"]:
            last_cycle = parsed_response["cycles"][-1]
            if "thought" in last_cycle:
                result["thought"] = last_cycle["thought"]
            if "action" in last_cycle:
                result["action"] = last_cycle["action"]
                
        # Check for final answer
        if parsed_response["final_answer"]:
            result["final_answer"] = parsed_response["final_answer"]
            
        # If no clear action or final answer, treat the whole response as a thought
        if not result:
            result["thought"] = response
            
        return result
    
    def _prepare_user_prompt(self, context: Dict[str, Any]) -> str:
        """
        Prepare the user prompt with the current context.
        
        Args:
            context: The current context
            
        Returns:
            The user prompt
        """
        # Basic task information
        prompt = f"Task: {context['task']}\n\n"
        
        # Add tool information
        if context["available_tools"]:
            prompt += "Available Tools:\n"
            for tool_name in context["available_tools"]:
                prompt += f"- {tool_name}: {context['tool_descriptions'].get(tool_name, '')}\n"
            prompt += "\n"
        
        # Add context history
        prompt += "Context History:\n"
        
        # Add thoughts, actions, and observations
        for i in range(max(len(context["thoughts"]), len(context["actions"]), len(context["observations"]))):
            # Add thought if available
            if i < len(context["thoughts"]):
                prompt += f"Thought: {context['thoughts'][i]}\n"
            
            # Add action and observation if available
            if i < len(context["actions"]) and i < len(context["observations"]):
                prompt += f"Action: {context['actions'][i]}\n"
                prompt += f"Observation: {context['observations'][i]}\n"
        
        # Add instruction for next step
        prompt += "\nContinue the reasoning process. Think about what to do next."
        prompt += "\nRemember to use the format: Thought: ... Action: ... Observation: ... Final Answer: ..."
        
        return prompt
    
    def _execute_action(self, action: str) -> str:
        """
        Execute an action and return the observation.
        
        Args:
            action: The action to execute
            
        Returns:
            The observation from executing the action
        """
        # Try to extract a tool call from the action
        tool_match = re.search(r"use (\w+)(?::|,|\s+with)?\s+(.*)", action, re.IGNORECASE)
        search_match = re.search(r"search(?::|,|\s+for)?\s+(.*)", action, re.IGNORECASE)
        
        try:
            # Handle explicit tool usage
            if tool_match:
                tool_name = tool_match.group(1).strip()
                tool_input = tool_match.group(2).strip()
                
                # Try to extract parameters as JSON
                try:
                    if tool_input.startswith("{") and tool_input.endswith("}"):
                        parameters = json.loads(tool_input)
                    else:
                        # For simple cases, assume the input is a single parameter
                        parameters = {"query": tool_input}
                except json.JSONDecodeError:
                    parameters = {"query": tool_input}
                
                # Execute the tool
                if tool_name.lower() in [t.name.lower() for t in self.tools]:
                    # Find the actual tool object (case-insensitive match)
                    tool = next((t for t in self.tools if t.name.lower() == tool_name.lower()), None)
                    result = tool.execute(parameters)
                    return f"Tool '{tool_name}' returned: {json.dumps(result, indent=2)}"
                else:
                    return f"Error: Tool '{tool_name}' not found. Available tools: {', '.join([t.name for t in self.tools])}"
            
            # Handle search
            elif search_match:
                search_query = search_match.group(1).strip()
                
                # Find a search tool if available
                search_tool = next((t for t in self.tools if t.name.lower() == "search"), None)
                if search_tool:
                    result = search_tool.execute({"query": search_query})
                    return f"Search results for '{search_query}': {json.dumps(result, indent=2)}"
                else:
                    return f"Error: No search tool available."
            
            # Generic action handling
            else:
                return f"Action '{action}' was taken, but no specific tool was utilized. Please use a tool if you need to retrieve information."
                
        except Exception as e:
            # Handle any errors during execution
            return f"Error executing action: {str(e)}"
    
    def _generate_partial_answer(self, context: Dict[str, Any]) -> str:
        """
        Generate a partial answer based on the current context when max iterations are reached.
        
        Args:
            context: The current context
            
        Returns:
            A partial answer
        """
        # Generate a system message asking for a partial answer
        system_prompt = "You are a helpful assistant. Based on the information gathered so far, provide a partial answer to the task."
        
        # Prepare a user prompt with the context
        user_prompt = f"Task: {context['task']}\n\n"
        user_prompt += "Information gathered so far:\n"
        
        # Add thoughts, actions, and observations
        for i in range(min(len(context["thoughts"]), len(context["actions"]), len(context["observations"]))):
            user_prompt += f"Thought: {context['thoughts'][i]}\n"
            user_prompt += f"Action: {context['actions'][i]}\n"
            user_prompt += f"Observation: {context['observations'][i]}\n\n"
        
        user_prompt += "\nPlease provide a partial answer based on the information gathered so far."
        
        # Generate response
        _, response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=500
        )
        
        return response