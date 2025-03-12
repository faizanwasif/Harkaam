"""
RAISE (Reasoning and Acting Through Scratch Pad and Examples) agent implementation.

RAISE is an agent architecture that uses a scratch pad for reasoning and
examples for guiding the reasoning process.
"""

from typing import Any, Dict, List, Tuple
import re
import json

from harkaam.agents.base import BaseAgent, AgentResult
from harkaam.core.tools import Tool, ToolRegistry
from harkaam.core.llm import create_llm
from harkaam.core.prompt import get_prompt_for_architecture

class RAISEAgent(BaseAgent):
    """
    RAISE agent implementation.
    
    The RAISE architecture leverages a scratch pad for reasoning and examples
    for guiding the reasoning process. The agent works through problems step by
    step, writing its intermediate reasoning on the scratch pad.
    """
    
    def __init__(self, **kwargs):
        """Initialize a new RAISE agent."""
        super().__init__(**kwargs)
        
        # Initialize core components
        self.llm_client = create_llm(self.config.llm)
        self.examples = kwargs.get("examples", [])
        self.max_iterations = kwargs.get("max_iterations", 10)
        self.tool_registry = ToolRegistry()
        
        # Register tools
        self.tools = kwargs.get("tools", [])
        for tool in self.tools:
            self.tool_registry.register(tool)
    
    def execute(self, task: str, **kwargs) -> AgentResult:
        """
        Execute a task using the RAISE architecture:
        1. Create initial scratch pad with task and context
        2. Retrieve relevant examples
        3. Generate thoughts and record in scratch pad
        4. Use tools when needed and record results
        5. Update scratch pad with new observations
        6. Generate a response based on the scratch pad
        7. Repeat until task is complete
        """
        # Initialize components
        context = self._initialize_context(task, **kwargs)
        scratch_pad = self._initialize_scratch_pad(context)
        context["scratch_pad"] = scratch_pad
        
        # Initialize tracking variables
        iterations = 0
        is_done = False
        final_answer = ""
        intermediate_steps = []
        
        # Main RAISE loop
        while not is_done and iterations < self.max_iterations:
            iterations += 1
            
            # Retrieve examples and update scratch pad
            examples = self._generate_llm_step(context, 
                "Retrieve and explain examples that are relevant to the current task.", 
                "examples")
            scratch_pad = self._update_scratch_pad(scratch_pad, "Examples", examples)
            context["scratch_pad"] = scratch_pad
            intermediate_steps.append({"type": "examples", "content": examples})
            
            # Generate thoughts and update scratch pad
            thoughts = self._generate_llm_step(context, 
                "Generate thoughts about how to approach this task. What steps should be taken?", 
                "thoughts")
            scratch_pad = self._update_scratch_pad(scratch_pad, "Thoughts", thoughts)
            context["scratch_pad"] = scratch_pad
            intermediate_steps.append({"type": "thoughts", "content": thoughts})
            
            # Determine if tools should be used
            should_use_tools = self._should_use_tools(context)
            
            if should_use_tools:
                # Use tools and update scratch pad
                tool_results = self._use_tools(context)
                scratch_pad = self._update_scratch_pad(scratch_pad, "Tool Results", tool_results)
                context["tool_results"] = tool_results
                context["scratch_pad"] = scratch_pad
                intermediate_steps.append({"type": "tool_results", "content": tool_results})
                
                # Get observations based on tool results
                observations = self._generate_llm_step(context, 
                    f"Tool Results:\n{tool_results}\n\nBased on these tool results, what new insights have we gained?", 
                    "observations")
                scratch_pad = self._update_scratch_pad(scratch_pad, "Observations", observations)
                context["scratch_pad"] = scratch_pad
                intermediate_steps.append({"type": "observations", "content": observations})
            
            # Edit working memory (scratch pad)
            updated_pad = self._generate_llm_step(context, 
                "Review and edit the scratch pad to reflect the current state of the task.", 
                "working_memory")
                
            # Only use full response if it looks like a complete scratch pad
            if updated_pad.strip().startswith("#") or "Scratch Pad" in updated_pad:
                scratch_pad = updated_pad
            else:
                # Otherwise just add it as a progress summary
                scratch_pad = self._update_scratch_pad(scratch_pad, "Progress Summary", updated_pad)
                
            context["scratch_pad"] = scratch_pad
            
            # Check if task is complete
            is_done, final_answer = self._check_completion(context)
            
            # Update state
            self._update_state(
                stage=f"iteration_{iterations}",
                step_count=iterations,
                add_to_history={
                    "scratch_pad": scratch_pad,
                    "examples": examples,
                    "thoughts": thoughts,
                    "tool_results": context.get("tool_results", "No tools used in this iteration")
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
            metadata={"architecture": "raise", "iterations": iterations, "scratch_pad": scratch_pad}
        )
    
    def _initialize_context(self, task: str, **kwargs) -> Dict[str, Any]:
        """Initialize the context for the RAISE agent."""
        # Create initial context
        context = {
            "task": task,
            "available_tools": [tool.name for tool in self.tools],
            "tool_descriptions": {tool.name: tool.description for tool in self.tools},
            "examples": self.examples,
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
    
    def _initialize_scratch_pad(self, context: Dict[str, Any]) -> str:
        """Initialize the scratch pad with the task and context."""
        response = self._generate_llm_step(context, 
            "Initialize a scratch pad for this task. The scratch pad will be used to record your thinking process.", 
            "scratch_pad_init")
        
        # If response doesn't look like a proper scratch pad, create a basic one
        if not "Scratch Pad" in response and not response.strip().startswith("#"):
            scratch_pad = f"# Scratch Pad for: {context['task']}\n\n"
            scratch_pad += "## Initial Context\n"
            scratch_pad += f"Task: {context['task']}\n\n"
            
            # Add available tools
            if context["available_tools"]:
                scratch_pad += "\n## Available Tools\n"
                for tool_name, tool_desc in context["tool_descriptions"].items():
                    scratch_pad += f"- {tool_name}: {tool_desc}\n"
                    
            return scratch_pad
        
        return response
    
    def _generate_llm_step(self, context: Dict[str, Any], prompt_addition: str, step_name: str) -> str:
        """Generate a response from the LLM for a specific step in the RAISE process."""
        # Get system prompt
        system_prompt = get_prompt_for_architecture(
            architecture="raise",
            prompt_type="system",
            agent_name=self.config.name,
            agent_description=self.config.description
        )
        
        # Create user prompt
        user_prompt = f"Task: {context['task']}\n\n"
        
        # Add scratch pad if available
        if "scratch_pad" in context:
            user_prompt += f"Scratch Pad:\n{context['scratch_pad']}\n\n"
        
        # Add tool information for relevant steps
        if step_name in ["thoughts", "working_memory"] and context["available_tools"]:
            user_prompt += "Available tools:\n"
            for tool_name, tool_desc in context["tool_descriptions"].items():
                user_prompt += f"- {tool_name}: {tool_desc}\n"
            user_prompt += "\n"
            
        # Add examples for the examples step
        if step_name == "examples" and self.examples:
            user_prompt += "Available examples:\n"
            for i, example in enumerate(self.examples):
                user_prompt += f"Example {i+1}: {example}\n\n"
        
        # Add prompt addition
        user_prompt += prompt_addition
        
        # Get response from LLM
        _, response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Update state for the specific step
        self._update_state(stage=f"{step_name}", add_to_history={step_name: response})
        
        return response
    
    def _update_scratch_pad(self, scratch_pad: str, section: str, content: str) -> str:
        """Update the scratch pad with new content in a specific section."""
        # Check if section already exists
        section_pattern = re.compile(f"## {section}\n", re.IGNORECASE)
        
        if section_pattern.search(scratch_pad):
            # Section exists - find it and update
            parts = scratch_pad.split(f"## {section}\n", 1)
            section_content = parts[1].split("\n##", 1)
            
            if len(section_content) > 1:
                # There's another section after this one
                updated_section = f"{content}\n\n##"
                updated_scratch_pad = f"{parts[0]}## {section}\n{updated_section}{section_content[1]}"
            else:
                # This is the last section
                updated_section = f"{content}\n\n"
                updated_scratch_pad = f"{parts[0]}## {section}\n{updated_section}"
        else:
            # Section doesn't exist - add it at the end
            updated_scratch_pad = f"{scratch_pad}\n\n## {section}\n{content}\n"
        
        return updated_scratch_pad
    
    def _should_use_tools(self, context: Dict[str, Any]) -> bool:
        """Determine if tools should be used based on the current context."""
        # If no tools are available, don't use tools
        if not self.tools:
            return False
        
        # Ask LLM if tools should be used
        user_prompt = f"Task: {context['task']}\n\n"
        user_prompt += f"Scratch Pad:\n{context['scratch_pad']}\n\n"
        user_prompt += "Available tools:\n"
        
        for tool_name, tool_desc in context["tool_descriptions"].items():
            user_prompt += f"- {tool_name}: {tool_desc}\n"
        
        user_prompt += "\nBased on the current state of the task, should any tools be used at this point? Answer Yes or No."
        
        # Get response with lower temperature for more deterministic answer
        _, response = self.llm_client.generate(
            system_prompt="Determine if tools should be used in the current step.",
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=100
        )
        
        # Check if tools should be used
        return "yes" in response.lower()[:100]
    
    def _use_tools(self, context: Dict[str, Any]) -> str:
        """Use tools based on the current context and scratch pad."""
        # Ask LLM which tool to use
        user_prompt = f"Task: {context['task']}\n\n"
        user_prompt += f"Scratch Pad:\n{context['scratch_pad']}\n\n"
        user_prompt += "Available tools:\n"
        
        for tool_name, tool_desc in context["tool_descriptions"].items():
            user_prompt += f"- {tool_name}: {tool_desc}\n"
        
        user_prompt += "\nSelect a tool to use and specify the parameters. Use the format 'TOOL_NAME: PARAMETERS'."
        
        _, response = self.llm_client.generate(
            system_prompt=get_prompt_for_architecture(
                architecture="raise",
                prompt_type="system",
                agent_name=self.config.name,
                agent_description=self.config.description
            ),
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Extract tool usage
        tool_pattern = r"(?:use\s+)?(\w+)(?::|,|\s+with|,?\s+)?\s+(.*)"
        match = re.search(tool_pattern, response, re.IGNORECASE)
        
        if match:
            # Found tool usage pattern
            tool_name = match.group(1).strip().lower()
            parameter = match.group(2).strip()
            
            # Find the tool (case-insensitive)
            tool = next((t for t in self.tools if t.name.lower() == tool_name), None)
            
            if tool:
                try:
                    # Execute the tool
                    result = tool.execute({"query": parameter})
                    return f"Used {tool.name} with parameter '{parameter}' and got: {json.dumps(result, indent=2)}"
                except Exception as e:
                    return f"Error executing tool {tool.name}: {str(e)}"
            else:
                return f"Could not find tool '{tool_name}'. Available tools: {', '.join(context['available_tools'])}"
        else:
            # No explicit tool usage found
            return f"Unclear tool usage in response: {response}"
    
    def _check_completion(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if the task is complete based on the scratch pad."""
        # Ask LLM if task is complete
        user_prompt = f"Task: {context['task']}\n\n"
        user_prompt += f"Scratch Pad:\n{context['scratch_pad']}\n\n"
        user_prompt += "Based on the scratch pad, has the task been completed? If yes, provide a final answer. If no, explain what's still needed."
        
        # Get response
        _, response = self.llm_client.generate(
            system_prompt="Determine if the task has been completed based on the scratch pad.",
            user_prompt=user_prompt,
            temperature=0.5,
            max_tokens=500
        )
        
        # Check if task is complete
        is_complete = "yes" in response.lower()[:100] or "complete" in response.lower()[:100]
        
        # Extract final answer
        final_answer = response.strip()
        
        if is_complete:
            # Try to extract just the answer content
            final_answer_match = re.search(r"final answer:?(.*)", final_answer, re.IGNORECASE | re.DOTALL)
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()
        
        return is_complete, final_answer
    
    def _generate_partial_answer(self, context: Dict[str, Any]) -> str:
        """Generate a partial answer when max iterations are reached."""
        # Ask LLM for partial answer
        user_prompt = f"Task: {context['task']}\n\n"
        user_prompt += f"Scratch Pad:\n{context['scratch_pad']}\n\n"
        user_prompt += "The maximum number of iterations has been reached. Based on the scratch pad, provide a partial answer to the task."
        
        # Get response
        _, response = self.llm_client.generate(
            system_prompt="Based on the scratch pad, provide a partial answer to the task.",
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=500
        )
        
        return response