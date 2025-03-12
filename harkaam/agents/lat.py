"""
LAT (Language Agent Tree Search) agent implementation.

LAT is an agent architecture that uses tree search algorithms with language
models to explore possible solution paths for complex tasks.
"""

from typing import Any, Dict, List, Optional
import re

from harkaam.agents.base import BaseAgent, AgentResult
from harkaam.core.tools import Tool, ToolRegistry
from harkaam.core.llm import create_llm
from harkaam.core.prompt import get_prompt_for_architecture

class LATAgent(BaseAgent):
    """
    LAT agent implementation.
    
    The LAT architecture leverages tree search algorithms with language models
    to explore and evaluate multiple reasoning paths.
    """
    
    def __init__(self, **kwargs):
        """Initialize a new LAT agent."""
        super().__init__(**kwargs)
        
        # Initialize LLM client
        self.llm_client = create_llm(self.config.llm)
        
        # LAT-specific configuration
        self.max_depth = kwargs.get("max_depth", 5)
        self.max_branches = kwargs.get("max_branches", 3)
        self.search_strategy = kwargs.get("search_strategy", "best_first")
        
        # Register tools
        self.tools = kwargs.get("tools", [])
        self.tool_registry = ToolRegistry()
        for tool in self.tools:
            self.tool_registry.register(tool)
    
    def execute(self, task: str, **kwargs) -> AgentResult:
        """
        Execute a task using the LAT architecture:
        1. Initialize the model with the task
        2. Create the initial decision tree
        3. Explore the tree by selecting nodes, simulating outcomes
        4. Select the best path and generate output
        """
        # Initialize context and tracking
        context = self._initialize_context(task, **kwargs)
        intermediate_steps = []
        
        # Create the decision tree
        decision_tree = self._generate_llm_response(context, 
            "Create a decision tree to approach this task. Identify key decision points and branches to explore.",
            "decision_tree_creation")
        context["decision_tree"] = decision_tree
        intermediate_steps.append({"type": "decision_tree_creation", "content": decision_tree})
        
        # Main search loop
        current_path = []
        current_depth = 0
        
        while current_depth < self.max_depth:
            # Select next node to explore
            node_prompt = f"Decision Tree:\n{context['decision_tree']}\n\n"
            if current_path:
                node_prompt += "Current path:\n" + "\n".join([f"{i+1}. {node}" for i, node in enumerate(current_path)]) + "\n\n"
            node_prompt += "Select the next node to explore in the decision tree."
            
            selected_node = self._generate_llm_response(context, node_prompt, "node_selection")
            context["selected_node"] = selected_node
            intermediate_steps.append({"type": "node_selection", "depth": current_depth, "content": selected_node})
            
            # Check if simulation is needed
            need_simulation = self._need_simulation(context, selected_node)
            
            if need_simulation:
                # Simulate possible outcomes
                simulation_results = self._generate_llm_response(context,
                    f"Selected node: {selected_node}\n\nSimulate the possible outcomes of this node.",
                    "simulation")
                context["simulation_results"] = simulation_results
                intermediate_steps.append({"type": "simulation", "depth": current_depth, "content": simulation_results})
                
                # Process simulation results (backpropagate and reflect)
                reflection = self._process_simulation(context, selected_node, simulation_results)
                context["reflection"] = reflection
                intermediate_steps.append({"type": "reflection", "depth": current_depth, "content": reflection})
            
            # Update current path
            current_path.append(selected_node)
            current_depth += 1
            
            # Check if we've reached a terminal state
            if self._is_terminal_state(context, selected_node):
                break
        
        # Select the best path
        best_path = self._generate_llm_response(context,
            "Current path:\n" + "\n".join([f"{i+1}. {node}" for i, node in enumerate(current_path)]) + 
            "\n\nBased on your search, select the best path to solve the task.",
            "best_path_selection")
        context["best_path"] = best_path
        intermediate_steps.append({"type": "best_path_selection", "content": best_path})
        
        # Generate the final output
        output = self._generate_llm_response(context,
            f"Best path: {best_path}\n\nGenerate the final output that completes the given task.",
            "output_generation")
        intermediate_steps.append({"type": "output_generation", "content": output})
        
        # Create result
        return AgentResult(
            agent_id=self.id,
            output=output,
            intermediate_steps=intermediate_steps,
            final_state=self.state,
            metadata={
                "architecture": "lat",
                "max_depth": self.max_depth,
                "search_strategy": self.search_strategy,
                "path_length": len(current_path)
            }
        )
    
    def _initialize_context(self, task: str, **kwargs) -> Dict[str, Any]:
        """Initialize the context for the LAT agent."""
        context = {
            "task": task,
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
    
    def _generate_llm_response(self, context: Dict[str, Any], prompt_addition: str, stage: str) -> str:
        """Generate a response from the LLM with unified prompt handling."""
        # Get system prompt
        system_prompt = get_prompt_for_architecture(
            architecture="lat",
            prompt_type="system",
            agent_name=self.config.name,
            agent_description=self.config.description
        )
        
        # Create user prompt
        user_prompt = f"Task: {context['task']}\n\n"
        
        # Add tool information if relevant
        if stage in ["decision_tree_creation", "simulation"] and context["available_tools"]:
            user_prompt += "Available tools:\n"
            for tool_name, tool_desc in context["tool_descriptions"].items():
                user_prompt += f"- {tool_name}: {tool_desc}\n"
            user_prompt += "\n"
        
        # Add prompt addition
        user_prompt += prompt_addition
        
        # Get response from LLM
        _, response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Update state
        self._update_state(stage=f"{stage}", add_to_history={stage: response})
        
        return response
    
    def _need_simulation(self, context: Dict[str, Any], selected_node: str) -> bool:
        """Determine if simulation is needed for the selected node."""
        user_prompt = f"Task: {context['task']}\n\nSelected node: {selected_node}\n\n"
        user_prompt += "Is simulation needed for this node? Answer Yes or No and explain why."
        
        # Get response from LLM with lower temperature for deterministic answer
        _, response = self.llm_client.generate(
            system_prompt="Determine if simulation is needed for this node.",
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=200
        )
        
        # Check if response indicates simulation is needed
        return "yes" in response.lower()[:100]
    
    def _process_simulation(self, context: Dict[str, Any], selected_node: str, 
                            simulation_results: str) -> str:
        """Process simulation results - combines backpropagation and reflection."""
        user_prompt = f"Task: {context['task']}\n\nSelected node: {selected_node}\n\n"
        user_prompt += f"Simulation results: {simulation_results}\n\n"
        user_prompt += "1. How does this new information affect previous decisions?\n"
        user_prompt += "2. What have you learned from these results?\n"
        user_prompt += "3. How should your approach change based on this learning?"
        
        # Get response from LLM
        _, response = self.llm_client.generate(
            system_prompt=get_prompt_for_architecture(
                architecture="lat",
                prompt_type="system",
                agent_name=self.config.name,
                agent_description=self.config.description
            ),
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Update state - combined step for backpropagation and reflection
        self._update_state(stage="process_simulation", 
                          add_to_history={"process_simulation": response})
        
        return response
    
    def _is_terminal_state(self, context: Dict[str, Any], node: str) -> bool:
        """Check if the node is a terminal state in the decision tree."""
        user_prompt = f"Task: {context['task']}\n\nCurrent node: {node}\n\n"
        user_prompt += "Is this a terminal node (a leaf node or a node that completes the task)? Answer Yes or No."
        
        # Get response from LLM with lower temperature
        _, response = self.llm_client.generate(
            system_prompt="Determine if this node is a terminal state.",
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=100
        )
        
        # Check if response indicates a terminal state
        return "yes" in response.lower()[:100]