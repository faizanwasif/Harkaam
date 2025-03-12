"""
Prompt module for the Harkaam framework.

This module provides utilities for constructing and managing prompts
for different agent architectures.
"""

from typing import Any, Dict, List, Optional, Union
from string import Template

class PromptTemplate:
    """
    A template for constructing prompts.
    
    Prompt templates allow for dynamic construction of prompts
    based on variables provided at runtime.
    """
    
    def __init__(self, template: str):
        """
        Initialize a new prompt template.
        
        Args:
            template: The template string with placeholders
        """
        self.template = Template(template)
    
    def format(self, **kwargs) -> str:
        """
        Format the template with the provided variables.
        
        Args:
            **kwargs: Variables to substitute in the template
            
        Returns:
            The formatted prompt
        """
        return self.template.safe_substitute(**kwargs)

class PromptLibrary:
    """
    A library of predefined prompts for different agent architectures.
    """
    
    # ReAct prompts
    REACT_SYSTEM_PROMPT = PromptTemplate("""
You are ${agent_name}, ${agent_description}.

You will solve tasks step-by-step by thinking and acting in alternating steps.

The process follows this pattern:
1. Think about the current situation and decide what to do next
2. Act by using available tools to gather information
3. Observe the results of your action
4. Decide if the task is done or if you need to continue
5. Repeat until the task is complete

Follow this format strictly:
Thought: Your reasoning about the current situation and what you should do next.
Action: The action to take (use a tool or search). Example: "use search, What is the capital of France?"
Observation: The result of the action (will be provided after your action).
... (repeat Thought/Action/Observation as needed)
Final Answer: Your final response to the task once you have all needed information.

Available actions: ${available_actions}
""")
    
    REACT_USER_PROMPT = PromptTemplate("""
Task: ${task}

${context}

Work through this step-by-step, using the Thought/Action/Observation format.
Start by thinking about how to approach the problem, then take actions as needed.
""")
    
    # OODA prompts
    OODA_SYSTEM_PROMPT = PromptTemplate("""
You are ${agent_name}, ${agent_description}.

You will solve tasks using the OODA loop:
1. Observe: Gather information about the situation
2. Orient: Analyze the information and form a mental model
3. Decide: Make a decision based on your analysis
4. Act: Execute your decision

Follow this format:
Observation: What you observe about the situation.
Orientation: Your analysis and understanding of the situation.
Decision: What you decide to do based on your orientation.
Action: The action you take.
... (repeat steps as needed)
Final Answer: Your final response to the task.
""")
    
    OODA_USER_PROMPT = PromptTemplate("""
Task: ${task}

${context}

Please solve this using the OODA loop.
""")
    
    # BDI prompts
    BDI_SYSTEM_PROMPT = PromptTemplate("""
You are ${agent_name}, ${agent_description}.

You will solve tasks using the BDI framework:
1. Beliefs: What you know about the world
2. Desires: Your goals or objectives
3. Intentions: Your committed plans to achieve your desires

Follow this format:
Beliefs: Your current understanding of the situation.
Desires: What you want to achieve.
Intentions: Your plan to achieve your desires.
Execution: The actions you take to execute your plan.
... (update beliefs and repeat as needed)
Final Answer: Your final response to the task.
""")
    
    BDI_USER_PROMPT = PromptTemplate("""
Task: ${task}

${context}

Please solve this using the BDI framework.
""")
    
    # LAT prompts
    LAT_SYSTEM_PROMPT = PromptTemplate("""
You are ${agent_name}, ${agent_description}.

You will solve tasks using tree search to explore different solution paths:
1. Root: Start with the initial problem
2. Expand: Generate possible solution steps
3. Evaluate: Assess the promising paths
4. Select: Choose the best path to continue

Follow this format:
Problem: The current problem or subproblem.
Branches:
- Option 1: First possible step
  Evaluation: Assessment of this path
- Option 2: Second possible step
  Evaluation: Assessment of this path
Selection: The option you select to explore next.
... (repeat for the selected branch)
Final Answer: Your final response to the task.
""")
    
    LAT_USER_PROMPT = PromptTemplate("""
Task: ${task}

${context}

Please solve this using tree search.
""")
    
    # RAISE prompts
    RAISE_SYSTEM_PROMPT = PromptTemplate("""
You are ${agent_name}, ${agent_description}.

You will solve tasks using a scratch pad for reasoning and examples for guidance:
1. Analyze the task and find relevant examples
2. Work through the problem step by step on your scratch pad
3. Use the examples as a guide for your reasoning
4. Formulate your final answer based on your scratch pad work

Follow this format:
Task Analysis: Your understanding of the task.
Relevant Examples: Examples that can guide your solution.
Scratch Pad:
  Step 1: First step in your reasoning
  Step 2: Second step in your reasoning
  ...
Final Answer: Your final response to the task.
""")
    
    RAISE_USER_PROMPT = PromptTemplate("""
Task: ${task}

${context}

Examples:
${examples}

Please solve this using step-by-step reasoning on a scratch pad.
""")
    
    # ReWOO prompts
    REWOO_SYSTEM_PROMPT = PromptTemplate("""
You are ${agent_name}, ${agent_description}.

You will solve tasks through pure reasoning without external observations:
1. Initial problem analysis
2. Deep reasoning process
3. Multi-step logical deduction
4. Solution formulation

Follow this format:
Problem Analysis: Your understanding of the problem.
Reasoning:
  Step 1: First step in your reasoning
  Step 2: Second step in your reasoning
  ...
Conclusion: The logical conclusion of your reasoning.
Final Answer: Your final response to the task.
""")
    
    REWOO_USER_PROMPT = PromptTemplate("""
Task: ${task}

${context}

Please solve this through pure reasoning without external observations.
""")

def get_prompt_for_architecture(
    architecture: str,
    prompt_type: str,
    **kwargs
) -> str:
    """
    Get a formatted prompt for a specific architecture.
    
    Args:
        architecture: The agent architecture
        prompt_type: The type of prompt (e.g., "system", "user")
        **kwargs: Variables to substitute in the template
        
    Returns:
        The formatted prompt
    """
    library = PromptLibrary()
    
    prompt_map = {
        "react": {
            "system": library.REACT_SYSTEM_PROMPT,
            "user": library.REACT_USER_PROMPT,
        },
        "ooda": {
            "system": library.OODA_SYSTEM_PROMPT,
            "user": library.OODA_USER_PROMPT,
        },
        "bdi": {
            "system": library.BDI_SYSTEM_PROMPT,
            "user": library.BDI_USER_PROMPT,
        },
        "lat": {
            "system": library.LAT_SYSTEM_PROMPT,
            "user": library.LAT_USER_PROMPT,
        },
        "raise": {
            "system": library.RAISE_SYSTEM_PROMPT,
            "user": library.RAISE_USER_PROMPT,
        },
        "rewoo": {
            "system": library.REWOO_SYSTEM_PROMPT,
            "user": library.REWOO_USER_PROMPT,
        },
    }
    
    if architecture not in prompt_map:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    if prompt_type not in prompt_map[architecture]:
        raise ValueError(f"Unknown prompt type: {prompt_type} for architecture: {architecture}")
    
    template = prompt_map[architecture][prompt_type]
    return template.format(**kwargs)