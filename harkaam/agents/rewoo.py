"""
ReWOO (Reasoning Without Observation) agent implementation.

ReWOO is an agent architecture that focuses on pure reasoning tasks without
relying on external observations. It emphasizes internal reasoning processes 
to solve problems.
"""

from typing import Any, Dict, List
import re

from harkaam.agents.base import BaseAgent, AgentResult
from harkaam.core.llm import create_llm
from harkaam.core.prompt import get_prompt_for_architecture

class ReWOOAgent(BaseAgent):
    """
    ReWOO agent implementation.
    
    The ReWOO architecture focuses on pure reasoning without external
    observations. It's designed for tasks that require deep reasoning
    rather than interaction with the environment.
    """
    
    def __init__(self, **kwargs):
        """Initialize a new ReWOO agent."""
        super().__init__(**kwargs)
        
        # Initialize core components
        self.llm_client = create_llm(self.config.llm)
        
        # ReWOO-specific configuration
        self.reasoning_depth = kwargs.get("reasoning_depth", 3)
        self.reasoning_style = kwargs.get("reasoning_style", "chain_of_thought")
        self.num_workers = kwargs.get("num_workers", 3)
    
    def execute(self, task: str, **kwargs) -> AgentResult:
        """
        Execute a task using the ReWOO architecture:
        1. Task planning by the planner
        2. Assignment of subtasks to workers
        3. Workers perform reasoning
        4. Solver combines worker outputs
        """
        # Initialize context and tracking
        context = self._initialize_context(task, **kwargs)
        intermediate_steps = []
        
        # Step 1: Create plan
        plan = self._generate_llm_response(
            context, 
            f"You are the Planner component. Create a plan with {self.num_workers} subtasks that can be solved in parallel by different worker agents.",
            "plan",
            "Planner"
        )
        context["plan"] = plan
        intermediate_steps.append({"type": "plan", "content": plan})
        
        # Step 2: Extract worker tasks from plan
        worker_tasks = self._assign_worker_tasks(context)
        context["worker_tasks"] = worker_tasks
        intermediate_steps.append({"type": "worker_tasks", "content": worker_tasks})
        
        # Step 3: Workers perform reasoning
        worker_results = []
        for i, worker_task in enumerate(worker_tasks):
            worker_id = i + 1
            worker_result = self._generate_llm_response(
                context,
                f"Your Subtask: {worker_task}\n\nYou are Worker {worker_id}. Solve your assigned subtask through pure reasoning, without using external tools or observations. Think step by step.",
                f"worker_{worker_id}",
                f"Worker {worker_id}"
            )
            worker_results.append(worker_result)
            intermediate_steps.append({"type": f"worker_{worker_id}_result", "content": worker_result})
        
        context["worker_results"] = worker_results
        
        # Step 4: Solver integrates results
        solution = self._generate_llm_response(
            context,
            self._format_worker_results(worker_results) + 
            "\nYou are the Solver. Integrate the results from all workers to provide a comprehensive solution to the main task.",
            "solution",
            "Solver"
        )
        
        intermediate_steps.append({"type": "solution", "content": solution})
        
        # Update final state
        self._update_state(
            stage="completed",
            add_to_history={
                "plan": plan,
                "worker_results": worker_results,
                "solution": solution
            }
        )
        
        # Return result
        return AgentResult(
            agent_id=self.id,
            output=solution,
            intermediate_steps=intermediate_steps,
            final_state=self.state,
            metadata={
                "architecture": "rewoo",
                "reasoning_depth": self.reasoning_depth,
                "reasoning_style": self.reasoning_style,
                "num_workers": self.num_workers
            }
        )
    
    def _initialize_context(self, task: str, **kwargs) -> Dict[str, Any]:
        """Initialize the context for the ReWOO agent."""
        # Create initial context
        context = {
            "task": task,
            "context_info": {},
            "exemplars": [],
        }
        
        # Add additional context from kwargs
        if "context" in kwargs and isinstance(kwargs["context"], dict):
            context["context_info"] = kwargs["context"]
        
        # Add exemplars if provided
        if "exemplars" in kwargs and isinstance(kwargs["exemplars"], list):
            context["exemplars"] = kwargs["exemplars"]
        
        # Update agent state
        self._update_state(
            stage="initializing",
            context_update={"task": task},
            working_memory=context
        )
        
        return context
    
    def _generate_llm_response(self, context: Dict[str, Any], prompt_addition: str, stage_name: str, role_name: str = None) -> str:
        """Generate a response from the LLM for a specific stage."""
        # Get system prompt
        system_prompt = get_prompt_for_architecture(
            architecture="rewoo",
            prompt_type="system",
            agent_name=role_name or self.config.name,
            agent_description=self.config.description
        )
        
        # Create user prompt
        user_prompt = f"Task: {context['task']}\n\n"
        
        # Add context information if available
        if context["context_info"]:
            user_prompt += "Context information:\n"
            for key, value in context["context_info"].items():
                user_prompt += f"- {key}: {value}\n"
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
        self._update_state(stage=stage_name, add_to_history={stage_name: response})
        
        return response
    
    def _assign_worker_tasks(self, context: Dict[str, Any]) -> List[str]:
        """Extract worker tasks from the plan."""
        if "plan" not in context:
            return self._create_default_tasks(context)
        
        plan = context["plan"]
        
        # Common patterns for task extraction
        task_patterns = [
            r"(?:Worker|Task)\s+\d+:\s*(.*?)(?=(?:Worker|Task)\s+\d+:|$)",  # Worker 1: task
            r"(?:Worker|Task)\s+\d+[\.|\)]\s*(.*?)(?=(?:Worker|Task)\s+\d+|$)",  # Worker 1. task
            r"\d+\.\s*(.*?)(?=\d+\.|$)",  # 1. task
            r"- (.*?)(?=-|$)",  # - task
            r"•\s*(.*?)(?=•|$)",  # • task
            r"Subtask\s+\d+:\s*(.*?)(?=Subtask|$)"  # Subtask 1: description
        ]
        
        # Try each pattern
        for pattern in task_patterns:
            tasks = re.findall(pattern, plan, re.DOTALL)
            if tasks and len(tasks) >= 1:
                # Clean and limit tasks
                cleaned_tasks = [task.strip() for task in tasks if task.strip()]
                if len(cleaned_tasks) > 0:
                    return cleaned_tasks[:self.num_workers]
        
        # If extraction fails, ask LLM to extract tasks
        extraction_prompt = f"Plan:\n{plan}\n\nExtract {self.num_workers} specific worker tasks from this plan. Number them clearly."
        
        _, extraction_response = self.llm_client.generate(
            system_prompt="Extract specific worker tasks from this plan.",
            user_prompt=extraction_prompt,
            temperature=0.3,
            max_tokens=500
        )
        
        # Try patterns again on the extraction response
        for pattern in task_patterns:
            tasks = re.findall(pattern, extraction_response, re.DOTALL)
            if tasks and len(tasks) >= 1:
                cleaned_tasks = [task.strip() for task in tasks if task.strip()]
                if len(cleaned_tasks) > 0:
                    return cleaned_tasks[:self.num_workers]
        
        # If still no success, use default tasks
        return self._create_default_tasks(context)
    
    def _create_default_tasks(self, context: Dict[str, Any]) -> List[str]:
        """Create default worker tasks when extraction fails."""
        worker_tasks = []
        for i in range(self.num_workers):
            if i == 0:
                task = f"Analyze the core aspects of {context['task']}"
            elif i == 1:
                task = f"Consider alternative approaches to {context['task']}"
            else:
                task = f"Identify potential challenges or edge cases in {context['task']}"
            worker_tasks.append(task)
        
        return worker_tasks
    
    def _format_worker_results(self, worker_results: List[str]) -> str:
        """Format worker results for the solver."""
        formatted = "Worker Results:\n"
        for i, result in enumerate(worker_results):
            formatted += f"Worker {i+1} Result:\n{result}\n\n"
        return formatted