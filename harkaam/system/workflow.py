"""
Workflow module for orchestrating multi-agent systems in the Harkaam framework.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid
import datetime

from pydantic import BaseModel, Field

class WorkflowNode(BaseModel):
    """A node in a workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    name: str
    description: str = ""
    dependencies: List[str] = Field(default_factory=list)
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    transform_input: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    transform_output: Optional[Callable[[Any], Any]] = None

class Workflow:
    """
    A workflow of agent tasks.
    
    A workflow represents a directed graph of tasks, where each task
    is executed by an agent. Tasks can depend on the results of other
    tasks.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a new workflow.
        
        Args:
            name: The name of the workflow
            description: A description of the workflow
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.nodes: Dict[str, WorkflowNode] = {}
        self.agents: Dict[str, Any] = {}
    
    def add_agent(self, agent: Any) -> str:
        """
        Add an agent to the workflow.
        
        Args:
            agent: The agent to add
            
        Returns:
            The ID of the agent
        """
        self.agents[agent.id] = agent
        return agent.id
    
    def add_node(
        self,
        agent: Any,
        name: str,
        description: str = "",
        dependencies: List[str] = None,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        transform_input: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        transform_output: Optional[Callable[[Any], Any]] = None,
    ) -> str:
        """
        Add a node to the workflow.
        
        Args:
            agent: The agent to execute the task
            name: The name of the node
            description: A description of the node
            dependencies: A list of node IDs this node depends on
            condition: A function that determines if the node should execute
            transform_input: A function to transform the input data
            transform_output: A function to transform the output data
            
        Returns:
            The ID of the new node
        """
        # Add the agent to the workflow if it's not already added
        if agent.id not in self.agents:
            self.add_agent(agent)
        
        node = WorkflowNode(
            agent_id=agent.id,
            name=name,
            description=description,
            dependencies=dependencies or [],
            condition=condition,
            transform_input=transform_input,
            transform_output=transform_output,
        )
        
        self.nodes[node.id] = node
        return node.id
    
    def execute(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the workflow.
        
        Args:
            input_data: Input data for the workflow
            
        Returns:
            The results of the workflow execution
        """
        if input_data is None:
            input_data = {}
            
        # Validate the workflow
        self._validate()
        
        # Initialize results dictionary
        results: Dict[str, Any] = {}
        
        # Get the execution order
        execution_order = self._get_execution_order()
        
        # Execute nodes in order
        for node_id in execution_order:
            node = self.nodes[node_id]
            
            # Check if the node should be executed
            if node.condition and not node.condition({**input_data, **results}):
                continue
            
            # Collect inputs from dependencies
            node_input = {**input_data}
            for dep_id in node.dependencies:
                if dep_id in results:
                    node_input[self.nodes[dep_id].name] = results[dep_id]
            
            # Transform input if necessary
            if node.transform_input:
                node_input = node.transform_input(node_input)
            
            # Get the agent for this node
            agent = self.agents[node.agent_id]
            
            # Create a task description
            task_description = f"{node.name}: {node.description}"
            if not task_description.strip():
                task_description = f"Execute task for node {node.name}"
            
            # Execute the task
            result = agent.run(task_description, context=node_input)
            
            # Transform output if necessary
            if node.transform_output:
                results[node_id] = node.transform_output(result)
            else:
                results[node_id] = result
        
        return results
    
    def _validate(self) -> None:
        """Validate the workflow for circular dependencies."""
        # Check that all agents exist
        for node in self.nodes.values():
            if node.agent_id not in self.agents:
                raise ValueError(f"Node {node.name} references non-existent agent {node.agent_id}")
        
        # Check that all dependencies exist
        for node in self.nodes.values():
            for dep_id in node.dependencies:
                if dep_id not in self.nodes:
                    raise ValueError(f"Node {node.name} depends on non-existent node {dep_id}")
        
        # Check for cycles in the dependency graph
        visited = set()
        temp = set()
        
        def has_cycle(node_id: str) -> bool:
            if node_id in temp:
                return True
            if node_id in visited:
                return False
            
            temp.add(node_id)
            
            for dep_id in self.nodes[node_id].dependencies:
                if has_cycle(dep_id):
                    return True
            
            temp.remove(node_id)
            visited.add(node_id)
            return False
        
        # Check each node
        for node_id in self.nodes:
            if has_cycle(node_id):
                raise ValueError("Circular dependency detected in workflow")
    
    def _get_execution_order(self) -> List[str]:
        """Determine the execution order of nodes."""
        # Implementation of topological sort
        result = []
        visited = set()
        temp = set()
        
        def visit(node_id: str) -> None:
            if node_id in temp:
                raise ValueError("Circular dependency detected")
            if node_id in visited:
                return
            
            temp.add(node_id)
            
            # Visit dependencies first
            for dep_id in self.nodes[node_id].dependencies:
                visit(dep_id)
            
            temp.remove(node_id)
            visited.add(node_id)
            result.append(node_id)
        
        # Visit each node
        for node_id in self.nodes:
            if node_id not in visited:
                visit(node_id)
        
        return result