# Memory Management Across Architectures

This document explains how memory works in the Harkaam framework and how different agent architectures use memory.

## Memory Types

The Harkaam framework provides several memory implementations that agents can use to store and retrieve information.

### Base Memory

All memory implementations inherit from the `BaseMemory` abstract base class, which defines the following methods:

```python
class BaseMemory:
    def add(self, key: str, value: Any) -> None:
        """Add a value to memory"""
        
    def get(self, key: str) -> Optional[Any]:
        """Get a value from memory"""
        
    def update(self, key: str, value: Any) -> None:
        """Update an existing value in memory"""
        
    def delete(self, key: str) -> None:
        """Delete a value from memory"""
        
    def clear(self) -> None:
        """Clear all values from memory"""
```

### SimpleMemory

The default memory implementation is `SimpleMemory`, which stores values in a Python dictionary:

```python
from harkaam.core.memory import SimpleMemory

# Create a simple memory
memory = SimpleMemory()

# Add a value
memory.add("task", "Calculate the square root of 144")

# Get a value
task = memory.get("task")

# Update a value
memory.update("task", "Calculate the cube root of 27")

# Get all stored values
all_data = memory.get_all()
```

### ConversationBufferMemory

For agents that need to maintain a conversation history, `ConversationBufferMemory` provides message storage:

```python
from harkaam.core.memory import ConversationBufferMemory

# Create a conversation buffer memory
memory = ConversationBufferMemory(max_messages=100)

# Add messages
memory.add_message("user", "What's the weather like?")
memory.add_message("agent", "It's sunny today.")

# Get conversation history
history = memory.get_conversation_history(n=5)  # Get last 5 messages
```

## Providing Memory to Agents

You can provide a memory implementation to any agent architecture when creating the agent:

```python
from harkaam import Agent
from harkaam.core.memory import ConversationBufferMemory

# Create a memory
memory = ConversationBufferMemory()

# Create an agent with custom memory
agent = Agent.create(
    architecture="react",  # Can be any architecture
    name="Agent with Memory",
    description="I remember past interactions",
    llm="anthropic:claude-3-haiku-20240307",
    memory=memory
)
```

## How Different Architectures Use Memory

Each agent architecture uses memory in a way that suits its particular approach:

### ReAct (Reasoning and Acting)

ReAct agents store:
- Tasks and their descriptions
- Thoughts generated during reasoning
- Actions taken and their results
- Observations from tool usage

```python
# In the ReAct architecture
self.memory.add(f"task:{task_id}", task_data)
self.memory.add(f"thought:{step_id}", thought)
self.memory.add(f"action:{step_id}", action)
self.memory.add(f"observation:{step_id}", observation)
```

### OODA (Observe, Orient, Decide, Act)

OODA agents maintain memory of:
- Observations from the environment
- Orientations (analyses of observations)
- Decisions made
- Actions taken and their results

```python
# In the OODA architecture
self.memory.add(f"observation:{cycle_id}", observation)
self.memory.add(f"orientation:{cycle_id}", orientation)
self.memory.add(f"decision:{cycle_id}", decision)
self.memory.add(f"action:{cycle_id}", action_result)
```

### BDI (Belief, Desire, Intention)

BDI agents have structured memory for:
- Beliefs about the world
- Desires (goals)
- Intentions (plans)
- Actions taken and their outcomes

```python
# In the BDI architecture
self.memory.add(f"beliefs:{cycle_id}", beliefs)
self.memory.add(f"desires:{cycle_id}", desires)
self.memory.add(f"intentions:{cycle_id}", intentions)
self.memory.add(f"actions:{cycle_id}", action_results)
```

### LAT (Language Agent Tree Search)

LAT agents maintain a tree structure in memory:
- Decision tree nodes
- Simulation results for each path
- Evaluation metrics for different branches
- Backpropagation data

```python
# In the LAT architecture
self.memory.add(f"node:{node_id}", node_data)
self.memory.add(f"simulation:{node_id}", simulation_results)
self.memory.add(f"evaluation:{node_id}", evaluation_metrics)
self.memory.add("best_path", path_data)
```

### RAISE (Reasoning and Acting Through Scratch Pad and Examples)

RAISE agents use memory primarily for the scratch pad:
- Scratch pad content and updates
- Examples relevant to the task
- Thought processes
- Tool usage and results

```python
# In the RAISE architecture
self.memory.add("scratch_pad", scratch_pad_content)
self.memory.add("examples", examples)
self.memory.add(f"thought:{step_id}", thought)
self.memory.add(f"tool_result:{step_id}", tool_result)
```

### ReWOO (Reasoning Without Observation)

ReWOO agents maintain memory of:
- Task plan
- Worker assignments
- Worker reasoning results
- Integrated solution

```python
# In the ReWOO architecture
self.memory.add("plan", plan)
self.memory.add(f"worker_task:{worker_id}", task)
self.memory.add(f"worker_result:{worker_id}", result)
self.memory.add("solution", integrated_solution)
```

## Agent State

In addition to explicit memory, all agents maintain an `AgentState` object that tracks:

- Current stage of execution
- Step count
- Context information
- Working memory
- Execution history

```python
class AgentState:
    stage: str = "idle"
    step_count: int = 0
    context: Dict[str, Any] = {}
    working_memory: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []
```

Agents update their state throughout execution:

```python
self._update_state(
    stage="thinking",
    step_count=3,
    context_update={"new_info": "value"},
    add_to_history={"type": "thought", "content": "This is a thought"}
)
```

## Creating Custom Memory Implementations

You can create custom memory implementations by inheriting from `BaseMemory`:

```python
from harkaam.core.memory import BaseMemory

class PersistentMemory(BaseMemory):
    """Memory that persists to disk."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._load()
    
    def _load(self):
        # Load data from file
        # ...
    
    def _save(self):
        # Save data to file
        # ...
    
    def add(self, key: str, value: Any) -> None:
        # Add value and save
        # ...
    
    # Implement other required methods
    # ...
```

## Factory Function

The Harkaam framework provides a factory function to create memory instances:

```python
from harkaam.core.memory import create_memory

# Create memory with specific type
memory = create_memory(
    memory_type="conversation_buffer",
    max_messages=50
)
```

## Memory in Workflows

When using multi-agent workflows, each agent maintains its own memory, but results can be passed between agents:

```python
# In workflow.py
results = workflow.execute({
    "initial_data": "Some data to start with"
})

# Results from agent1 are passed to agent2
node2_input = {
    "agent1_result": results[node1_id]
}
```

This allows for complex workflows where information is shared between agents with different memory structures.