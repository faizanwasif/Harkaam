# Agent Architectures and Parameters

This document outlines the available agent architectures in the Harkaam framework and their specific parameters.

## Common Parameters (All Architectures)

All agent architectures inherit from the `BaseAgent` class and share these common parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `name` | string | The name of the agent | Required |
| `description` | string | Description of the agent's purpose | Required |
| `llm` | string | LLM provider and model (format: "provider:model") | Required |
| `temperature` | float | Temperature for LLM generation | 0.7 |
| `max_tokens` | int | Maximum tokens for LLM response | 1000 |
| `system_prompt` | string | Custom system prompt | Auto-generated |
| `tools` | list | List of Tool objects | [] |
| `memory` | object | Memory implementation | SimpleMemory |
| `verbose` | object | Shows the thinking of each agent | False |

## Architecture-Specific Parameters

### ReAct (Reasoning and Acting)

```python
agent = Agent.create(
    architecture="react",
    name="React Agent",
    description="I think and act in alternating steps",
    llm="openai:gpt-4",
    max_iterations=5  # React-specific
)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `max_iterations` | int | Maximum number of think-act cycles | 10 |

### OODA (Observe, Orient, Decide, Act)

```python
agent = Agent.create(
    architecture="ooda",
    name="OODA Agent",
    description="I adapt to changing conditions",
    llm="anthropic:claude-3-haiku-20240307",
    max_iterations=5  # OODA-specific
)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `max_iterations` | int | Maximum number of OODA loops | 10 |

### BDI (Belief, Desire, Intention)

```python
agent = Agent.create(
    architecture="bdi",
    name="BDI Agent",
    description="I form beliefs, desires, and intentions",
    llm="anthropic:claude-3-haiku-20240307",
    max_iterations=5  # BDI-specific
)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `max_iterations` | int | Maximum number of BDI cycles | 10 |

### LAT (Language Agent Tree Search)

```python
agent = Agent.create(
    architecture="lat",
    name="LAT Agent",
    description="I explore multiple solution paths",
    llm="anthropic:claude-3-haiku-20240307", 
    max_depth=5,           # LAT-specific
    max_branches=3,        # LAT-specific
    search_strategy="best_first"  # LAT-specific
)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `max_depth` | int | Maximum depth of the search tree | 5 |
| `max_branches` | int | Maximum branches to explore at each node | 3 |
| `search_strategy` | string | Strategy for tree search ("best_first", "breadth_first", "depth_first") | "best_first" |

### RAISE (Reasoning and Acting Through Scratch Pad and Examples)

```python
agent = Agent.create(
    architecture="raise",
    name="RAISE Agent",
    description="I use a scratch pad and examples",
    llm="anthropic:claude-3-haiku-20240307",
    max_iterations=5,      # RAISE-specific
    examples=[             # RAISE-specific
        "Example 1: Step-by-step solution...",
        "Example 2: Another approach..."
    ],
    scratch_pad_format="markdown"  # RAISE-specific
)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `max_iterations` | int | Maximum number of reasoning cycles | 10 |
| `examples` | list | List of example solutions or approaches | [] |
| `scratch_pad_format` | string | Format for the scratch pad ("markdown", "text") | "markdown" |

### ReWOO (Reasoning Without Observation)

```python
agent = Agent.create(
    architecture="rewoo",
    name="ReWOO Agent",
    description="I reason without external observations",
    llm="anthropic:claude-3-haiku-20240307",
    reasoning_depth=3,     # ReWOO-specific
    num_workers=3,         # ReWOO-specific
    reasoning_style="chain_of_thought"  # ReWOO-specific
)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `reasoning_depth` | int | Depth of reasoning for each worker | 3 |
| `num_workers` | int | Number of parallel reasoning workers | 3 |
| `reasoning_style` | string | Style of reasoning ("chain_of_thought", "tree_of_thought") | "chain_of_thought" |

## Creating an Agent

Agents are created using the `Agent.create()` factory method, which returns an instance of the appropriate agent class based on the specified architecture:

```python
from harkaam import Agent

# Create a ReAct agent
react_agent = Agent.create(
    architecture="react",
    name="React Agent",
    description="I solve tasks step by step",
    llm="anthropic:claude-3-haiku-20240307",
    max_iterations=5
)

# Run the agent on a task
result = react_agent.run("What is the square root of 144?")
print(result.output)
```

## Agent Result Structure

All agent architectures return an `AgentResult` object with the following properties:

| Property | Type | Description |
|----------|------|-------------|
| `agent_id` | string | ID of the agent that produced the result |
| `output` | string | Final output/answer from the agent |
| `intermediate_steps` | list | List of intermediate steps taken by the agent |
| `final_state` | AgentState | Final state of the agent |
| `metadata` | dict | Additional metadata (architecture-specific) |