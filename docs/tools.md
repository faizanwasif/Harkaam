# Tool Usage Across Architectures

This document explains how to create and use tools in the Harkaam framework, and how they work with different agent architectures.

## Tool Definition

Tools provide agents with the ability to interact with external systems or perform specific operations.

### Creating a Tool

```python
from harkaam.core.tools import Tool, ToolParameter

# Define a function that the tool will execute
def calculator(expression: str) -> dict:
    try:
        result = eval(expression)  # Simple eval for demonstration
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# Create the tool
calc_tool = Tool(
    name="calculator",
    description="Perform mathematical calculations",
    func=calculator,
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Mathematical expression to evaluate",
            required=True
        )
    ]
)
```

### Tool Parameters

The `Tool` constructor accepts the following parameters:

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `name` | string | Name of the tool | Yes |
| `description` | string | Description of what the tool does | Yes |
| `func` | callable | Function that the tool executes | Yes |
| `parameters` | list | List of ToolParameter objects | No |

The `ToolParameter` constructor accepts:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `name` | string | Name of the parameter | Required |
| `type` | string | Parameter type ("string", "integer", "float", "boolean", "array", "object") | Required |
| `description` | string | Description of the parameter | Required |
| `required` | boolean | Whether the parameter is required | True |
| `default` | any | Default value if not provided | None |

## Providing Tools to Agents

You can provide tools to any agent architecture when creating the agent:

```python
from harkaam import Agent
from harkaam.core.tools import Tool, ToolParameter

# Create tools
search_tool = Tool(
    name="search",
    description="Search for information",
    func=search_function,
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Search query",
            required=True
        )
    ]
)

# Create an agent with tools
agent = Agent.create(
    architecture="react",  # Can be any architecture
    name="Agent with Tools",
    description="I use tools to help solve problems",
    llm="anthropic:claude-3-haiku-20240307",
    tools=[search_tool]  # Provide the tools here
)
```

## How Different Architectures Use Tools

### ReAct (Reasoning and Acting)

ReAct agents are specifically designed for tool use. They alternate between thinking and acting (using tools):

```
Thought: I need to calculate 25 * 16
Action: calculator: 25 * 16
Observation: Tool 'calculator' result: {"result": 400}
Thought: Now I know that 25 * 16 = 400
```

### OODA (Observe, Orient, Decide, Act)

OODA agents use tools in the Act phase of their cycle:

```
Observation: I need to find the population of Tokyo
Orientation: I should use a search tool to find this information
Decision: I will search for "population of Tokyo"
Action: search: population of Tokyo
```

### BDI (Belief, Desire, Intention)

BDI agents use tools as part of executing their intentions:

```
Beliefs: I need to find recent weather data for New York
Desires: I want to determine if it will rain tomorrow
Intentions: I will search for weather forecasts for New York
Actions: search: weather forecast New York
```

### LAT (Language Agent Tree Search)

LAT agents can use tools when simulating outcomes of different paths:

```
Problem: Need to know the GDP of France
Branch 1: Search for current GDP of France
Branch 2: Calculate GDP based on available economic indicators
Evaluation: Branch 1 is more direct
Selection: search: GDP of France
```

### RAISE (Reasoning and Acting Through Scratch Pad and Examples)

RAISE agents record tool usage and results in their scratch pad:

```
Thought: I need to convert 25 EUR to USD
Action: currency_converter: EUR to USD 25
Observation: Tool 'currency_converter' result: {"result": 27.32}
Scratch Pad Update: 25 EUR = 27.32 USD
```

### ReWOO (Reasoning Without Observation)

ReWOO agents typically don't use external tools (they focus on pure reasoning). However, if needed, they can use tools through their workers or the planner component:

```
Planner: Need to gather information for the analysis
Worker 1: search: recent advancements in quantum computing
Worker 2: Pure reasoning about implications
Worker 3: Pure reasoning about applications
Solver: Integrates all worker outputs
```

## Simplified Tool Usage Format

For all architectures, Harkaam uses a simplified format for tool usage:

```
TOOL_NAME: PARAMETER
```

Examples:
- `search: weather in Tokyo`
- `calculator: 25 * 16`
- `translate: Hello world to Spanish`

This simple format makes it easy for agents to use tools without needing complex syntax.

## Creating Custom Tools

You can create custom tools to extend the capabilities of your agents:

```python
def api_call(endpoint: str, params: dict) -> dict:
    """Make an API call to the specified endpoint with parameters."""
    # Implementation details...
    return {"status": "success", "data": {...}}

api_tool = Tool(
    name="api",
    description="Make API calls to external services",
    func=api_call,
    parameters=[
        ToolParameter(
            name="endpoint",
            type="string",
            description="API endpoint",
            required=True
        ),
        ToolParameter(
            name="params",
            type="object",
            description="API parameters",
            required=False,
            default={}
        )
    ]
)
```

## Tool Registry

The Harkaam framework maintains a registry of tools for each agent. This ensures that agents can only use tools that have been explicitly provided to them:

```python
from harkaam.core.tools import ToolRegistry

# Create a registry
registry = ToolRegistry()

# Register tools
registry.register(search_tool)
registry.register(calculator_tool)

# Get a tool
search = registry.get("search")

# List all tools
all_tools = registry.list()
```