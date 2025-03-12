# Harkaam

Harkaam is a modular framework for building agent-based systems with multiple agent architectures. It provides a flexible foundation for creating both single-agent and multi-agent systems.

## Features

- Support for multiple agent architectures:
  - **ReAct** (Reasoning and Acting)
  - **OODA** (Observe, Orient, Decide, Act)
  - **BDI** (Belief, Desire, Intention)
  - **LAT** (Language Agent Tree Search)
  - **RAISE** (Reasoning and Acting Through Scratch Pad and Examples)
  - **ReWOO** (Reasoning Without Observation)
- Mix and match architectures in multi-agent systems
- Shared components across architectures (memory, tools, LLM integration)
- Workflow orchestration for complex agent interactions
- Extensible design for adding custom architectures

## Installation

```bash
pip install harkaam
```

## Quick Start

Here's a simple example using the ReAct architecture:

```python
from harkaam import Agent
from harkaam.core.tools import Tool

# Define a tool
search_tool = Tool(
    name="search",
    description="Search for information on the web",
    func=lambda query: {"results": ["Result 1", "Result 2"]}
)

# Create a ReAct agent
agent = Agent.create(
    architecture="react",
    name="ResearchAssistant",
    description="Researches information and provides summaries",
    llm="openai:gpt-4",
    tools=[search_tool]
)

# Run the agent
result = agent.run("Research the impact of AI on healthcare")
print(result)
```

## Architecture Selection

Harkaam allows you to choose the right architecture for your specific use case:

- **ReAct**: Good for tasks requiring reasoning and action steps
- **OODA**: Ideal for dynamic environments with changing conditions
- **BDI**: Best for goal-oriented agents with complex belief systems
- **LAT**: Excellent for tasks benefiting from tree search algorithms
- **RAISE**: Useful for complex reasoning with examples and scratch pad
- **ReWOO**: Best for pure reasoning tasks without external observations

## License

