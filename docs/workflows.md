# Multi-Agent Workflow Management

This document explains how to create and manage workflows with multiple agents in the Harkaam framework.

## Workflow Basics

Workflows in Harkaam allow you to orchestrate multiple agents to collaborate on complex tasks. Each agent can use a different architecture optimized for its specific role.

### Creating a Workflow

```python
from harkaam.system.workflow import Workflow

# Create a new workflow
workflow = Workflow(
    name="Research Report Generator",
    description="Generate comprehensive research reports on various topics"
)
```

### Adding Agents to a Workflow

First, create the agents you want to include in the workflow:

```python
from harkaam import Agent

# Create agents with different architectures
researcher = Agent.create(
    architecture="react",
    name="Researcher",
    description="I gather and synthesize information",
    llm="anthropic:claude-3-haiku-20240307",
    tools=[search_tool, web_tool]
)

analyst = Agent.create(
    architecture="bdi",
    name="Analyst",
    description="I analyze information and identify key insights",
    llm="anthropic:claude-3-haiku-20240307",
    tools=[analyze_tool]
)

writer = Agent.create(
    architecture="rewoo",
    name="Writer",
    description="I create well-written reports",
    llm="anthropic:claude-3-haiku-20240307"
)
```

Then add them to the workflow as nodes:

```python
# Add nodes to the workflow
researcher_node = workflow.add_node(
    agent=researcher,
    name="Research",
    description="Research the given topic"
)

analyst_node = workflow.add_node(
    agent=analyst,
    name="Analysis",
    description="Analyze the research findings",
    dependencies=[researcher_node]  # Depends on researcher results
)

writer_node = workflow.add_node(
    agent=writer,
    name="Writing",
    description="Write a comprehensive report",
    dependencies=[researcher_node, analyst_node]  # Depends on both
)
```

### Executing a Workflow

Once your workflow is set up, you can execute it with initial input data:

```python
# Execute the workflow
results = workflow.execute({
    "topic": "The impact of quantum computing on cryptography",
    "format": "executive summary",
    "length": "1000 words"
})

# Get the final report from the writer node
final_report = results[writer_node]
print(final_report)
```

## Workflow Nodes

Workflow nodes represent agent tasks in the workflow graph:

```python
class WorkflowNode:
    id: str  # Unique identifier
    agent: Agent  # The agent to execute the task
    name: str  # Node name
    description: str  # Node description
    dependencies: List[str]  # IDs of dependent nodes
    condition: Optional[Callable]  # Condition to execute
    transform_input: Optional[Callable]  # Transform input data
    transform_output: Optional[Callable]  # Transform output data
```

### Node Dependencies

Dependencies define the execution order. A node will only execute after all its dependencies have completed:

```python
# Node C depends on Nodes A and B
node_c = workflow.add_node(
    agent=agent_c,
    name="Node C",
    description="Task C",
    dependencies=[node_a, node_b]
)
```

### Conditional Execution

You can add conditions to nodes to determine if they should execute:

```python
def should_execute_detailed_analysis(data):
    # Only run detailed analysis if confidence is low
    return data.get("confidence_score", 1.0) < 0.7

detailed_analysis_node = workflow.add_node(
    agent=analyst,
    name="Detailed Analysis",
    description="Perform detailed analysis when confidence is low",
    dependencies=[initial_analysis_node],
    condition=should_execute_detailed_analysis
)
```

### Data Transformation

You can transform input data before it's sent to an agent and transform output data after the agent completes:

```python
def prepare_analysis_input(data):
    """Transform research data for the analyst."""
    research_data = data.get("Research", {})
    return {
        "text_to_analyze": research_data.get("summary", ""),
        "key_points": research_data.get("key_points", []),
        "sentiment": research_data.get("sentiment", "neutral")
    }

def format_analysis_output(result):
    """Extract and format the analysis results."""
    return {
        "insights": result.output,
        "confidence": result.metadata.get("confidence", 0.5),
        "recommendations": extract_recommendations(result.output)
    }

analysis_node = workflow.add_node(
    agent=analyst,
    name="Analysis",
    description="Analyze research findings",
    dependencies=[research_node],
    transform_input=prepare_analysis_input,
    transform_output=format_analysis_output
)
```

## Advanced Workflow Features

### Parallel Execution

Nodes without dependencies on each other can execute in parallel:

```python
market_research = workflow.add_node(
    agent=researcher,
    name="Market Research",
    description="Research market trends"
)

competitor_analysis = workflow.add_node(
    agent=researcher,
    name="Competitor Analysis",
    description="Analyze competitors"
)

# Both nodes above can run in parallel
strategy_node = workflow.add_node(
    agent=strategist,
    name="Strategy Development",
    description="Develop strategy based on research",
    dependencies=[market_research, competitor_analysis]
)
```

### Error Handling

You can wrap node execution in try-except blocks to handle errors:

```python
try:
    results = workflow.execute(input_data)
except Exception as e:
    print(f"Workflow execution failed: {e}")
    # Implement fallback or recovery logic
```

### Workflow Visualization

You can visualize your workflow structure:

```python
def visualize_workflow(workflow):
    """Generate a simple visualization of the workflow."""
    print(f"Workflow: {workflow.name}")
    print("=" * 50)
    
    for node_id, node in workflow.nodes.items():
        deps = ", ".join([workflow.nodes[dep_id].name for dep_id in node.dependencies])
        deps = deps if deps else "None"
        
        print(f"Node: {node.name}")
        print(f"  Agent: {node.agent.config.name} ({node.agent.__class__.__name__})")
        print(f"  Description: {node.description}")
        print(f"  Dependencies: {deps}")
        print("-" * 30)
```

## Example: Research Workflow

Here's a complete example of a research workflow using multiple architectures:

```python
from harkaam import Agent
from harkaam.system.workflow import Workflow
from harkaam.core.tools import Tool, ToolParameter

# Define tools
search_tool = Tool(
    name="search",
    description="Search for information",
    func=search_function,
    parameters=[ToolParameter("query", "string", "Search query", required=True)]
)

# Create agents
researcher = Agent.create(
    architecture="react",  # Good for information gathering
    name="Researcher",
    description="I gather information on topics",
    llm="anthropic:claude-3-haiku-20240307",
    tools=[search_tool]
)

analyst = Agent.create(
    architecture="bdi",  # Good for analysis with beliefs and goals
    name="Analyst",
    description="I analyze information and identify insights",
    llm="anthropic:claude-3-haiku-20240307"
)

writer = Agent.create(
    architecture="rewoo",  # Good for creative generation
    name="Writer",
    description="I create well-written reports",
    llm="anthropic:claude-3-haiku-20240307"
)

# Create workflow
workflow = Workflow(
    name="Research Report Generator",
    description="Generate research reports on various topics"
)

# Add nodes
research_node = workflow.add_node(
    agent=researcher,
    name="Research",
    description="Research the topic"
)

analysis_node = workflow.add_node(
    agent=analyst,
    name="Analysis",
    description="Analyze the research findings",
    dependencies=[research_node]
)

writing_node = workflow.add_node(
    agent=writer,
    name="Writing",
    description="Write a comprehensive report",
    dependencies=[research_node, analysis_node]
)

# Execute workflow
results = workflow.execute({
    "topic": "Artificial intelligence trends in 2025",
    "format": "executive summary",
    "length": "2 pages"
})

# Print report
print(results[writing_node])
```

This workflow combines the strengths of different agent architectures:
- ReAct for research (good at using tools to gather information)
- BDI for analysis (good at reasoning with beliefs and goals)
- ReWOO for writing (good at pure reasoning to create coherent output)

## Best Practices for Workflows

1. **Choose the right architecture for each task:**
   - ReAct for tool-heavy information gathering
   - OODA for dynamic environments
   - BDI for goal-oriented analysis
   - LAT for exploring multiple paths
   - RAISE for detailed reasoning with examples
   - ReWOO for pure reasoning generation

2. **Design clear data flow:**
   - Use transform functions to ensure clean data handoffs
   - Make sure each node gets exactly what it needs

3. **Handle errors gracefully:**
   - Add error handling for each node
   - Consider fallback strategies

4. **Validate the workflow before running:**
   - Check for circular dependencies
   - Ensure all tools and resources are available

5. **Monitor execution:**
   - Log intermediate results
   - Track execution time and resource usage