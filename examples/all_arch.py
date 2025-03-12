from harkaam import Agent

def run_example(architecture, task, **kwargs):
    """Run an example with the specified architecture and task."""
    print(f"\n{'='*80}")
    print(f"Architecture: {architecture.upper()}")
    print(f"{'='*80}")
    print(f"Task: {task}\n")
    
    # Create agent with the specified architecture
    agent = Agent.create(
        architecture=architecture,
        name=f"{architecture.capitalize()} Agent",
        llm="anthropic:claude-3-haiku-20240307",
        description=f"I solve problems using the {architecture} architecture",
        **kwargs
    )
    
    # Run the agent
    result = agent.run(task)
    
    # Print the result
    print(f"\nResult: {result.output}\n")

def main():
    print("\nHarkaam Framework - All Architectures Example\n")
    print("This example demonstrates each agent architecture with a suitable task")
    
    # ReAct (Reasoning and Acting) - best for tasks requiring reasoning and tool use
    run_example(
        "react", 
        "What is the population of Tokyo, and how does it compare to New York City?",
        max_iterations=5
    )
    
    # OODA (Observe, Orient, Decide, Act) - best for dynamic environments
    run_example(
        "ooda",
        "You're a pilot and your engine is experiencing issues. The plane is descending at 500 feet per minute. "
        "Your altitude is 10,000 feet. The nearest airport is 50 miles away. What actions should you take?",
        max_iterations=5
    )
    
    # BDI (Belief, Desire, Intention) - best for goal-oriented planning
    run_example(
        "bdi",
        "You have $1000 to invest. Your goals are: save for retirement, have emergency funds, and grow wealth. "
        "Create an investment plan that balances these goals.",
        max_iterations=5
    )
    
    # LAT (Language Agent Tree Search) - best for exploring multiple solution paths
    run_example(
        "lat",
        "Design a marketing strategy for a new eco-friendly water bottle. Consider different target demographics, "
        "pricing strategies, and marketing channels.",
        max_depth=4,
        max_branches=3
    )
    
    # RAISE (Reasoning and Acting Through Scratch Pad and Examples) - best for step-by-step reasoning
    run_example(
        "raise",
        "Solve this puzzle: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
        "How much does the ball cost?",
        max_iterations=5,
        examples=["To solve algebraic equations, define variables and create equations based on the problem statement"]
    )
    
    # ReWOO (Reasoning Without Observation) - best for pure reasoning without external data
    run_example(
        "rewoo",
        "What philosophical arguments can be made for and against artificial consciousness? "
        "Consider perspectives from different philosophical traditions.",
        reasoning_depth=3,
        num_workers=3
    )

if __name__ == "__main__":
    main()