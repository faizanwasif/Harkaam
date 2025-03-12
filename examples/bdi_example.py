from harkaam import Agent

class BDIExample:
    def __init__(self):
        self.agent = Agent.create(
            architecture="bdi",
            name="BDI Agent",
            llm="anthropic:claude-3-haiku-20240307",
            description="I form beliefs about the world, generate desires, and create intentions to achieve my goals",
            max_iterations=5,
            verbose=True
        )

    def run(self, task):
        print(f"\nTask: {task}\n")
        print("Executing with BDI architecture...\n")

        # Run the agent
        result = self.agent.run(task)

        print(f"\nResult: {result}\n")

def main():
    example = BDIExample()
    
    # BDI is excellent for goal-oriented planning with multiple objectives
    example.run(
        "You're planning a family vacation with the following constraints: "
        "Budget of $3000, 5-day timeframe, must be kid-friendly, one family member has mobility issues, "
        "and another is allergic to seafood. Create a vacation plan that satisfies all these requirements."
    )

if __name__ == "__main__":
    main()