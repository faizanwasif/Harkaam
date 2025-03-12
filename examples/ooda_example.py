from harkaam import Agent

class OODAExample:
    def __init__(self):
        self.agent = Agent.create(
            architecture="ooda",
            name="OODA Agent",
            llm="anthropic:claude-3-haiku-20240307",
            description="I adapt to changing conditions and make decisions based on situational awareness",
            max_iterations=5,
            verbose=True
        )

    def run(self, task):
        print(f"\nTask: {task}\n")
        print("Executing with OODA architecture...\n")

        # Run the agent
        result = self.agent.run(task)

        print(f"\nResult: {result}\n")

def main():
    example = OODAExample()
    
    # OODA is great for dynamic environments with changing conditions
    example.run(
        "You're navigating a ship through a storm. The wind is coming from the east at 30 knots. "
        "The current is pushing you southwest. There are rocks to the north. "
        "What's your best course of action, and how would you adjust if the wind changes direction?"
    )

if __name__ == "__main__":
    main()