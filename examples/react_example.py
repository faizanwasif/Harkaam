from harkaam import Agent

class ReactExample:
    def __init__(self):
        self.agent = Agent.create(
            architecture="react",
            name="react agent",
            llm="anthropic:claude-3-haiku-20240307",
            description="I help solve problems step by step",
            max_iterations=5,
            verbose=True

        )

    def run(self, task):
        print(f"\nTask: {task}\n")
        print("Executing with React architecture...\n")

        # Run the agent
        result = self.agent.run(task)

        print(f"\nResult: {result}\n")

def main():
    example = ReactExample()
    example.run("How many Rs in the word 'strawberry'?")

if __name__ == "__main__":
    main()