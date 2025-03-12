from harkaam import Agent

class ReWOOExample:
    def __init__(self):
        self.agent = Agent.create(
            architecture="rewoo",
            name="ReWOO Agent",
            llm="anthropic:claude-3-haiku-20240307",
            description="I solve problems through pure reasoning without external observations",
            reasoning_depth=3,
            num_workers=3,
            verbose=True
        )

    def run(self, task):
        print(f"\nTask: {task}\n")
        print("Executing with ReWOO architecture...\n")

        # Run the agent
        result = self.agent.run(task)

        print(f"\nResult: {result}\n")

def main():
    example = ReWOOExample()
    
    # ReWOO is perfect for pure reasoning tasks that can be decomposed into subtasks
    example.run(
        "Consider the ethical implications of using AI in healthcare. "
     
    )

if __name__ == "__main__":
    main()