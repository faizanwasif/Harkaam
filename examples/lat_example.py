from harkaam import Agent

class LATExample:
    def __init__(self):
        self.agent = Agent.create(
            architecture="lat",
            name="LAT Agent",
            llm="anthropic:claude-3-haiku-20240307",
            description="I explore multiple solution paths using tree search to find optimal solutions",
            max_depth=5,
            max_branches=3,
            search_strategy="best_first",
            verbose=True
        )

    def run(self, task):
        print(f"\nTask: {task}\n")
        print("Executing with LAT architecture...\n")

        # Run the agent
        result = self.agent.run(task)

        print(f"\nResult: {result}\n")

def main():
    example = LATExample()
    
    # LAT is ideal for problems with multiple possible solution paths
    example.run(
        "You're designing a product recommendation system for an e-commerce website. "
        "Explore different approaches (content-based filtering, collaborative filtering, "
        "hybrid approaches) and determine the best solution considering factors like "
        "cold start problems, scalability, and personalization quality."
    )

if __name__ == "__main__":
    main()