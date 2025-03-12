from harkaam import Agent

class RAISEExample:
    def __init__(self):
        self.agent = Agent.create(
            architecture="raise",
            name="RAISE Agent",
            llm="anthropic:claude-3-haiku-20240307",
            description="I use a scratch pad and examples to solve problems with detailed step-by-step reasoning",
            max_iterations=5,
            examples=[
                "To find the area of a triangle: multiply base × height ÷ 2",
                "To solve a quadratic equation: use the formula x = (-b ± sqrt(b² - 4ac)) ÷ 2a"
            ],
            verbose=True
        )

    def run(self, task):
        print(f"\nTask: {task}\n")
        print("Executing with RAISE architecture...\n")

        # Run the agent
        result = self.agent.run(task)

        print(f"\nResult: {result}\n")

def main():
    example = RAISEExample()
    
    # RAISE excels at problems requiring detailed step-by-step reasoning with examples
    example.run(
        "Solve this problem step by step, showing your work: A rectangular garden plot is 12 feet by 8 feet. "
        "You want to create a 2-foot wide path around the outside of the garden. "
        "What is the total area of the garden plus the path? What is the area of just the path?"
    )

if __name__ == "__main__":
    main()