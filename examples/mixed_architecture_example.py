from harkaam import Agent
from harkaam.system.workflow import Workflow

class MixedArchitectureExample:
    def __init__(self):
        # Create agents with different architectures
        self.researcher = Agent.create(
            architecture="react",
            name="Research Agent",
            llm="anthropic:claude-3-haiku-20240307",
            description="I gather and synthesize information",
            max_iterations=3,
            verbose=True
        
        )
        
        self.analyst = Agent.create(
            architecture="bdi",
            name="Analysis Agent",
            llm="anthropic:claude-3-haiku-20240307",
            description="I analyze information and identify key insights",
            max_iterations=3,
            verbose=True
        )
        
        self.writer = Agent.create(
            architecture="rewoo",
            name="Writing Agent",
            llm="anthropic:claude-3-haiku-20240307",
            description="I create well-written reports based on analysis",
            reasoning_depth=3,
            verbose=True
        )
        
        # Create a workflow
        self.workflow = Workflow(
            name="Multi-Architecture Report Generator",
            description="Generate a comprehensive report using multiple agent architectures"

        )
        
        # Add nodes to the workflow
        self.researcher_node = self.workflow.add_node(
            agent=self.researcher,
            name="Research",
            description="Research the topic"
        )
        
        self.analyst_node = self.workflow.add_node(
            agent=self.analyst,
            name="Analysis",
            description="Analyze the research findings",
            dependencies=[self.researcher_node]
        )
        
        self.writer_node = self.workflow.add_node(
            agent=self.writer,
            name="Report Writing",
            description="Write a comprehensive report",
            dependencies=[self.researcher_node, self.analyst_node]
        )

    def run(self, topic):
        print(f"\nTopic: {topic}\n")
        print("Executing multi-architecture workflow...\n")
        
        # Execute the workflow
        results = self.workflow.execute({
            "topic": topic,
            "format": "report",
            "length": "concise"
        })
        
        # Print the final report
        if self.writer_node in results:
            print(f"\nFinal Report:\n{results[self.writer_node]}\n")
        else:
            print("Workflow did not complete successfully")

def main():
    example = MixedArchitectureExample()
    
    # This example combines multiple architectures in a workflow
    # ReAct for research (good at tool use and information gathering)
    # BDI for analysis (good at goal-oriented reasoning)
    # ReWOO for writing (good at pure reasoning without external data)
    example.run("The future impact of quantum computing on cybersecurity")

if __name__ == "__main__":
    main()