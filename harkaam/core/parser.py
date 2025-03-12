"""
Parser module for the Harkaam framework.

This module provides utilities for parsing LLM responses
for different agent architectures.
"""

from typing import Any, Dict, List, Optional, Tuple
import re
import json

class BaseParser:
    """
    Base class for response parsers.
    
    Response parsers extract structured information from
    LLM responses based on the agent architecture.
    """
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse the text into a structured format.
        
        Args:
            text: The text to parse
            
        Returns:
            A dictionary with the parsed information
        """
        raise NotImplementedError("Subclasses must implement this method")

class ReActParser(BaseParser):
    """
    Parser for ReAct architecture responses.
    """
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse a ReAct response.
        
        Args:
            text: The text to parse
            
        Returns:
            A dictionary with thoughts, actions, observations, and final answer
        """
        # Extract thought-action-observation cycles
        cycles = []
        current_cycle = {}
        
        # Extract final answer
        final_answer_match = re.search(r"Final Answer:\s*(.*?)(?:$|Thought:)", text, re.DOTALL)
        final_answer = final_answer_match.group(1).strip() if final_answer_match else ""
        
        # Extract thoughts
        thought_matches = re.finditer(r"Thought:\s*(.*?)(?:Action:|Observation:|Final Answer:|$)", text, re.DOTALL)
        for match in thought_matches:
            thought = match.group(1).strip()
            if thought:
                current_cycle = {"thought": thought}
                
        # Extract actions
        action_matches = re.finditer(r"Action:\s*(.*?)(?:Observation:|Thought:|Final Answer:|$)", text, re.DOTALL)
        for match in action_matches:
            action = match.group(1).strip()
            if action and "thought" in current_cycle:
                current_cycle["action"] = action
                
        # Extract observations
        obs_matches = re.finditer(r"Observation:\s*(.*?)(?:Thought:|Action:|Final Answer:|$)", text, re.DOTALL)
        for match in obs_matches:
            observation = match.group(1).strip()
            if observation and "thought" in current_cycle and "action" in current_cycle:
                current_cycle["observation"] = observation
                cycles.append(current_cycle.copy())
                current_cycle = {}
        
        # If there's an incomplete cycle, add it anyway
        if current_cycle and "thought" in current_cycle:
            cycles.append(current_cycle.copy())
        
        return {
            "cycles": cycles,
            "final_answer": final_answer,
            "raw_response": text
        }

class OODAParser(BaseParser):
    """
    Parser for OODA architecture responses.
    """
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse an OODA response.
        
        Args:
            text: The text to parse
            
        Returns:
            A dictionary with observations, orientations, decisions, actions, and final answer
        """
        # Extract OODA loops
        loops = []
        current_loop = {}
        
        # Extract final answer
        final_answer_match = re.search(r"Final Answer:\s*(.*?)(?:$|Observation:)", text, re.DOTALL)
        final_answer = final_answer_match.group(1).strip() if final_answer_match else ""
        
        # Extract observations
        obs_matches = re.finditer(r"Observation:\s*(.*?)(?:Orientation:|Decision:|Action:|Final Answer:|$)", text, re.DOTALL)
        for match in obs_matches:
            observation = match.group(1).strip()
            if observation:
                current_loop = {"observation": observation}
                
        # Extract orientations
        orient_matches = re.finditer(r"Orientation:\s*(.*?)(?:Decision:|Action:|Observation:|Final Answer:|$)", text, re.DOTALL)
        for match in orient_matches:
            orientation = match.group(1).strip()
            if orientation and "observation" in current_loop:
                current_loop["orientation"] = orientation
                
        # Extract decisions
        decision_matches = re.finditer(r"Decision:\s*(.*?)(?:Action:|Observation:|Orientation:|Final Answer:|$)", text, re.DOTALL)
        for match in decision_matches:
            decision = match.group(1).strip()
            if decision and "observation" in current_loop and "orientation" in current_loop:
                current_loop["decision"] = decision
                
        # Extract actions
        action_matches = re.finditer(r"Action:\s*(.*?)(?:Observation:|Orientation:|Decision:|Final Answer:|$)", text, re.DOTALL)
        for match in action_matches:
            action = match.group(1).strip()
            if action and "observation" in current_loop and "orientation" in current_loop and "decision" in current_loop:
                current_loop["action"] = action
                loops.append(current_loop.copy())
                current_loop = {}
        
        return {
            "loops": loops,
            "final_answer": final_answer,
            "raw_response": text
        }

class BDIParser(BaseParser):
    """
    Parser for BDI architecture responses.
    """
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse a BDI response.
        
        Args:
            text: The text to parse
            
        Returns:
            A dictionary with beliefs, desires, intentions, executions, and final answer
        """
        # Extract BDI cycles
        cycles = []
        current_cycle = {}
        
        # Extract final answer
        final_answer_match = re.search(r"Final Answer:\s*(.*?)(?:$|Beliefs:)", text, re.DOTALL)
        final_answer = final_answer_match.group(1).strip() if final_answer_match else ""
        
        # Extract beliefs
        belief_matches = re.finditer(r"Beliefs:\s*(.*?)(?:Desires:|Intentions:|Execution:|Final Answer:|$)", text, re.DOTALL)
        for match in belief_matches:
            beliefs = match.group(1).strip()
            if beliefs:
                current_cycle = {"beliefs": beliefs}
                
        # Extract desires
        desire_matches = re.finditer(r"Desires:\s*(.*?)(?:Intentions:|Execution:|Beliefs:|Final Answer:|$)", text, re.DOTALL)
        for match in desire_matches:
            desires = match.group(1).strip()
            if desires and "beliefs" in current_cycle:
                current_cycle["desires"] = desires
                
        # Extract intentions
        intention_matches = re.finditer(r"Intentions:\s*(.*?)(?:Execution:|Beliefs:|Desires:|Final Answer:|$)", text, re.DOTALL)
        for match in intention_matches:
            intentions = match.group(1).strip()
            if intentions and "beliefs" in current_cycle and "desires" in current_cycle:
                current_cycle["intentions"] = intentions
                
        # Extract executions
        execution_matches = re.finditer(r"Execution:\s*(.*?)(?:Beliefs:|Desires:|Intentions:|Final Answer:|$)", text, re.DOTALL)
        for match in execution_matches:
            execution = match.group(1).strip()
            if execution and "beliefs" in current_cycle and "desires" in current_cycle and "intentions" in current_cycle:
                current_cycle["execution"] = execution
                cycles.append(current_cycle.copy())
                current_cycle = {}
        
        return {
            "cycles": cycles,
            "final_answer": final_answer,
            "raw_response": text
        }

class LATParser(BaseParser):
    """
    Parser for LAT architecture responses.
    """
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse a LAT response.
        
        Args:
            text: The text to parse
            
        Returns:
            A dictionary with problems, branches, selections, and final answer
        """
        # Extract tree nodes
        nodes = []
        current_node = {}
        
        # Extract final answer
        final_answer_match = re.search(r"Final Answer:\s*(.*?)(?:$|Problem:)", text, re.DOTALL)
        final_answer = final_answer_match.group(1).strip() if final_answer_match else ""
        
        # Extract problems
        problem_matches = re.finditer(r"Problem:\s*(.*?)(?:Branches:|Selection:|Final Answer:|$)", text, re.DOTALL)
        for match in problem_matches:
            problem = match.group(1).strip()
            if problem:
                current_node = {"problem": problem}
                
        # Extract branches
        branch_matches = re.finditer(r"Branches:(.*?)(?:Selection:|Problem:|Final Answer:|$)", text, re.DOTALL)
        for match in branch_matches:
            branches_text = match.group(1).strip()
            
            # Parse individual branches
            branches = []
            option_matches = re.finditer(r"- Option \d+:\s*(.*?)(?:  Evaluation:|$)", branches_text, re.DOTALL)
            eval_matches = re.finditer(r"  Evaluation:\s*(.*?)(?:- Option \d+:|$)", branches_text, re.DOTALL)
            
            options = [match.group(1).strip() for match in option_matches]
            evaluations = [match.group(1).strip() for match in eval_matches]
            
            for i in range(min(len(options), len(evaluations))):
                branches.append({
                    "option": options[i],
                    "evaluation": evaluations[i]
                })
            
            if branches and "problem" in current_node:
                current_node["branches"] = branches
                
        # Extract selections
        selection_matches = re.finditer(r"Selection:\s*(.*?)(?:Problem:|Branches:|Final Answer:|$)", text, re.DOTALL)
        for match in selection_matches:
            selection = match.group(1).strip()
            if selection and "problem" in current_node and "branches" in current_node:
                current_node["selection"] = selection
                nodes.append(current_node.copy())
                current_node = {}
        
        return {
            "nodes": nodes,
            "final_answer": final_answer,
            "raw_response": text
        }

class RAISEParser(BaseParser):
    """
    Parser for RAISE architecture responses.
    """
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse a RAISE response.
        
        Args:
            text: The text to parse
            
        Returns:
            A dictionary with task analysis, relevant examples, scratch pad, and final answer
        """
        # Extract task analysis
        task_analysis_match = re.search(r"Task Analysis:\s*(.*?)(?:Relevant Examples:|Scratch Pad:|Final Answer:|$)", text, re.DOTALL)
        task_analysis = task_analysis_match.group(1).strip() if task_analysis_match else ""
        
        # Extract relevant examples
        examples_match = re.search(r"Relevant Examples:\s*(.*?)(?:Scratch Pad:|Task Analysis:|Final Answer:|$)", text, re.DOTALL)
        examples = examples_match.group(1).strip() if examples_match else ""
        
        # Extract scratch pad
        scratch_pad_match = re.search(r"Scratch Pad:(.*?)(?:Final Answer:|Task Analysis:|Relevant Examples:|$)", text, re.DOTALL)
        scratch_pad = scratch_pad_match.group(1).strip() if scratch_pad_match else ""
        
        # Extract final answer
        final_answer_match = re.search(r"Final Answer:\s*(.*?)(?:$|Task Analysis:|Relevant Examples:|Scratch Pad:)", text, re.DOTALL)
        final_answer = final_answer_match.group(1).strip() if final_answer_match else ""
        
        # Parse scratch pad steps
        steps = []
        if scratch_pad:
            step_matches = re.finditer(r"  Step \d+:\s*(.*?)(?:  Step \d+:|$)", scratch_pad, re.DOTALL)
            steps = [match.group(1).strip() for match in step_matches]
        
        return {
            "task_analysis": task_analysis,
            "relevant_examples": examples,
            "scratch_pad": {
                "raw": scratch_pad,
                "steps": steps
            },
            "final_answer": final_answer,
            "raw_response": text
        }

class ReWOOParser(BaseParser):
    """
    Parser for ReWOO architecture responses.
    """
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse a ReWOO response.
        
        Args:
            text: The text to parse
            
        Returns:
            A dictionary with problem analysis, reasoning steps, conclusion, and final answer
        """
        # Extract problem analysis
        analysis_match = re.search(r"Problem Analysis:\s*(.*?)(?:Reasoning:|Conclusion:|Final Answer:|$)", text, re.DOTALL)
        analysis = analysis_match.group(1).strip() if analysis_match else ""
        
        # Extract reasoning
        reasoning_match = re.search(r"Reasoning:(.*?)(?:Conclusion:|Problem Analysis:|Final Answer:|$)", text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Extract conclusion
        conclusion_match = re.search(r"Conclusion:\s*(.*?)(?:Final Answer:|Problem Analysis:|Reasoning:|$)", text, re.DOTALL)
        conclusion = conclusion_match.group(1).strip() if conclusion_match else ""
        
        # Extract final answer
        final_answer_match = re.search(r"Final Answer:\s*(.*?)(?:$|Problem Analysis:|Reasoning:|Conclusion:)", text, re.DOTALL)
        final_answer = final_answer_match.group(1).strip() if final_answer_match else ""
        
        # Parse reasoning steps
        steps = []
        if reasoning:
            step_matches = re.finditer(r"  Step \d+:\s*(.*?)(?:  Step \d+:|$)", reasoning, re.DOTALL)
            steps = [match.group(1).strip() for match in step_matches]
        
        return {
            "problem_analysis": analysis,
            "reasoning": {
                "raw": reasoning,
                "steps": steps
            },
            "conclusion": conclusion,
            "final_answer": final_answer,
            "raw_response": text
        }

# Factory function to create a parser for an architecture
def create_parser(architecture: str) -> BaseParser:
    """
    Create a parser for the specified architecture.
    
    Args:
        architecture: The agent architecture
        
    Returns:
        A parser for the architecture
    """
    parsers = {
        "react": ReActParser,
        "ooda": OODAParser,
        "bdi": BDIParser,
        "lat": LATParser,
        "raise": RAISEParser,
        "rewoo": ReWOOParser,
    }
    
    if architecture.lower() not in parsers:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    parser_class = parsers[architecture.lower()]
    return parser_class()