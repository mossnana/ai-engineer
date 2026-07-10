import re

class ReACTAgent:
    """
    A simplified ReACT (Reasoning and Acting) Agent loop.
    It alternates between THINKING (Reasoning) and ACTING (Tools).
    """
    def __init__(self, tools):
        self.tools = tools
        self.max_steps = 5
        self.history = []
        
    def mock_llm_generate(self, prompt):
        """
        In a real app, this calls OpenAI or a local LLM.
        Here we mock the response based on the history length.
        """
        step_count = len([msg for msg in self.history if msg.startswith("Thought:")])
        
        if step_count == 0:
            return "Thought: I need to find out the capital of France.\nAction: search\nAction Input: Capital of France"
        elif step_count == 1:
            return "Thought: The search result says Paris is the capital. Now I need to find its population.\nAction: search\nAction Input: Population of Paris 2024"
        elif step_count == 2:
            return "Thought: The population is 2.1 million. I have all the information.\nAction: finish\nAction Input: The capital of France is Paris and its population is approximately 2.1 million."
        else:
            return "Action: finish\nAction Input: I reached my step limit."

    def execute_action(self, action_name, action_input):
        print(f"  --> [Executing Tool: {action_name} with '{action_input}']")
        if action_name == "search":
            if "Capital" in action_input:
                return "Observation: Paris is the capital city of France."
            elif "Population" in action_input:
                return "Observation: As of 2024, the population of Paris is roughly 2.1 million."
            return "Observation: No results found."
        else:
            return f"Observation: Unknown tool {action_name}"

    def run(self, user_query):
        print(f"User: {user_query}")
        self.history.append(f"User: {user_query}")
        
        for step in range(self.max_steps):
            print(f"\n--- Step {step + 1} ---")
            
            # 1. Generate next thought and action
            # (Passing the whole history to the LLM as context)
            prompt = "\n".join(self.history)
            llm_response = self.mock_llm_generate(prompt)
            print(f"Agent:\n{llm_response}")
            self.history.append(llm_response)
            
            # 2. Parse the LLM output using Regex
            action_match = re.search(r"Action: (.*?)\nAction Input: (.*)", llm_response, re.DOTALL)
            
            if action_match:
                action_name = action_match.group(1).strip()
                action_input = action_match.group(2).strip()
                
                # 3. Check for stopping condition
                if action_name.lower() == "finish":
                    print(f"\nFinal Answer: {action_input}")
                    return action_input
                
                # 4. Execute the tool (Observation)
                observation = self.execute_action(action_name, action_input)
                print(observation)
                self.history.append(observation)
            else:
                print("Error: Could not parse action from LLM output.")
                break
                
        print("\nAgent stopped: Reached maximum steps.")
        return "I could not find the answer in time."

if __name__ == "__main__":
    print("--- ReACT Agent Loop Simulation ---")
    agent = ReACTAgent(tools=["search"])
    agent.run("What is the capital of France, and what is its population?")
