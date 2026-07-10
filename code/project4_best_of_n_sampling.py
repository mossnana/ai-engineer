import numpy as np

def mock_generate_reasoning_paths(prompt, N=5):
    """
    Simulates generating N independent reasoning paths from an LLM.
    Returns a list of dictionaries with 'reasoning', 'answer', and a hidden 'correctness' flag.
    """
    paths = []
    for i in range(N):
        # We randomly determine if this path arrived at the right answer
        is_correct = np.random.choice([True, False], p=[0.4, 0.6]) 
        
        if is_correct:
            answer = "42"
            reasoning = f"Step 1: Parse... Step 2: Calculate... Output is 42. (Path {i})"
        else:
            answer = str(np.random.randint(10, 99))
            reasoning = f"Step 1: Missed a carry... Step 2: Output is {answer}. (Path {i})"
            
        paths.append({"id": i, "reasoning": reasoning, "answer": answer, "is_correct": is_correct})
        
    return paths

def mock_verifier_model(path):
    """
    Simulates an external Verifier (or Process Reward Model / PRM).
    It assigns a score (0.0 to 1.0) evaluating how logically sound the reasoning is.
    """
    if path["is_correct"]:
        # Correct logic usually gets a high score, but with some noise
        score = np.random.uniform(0.7, 0.99)
    else:
        # Incorrect logic gets a lower score, but sometimes the model is tricked (Reward Hacking)
        score = np.random.uniform(0.1, 0.6)
        
    return score

def best_of_n_sampling(prompt, N=5):
    """
    Implements Best-of-N Rejection Sampling to boost inference-time compute.
    """
    print(f"Prompt: {prompt}")
    print(f"Generating {N} parallel paths...\n")
    
    paths = mock_generate_reasoning_paths(prompt, N)
    
    best_path = None
    best_score = -1.0
    
    # Evaluate all N paths
    for path in paths:
        score = mock_verifier_model(path)
        print(f"Path {path['id']}: Answer '{path['answer']}' | Verifier Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_path = path
            
    return best_path, best_score

if __name__ == "__main__":
    # Fix seed for reproducible simulation
    np.random.seed(101) 
    
    print("--- Best-of-N Sampling (Test-Time Compute) ---")
    prompt = "If I have 10 apples and multiply by 4, then add 2... what is the result?"
    
    best_result, final_score = best_of_n_sampling(prompt, N=10)
    
    print("\n--- Final Selection ---")
    print(f"Selected Path ID: {best_result['id']}")
    print(f"Reasoning: {best_result['reasoning']}")
    print(f"Final Output: {best_result['answer']} (Score: {final_score:.4f})")
    
    if best_result['is_correct']:
        print("Outcome: SUCCESS - The verifier correctly picked the right answer!")
    else:
        print("Outcome: FAILED - The verifier got tricked by bad logic.")
