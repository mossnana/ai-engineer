import numpy as np

def sample_next_token(logits, temperature=1.0, top_k=0):
    """
    Samples the next token index given the logits.
    Supports temperature scaling and Top-K sampling.
    """
    logits = np.array(logits)
    
    # 1. Apply Temperature Scaling
    if temperature != 1.0:
        logits = logits / temperature
        
    # 2. Apply Top-K filtering
    if top_k > 0:
        # Find indices of the top_k elements
        indices_to_keep = np.argsort(logits)[-top_k:]
        # Create a mask of negative infinity for elements not in top_k
        filtered_logits = np.full_like(logits, -np.inf)
        filtered_logits[indices_to_keep] = logits[indices_to_keep]
        logits = filtered_logits
        
    # 3. Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    # 4. Sample from the probability distribution
    token_index = np.random.choice(len(probs), p=probs)
    return token_index

if __name__ == "__main__":
    np.random.seed(42)
    
    vocab = {0: "AI", 1: "is", 2: "the", 3: "future", 4: "of", 5: "technology", 6: "<EOS>"}
    
    print("--- Autoregressive Generation Loop ---")
    
    # Starting context
    generated_sequence = [0, 1] # "AI is"
    max_tokens = 5
    
    print(f"Initial sequence: {' '.join([vocab[idx] for idx in generated_sequence])}")
    
    # The Autoregressive Loop
    for step in range(max_tokens):
        # In a real model, we would pass 'generated_sequence' into the Transformer
        # Here we mock the output logits for demonstration
        mock_logits = np.random.randn(len(vocab)) 
        
        # We simulate the model strongly preferring certain tokens depending on the step
        if step == 0:
            mock_logits[2] = 5.0 # high probability for "the"
        elif step == 1:
            mock_logits[3] = 4.0 # high probability for "future"
        
        # Generate the next token using sampling
        next_token = sample_next_token(mock_logits, temperature=0.7, top_k=3)
        
        # Append the new token to our sequence
        generated_sequence.append(next_token)
        
        word = vocab[next_token]
        print(f"Step {step+1}: Sampled '{word}' -> Current sequence: {' '.join([vocab[idx] for idx in generated_sequence])}")
        
        if word == "<EOS>":
            print("End of sequence token generated. Stopping.")
            break
