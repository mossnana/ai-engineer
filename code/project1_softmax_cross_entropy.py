import numpy as np

def softmax(logits):
    """
    Computes the softmax of logits.
    Uses max subtraction for numerical stability to prevent overflow.
    """
    # Subtract max for numerical stability
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def cross_entropy_loss(probabilities, target_index):
    """
    Computes the Cross-Entropy Loss for a single prediction.
    """
    # Add a small epsilon to prevent log(0) which is undefined
    epsilon = 1e-12
    return -np.log(probabilities[target_index] + epsilon)

if __name__ == "__main__":
    print("--- Softmax and Cross-Entropy Loss Demonstration ---")
    
    # Example: Vocabulary of 5 words
    vocab = ["The", "cat", "sat", "on", "mat"]
    
    # Raw output scores (logits) from the neural network before Softmax
    logits = np.array([1.2, 0.5, 3.8, -1.0, 2.1])
    print(f"1. Raw Logits: {logits}")
    
    # Apply Softmax to convert logits into probability distribution
    probabilities = softmax(logits)
    print("2. Probabilities after Softmax:")
    for word, prob in zip(vocab, probabilities):
        print(f"   {word:5}: {prob:.4f} ({prob*100:.2f}%)")
        
    print(f"   (Sum of all probabilities: {np.sum(probabilities):.1f})")
    
    # Suppose the actual correct next word in the training data is "sat" (index 2)
    target_index = 2 
    print(f"\n3. Target word index: {target_index} ('{vocab[target_index]}')")
    
    # Calculate the Cross-Entropy Loss
    loss = cross_entropy_loss(probabilities, target_index)
    print(f"4. Cross-Entropy Loss: {loss:.4f}")
    
    # What if the model was very wrong and assigned high probability to "mat" (index 4) instead?
    bad_logits = np.array([1.0, 1.0, -2.0, 1.0, 5.0])
    bad_probs = softmax(bad_logits)
    bad_loss = cross_entropy_loss(bad_probs, target_index)
    print(f"5. Loss if model was wrong (low probability for 'sat'): {bad_loss:.4f}")
