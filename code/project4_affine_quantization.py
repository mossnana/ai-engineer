import numpy as np

def quantize_fp32_to_int8(weights_fp32):
    """
    Quantizes a floating-point array (FP32) to an 8-bit integer (INT8) format
    using Affine Quantization (Asymmetric).
    """
    # 1. Determine the range of the floating point values
    beta = np.min(weights_fp32)
    alpha = np.max(weights_fp32)
    
    # INT8 range: -128 to 127
    q_min = -128
    q_max = 127
    
    # 2. Calculate the Scaling Factor (S)
    # S = (alpha - beta) / (q_max - q_min)
    if alpha == beta: # Handle edge case where all weights are the same
        S = 1.0
    else:
        S = (alpha - beta) / (q_max - q_min)
        
    # 3. Calculate the Zero-point (Z)
    # Z = round(-beta / S) + q_min
    Z = np.round(-beta / S) + q_min
    
    # Ensure Z stays within the INT8 range
    Z = np.clip(Z, q_min, q_max)
    
    # 4. Quantize the weights
    # q = round(x / S) + Z
    q = np.round(weights_fp32 / S) + Z
    
    # Clip to ensure no values fall outside the INT8 bounds
    q_clipped = np.clip(q, q_min, q_max)
    
    # Cast to INT8
    weights_int8 = q_clipped.astype(np.int8)
    
    return weights_int8, S, Z

def dequantize_int8_to_fp32(weights_int8, S, Z):
    """
    Restores INT8 weights back to an approximate FP32 format for computation.
    """
    # x_approx = S * (q - Z)
    return S * (weights_int8.astype(np.float32) - Z)

if __name__ == "__main__":
    print("--- Affine Quantization Demonstration ---")
    
    # Mocking a small layer of weights from an LLM
    # These are high-precision FP32 floats
    original_weights = np.array([-0.84, -0.15, 0.23, 1.45, 2.05, 0.0, -1.9], dtype=np.float32)
    print(f"Original FP32 weights:\n{original_weights}")
    
    # Compress the weights to INT8
    int8_weights, scale, zero_point = quantize_fp32_to_int8(original_weights)
    print(f"\nQuantized INT8 weights:\n{int8_weights}")
    print(f"Scale (S): {scale:.4f}")
    print(f"Zero-point (Z): {zero_point}")
    
    # During inference, the model needs to dequantize them on-the-fly
    restored_weights = dequantize_int8_to_fp32(int8_weights, scale, zero_point)
    print(f"\nDequantized (Restored) FP32 weights:\n{restored_weights}")
    
    # Let's check the compression error (Quantization Loss)
    error = np.abs(original_weights - restored_weights)
    print(f"\nQuantization Error (Absolute Difference):\n{error}")
    print(f"Mean Absolute Error: {np.mean(error):.4f}")
