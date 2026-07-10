import numpy as np
import matplotlib.pyplot as plt

def generate_linear_schedule(num_timesteps, beta_start=0.0001, beta_end=0.02):
    """Generates a linear variance schedule (betas)."""
    return np.linspace(beta_start, beta_end, num_timesteps)

def forward_diffusion_sample(x_0, t, betas):
    """
    Simulates the forward process at a specific timestep t.
    Calculates q(x_t | x_0) directly using the closed-form math.
    """
    # 1. Calculate alphas
    alphas = 1.0 - betas
    
    # 2. Calculate alpha_bar (cumulative product of alphas up to step t)
    alphas_cumprod = np.cumprod(alphas)
    
    # 3. Get the specific alpha_bar for timestep t
    # Note: t is 0-indexed in code, but conceptually ranges from 1 to T
    alpha_bar_t = alphas_cumprod[t]
    
    # 4. Generate random gaussian noise (epsilon)
    noise = np.random.randn(*x_0.shape)
    
    # 5. Calculate x_t using the reparameterization trick
    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    mean = np.sqrt(alpha_bar_t) * x_0
    variance_term = np.sqrt(1.0 - alpha_bar_t) * noise
    
    x_t = mean + variance_term
    
    return x_t, noise

if __name__ == "__main__":
    print("--- Diffusion Forward Process Simulation ---")
    
    # Let's mock a 1D "image" or vector (e.g., 5 pixels)
    # Original data x_0 (clean image)
    x_0 = np.array([0.5, -0.2, 0.9, -0.8, 0.1])
    print(f"Original clean data (x_0): \n{x_0}")
    
    # Total timesteps T
    T = 1000
    
    # Generate beta schedule
    betas = generate_linear_schedule(T)
    
    # Simulate adding noise at various timesteps
    timesteps_to_check = [0, 50, 200, 500, 999]
    
    for t in timesteps_to_check:
        x_t, added_noise = forward_diffusion_sample(x_0, t, betas)
        print(f"\n--- Timestep {t} ---")
        print(f"Noisy data (x_{t}): \n{x_t}")
        print(f"Signal strength approx: {np.mean(np.abs(x_t)):.4f}")
        
    print("\nNotice how as T approaches 1000, the data becomes pure random Gaussian noise (Mean ~ 0, Variance ~ 1).")
