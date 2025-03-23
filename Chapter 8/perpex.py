import numpy as np

def compute_perplexity(loss):
    return np.exp(loss)

# After training
final_loss = history.history['loss'][-1]
perplexity = compute_perplexity(final_loss)
print(f"Perplexity: {perplexity:.2f}")
