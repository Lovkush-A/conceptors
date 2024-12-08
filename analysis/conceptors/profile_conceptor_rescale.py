import torch
import time
from src.functionvectors.utils.conceptors import rescale_conceptor

n_repeats = 5

# Create a random matrix for testing
matrix_size = (6144, 6144)  # Adjust size as needed
matrix = torch.randn(matrix_size)

# Function to profile execution time
def profile_time(matrix, device):
    matrix = matrix.to(device)
    start_time = time.time()
    result = rescale_conceptor(matrix, 0.01, 0.1)
    torch.cuda.synchronize() if device == 'cuda' else None  # Ensure all operations are completed
    end_time = time.time()
    return end_time - start_time

# Profile on CPU
cpu_times = []
for _ in range(n_repeats):
    cpu_time = profile_time(matrix, 'cpu')
    cpu_times.append(cpu_time)
    print(f"Time on CPU: {cpu_time:.6f} seconds")
print(f"Average time on CPU: {sum(cpu_times) / len(cpu_times):.6f} seconds")

# Profile on GPU (if available)
if torch.cuda.is_available():
    gpu_times = []
    for _ in range(n_repeats):
        gpu_time = profile_time(matrix, 'cuda')
        gpu_times.append(gpu_time)
        print(f"Time on GPU: {gpu_time:.6f} seconds")
    print(f"Average time on GPU: {sum(gpu_times) / len(gpu_times):.6f} seconds")
else:
    print("GPU is not available.")

# Time on CPU: 2.740021 seconds
# Time on CPU: 2.681866 seconds
# Time on CPU: 2.678697 seconds
# Time on CPU: 2.725862 seconds
# Time on CPU: 2.718664 seconds
# Average time on CPU: 2.709022 seconds
# Time on GPU: 0.306735 seconds
# Time on GPU: 0.052717 seconds
# Time on GPU: 0.054588 seconds
# Time on GPU: 0.054866 seconds
# Time on GPU: 0.054733 seconds
# Average time on GPU: 0.104728 seconds