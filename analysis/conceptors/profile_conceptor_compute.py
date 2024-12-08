import torch
import time
from src.functionvectors.utils.conceptors import compute_conceptor

n_repeats = 5

# Create a random matrix for testing
matrix_size = (100, 6144)  # Adjust size as needed
matrix = torch.randn(matrix_size)

# Function to profile execution time
def profile_time(matrix, device):
    matrix = matrix.to(device)
    start_time = time.time()
    result = compute_conceptor(matrix, aperture=0.01)
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

# Time on CPU: 15.670605 seconds
# Time on CPU: 15.588026 seconds
# Time on CPU: 15.384402 seconds
# Time on CPU: 15.643998 seconds
# Time on CPU: 15.471345 seconds
# Average time on CPU: 15.551675 seconds
# Time on GPU: 22.783860 seconds
# Time on GPU: 22.248475 seconds
# Time on GPU: 22.285500 seconds
# Time on GPU: 22.320218 seconds
# Time on GPU: 22.340421 seconds
# Average time on GPU: 22.395695 seconds