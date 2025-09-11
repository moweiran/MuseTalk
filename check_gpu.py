
import torch

def select_gpu():
    if not torch.cuda.is_available():
        print("CUDA is not available, using CPU")
        return "cpu"
    
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s):")
    
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {name} (Free: {free_mem/1024**3:.1f}GB / Total: {total_mem/1024**3:.1f}GB)")
    
    # Select GPU 1 if available, otherwise GPU 0, otherwise CPU
    if gpu_count > 1:
        selected_gpu = 1  # Your preference
    else:
        selected_gpu = 0
    
    print(f"Selected GPU: {selected_gpu}")
    return selected_gpu

if __name__ == "__main__":
    selected_gpu = select_gpu()
    print(f"Using GPU: {selected_gpu}")