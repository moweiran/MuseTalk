import os
import multiprocessing

def check_system_resources():
    """Check system resources to determine optimal thread pool size"""
    
    # Get CPU information
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU cores: {cpu_count}")
    
    # For I/O bound tasks like image writing, you can typically use more threads than CPU cores
    # A common formula for I/O bound tasks:
    optimal_threads = min(cpu_count * 4, 32)  # Usually 2-4x CPU cores, with a reasonable maximum
    
    print(f"\nRecommended ThreadPoolExecutor max_workers values:")
    print(f"  For CPU-bound tasks: {cpu_count}")
    print(f"  For I/O-bound tasks (recommended): {optimal_threads}")
    print(f"  Maximum (be careful): {cpu_count * 8}")
    
    print(f"\nTo determine the best value for your system:")
    print(f"  1. Start with {optimal_threads} workers")
    print(f"  2. Test performance with different values")
    print(f"  3. Monitor CPU and disk usage during processing")
    print(f"  4. Adjust based on resource utilization and performance")
    
    print(f"\nIn the optimized code, max_workers is determined by:")
    print(f"  max_workers = min(cpu_count * 4, 32)")
    print(f"  This means it will use 4 times the number of CPU cores,")
    print(f"  but capped at 32 to prevent excessive resource usage.")

if __name__ == "__main__":
    check_system_resources()