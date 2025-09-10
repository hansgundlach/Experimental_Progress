# cpu_optimization_utils.py
import multiprocessing as mp
import psutil
import torch
import os
import platform


def get_optimal_dataloader_config(
    batch_size: int = 32,
    model_type: str = "transformer",  # or "lstm"
    force_conservative: bool = False
):
    """
    Get optimal DataLoader configuration based on system resources.
    
    Args:
        batch_size: Batch size per device
        model_type: "transformer" or "lstm" 
        force_conservative: Force conservative settings for shared systems
        
    Returns:
        Dict with optimal DataLoader settings
    """
    cpu_count = mp.cpu_count()
    
    # Get available memory in GB
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Detect if we're on a shared system (common indicators)
    is_shared_system = detect_shared_system()
    
    # Base configuration
    config = {
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": True,
    }
    
    if force_conservative or is_shared_system:
        # Conservative settings for shared systems
        config.update({
            "num_workers": min(cpu_count // 4, 6),
            "prefetch_factor": 2,
        })
    else:
        # Aggressive settings for dedicated systems
        optimal_workers = calculate_optimal_workers(
            cpu_count, available_memory_gb, batch_size, model_type
        )
        config.update({
            "num_workers": optimal_workers,
            "prefetch_factor": min(8, max(2, optimal_workers // 2)),
        })
    
    # Disable workers on very small systems or when problematic
    if cpu_count <= 2 or available_memory_gb < 4:
        config.update({
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": 2,
        })
    
    return config


def calculate_optimal_workers(
    cpu_count: int, 
    memory_gb: float, 
    batch_size: int, 
    model_type: str
) -> int:
    """Calculate optimal number of workers based on system resources."""
    
    # Base formula: Use most CPUs but leave some for system/training
    base_workers = max(1, cpu_count - 2)  # Leave 2 cores for system/training
    
    # Memory-based scaling
    # Estimate memory per worker (rough heuristics)
    if model_type == "transformer":
        memory_per_worker_gb = 0.5  # Transformers need more memory for tokenization
        max_workers_by_memory = int(memory_gb / memory_per_worker_gb)
    else:  # LSTM
        memory_per_worker_gb = 0.3  # LSTMs are simpler
        max_workers_by_memory = int(memory_gb / memory_per_worker_gb)
    
    # Batch size scaling (smaller batches can use more workers)
    if batch_size >= 128:
        batch_scale = 0.8  # Large batches, fewer workers
    elif batch_size >= 64:
        batch_scale = 1.0  # Medium batches
    else:
        batch_scale = 1.2  # Small batches, more workers
    
    # Combine factors
    optimal = int(min(base_workers, max_workers_by_memory) * batch_scale)
    
    # Apply reasonable bounds
    optimal = max(1, min(optimal, cpu_count - 1))  # At least 1, at most n-1
    
    # Cap based on empirical limits (diminishing returns)
    if model_type == "transformer":
        optimal = min(optimal, 16)  # Transformers rarely benefit from >16
    else:  # LSTM
        optimal = min(optimal, 12)  # LSTMs benefit less from high parallelism
        
    return optimal


def detect_shared_system() -> bool:
    """
    Detect if we're running on a shared system where we should be conservative.
    """
    # Check for common shared system indicators
    shared_indicators = [
        # SLURM environment
        "SLURM_JOB_ID" in os.environ,
        "SLURM_PROCID" in os.environ,
        
        # PBS/Torque
        "PBS_JOBID" in os.environ,
        
        # SGE
        "JOB_ID" in os.environ and "SGE_ROOT" in os.environ,
        
        # Common HPC hostnames/patterns
        any(pattern in platform.node().lower() for pattern in [
            "login", "head", "master", "compute", "node", "gpu", 
            "supercloud", "cluster", "hpc", "slurm"
        ]),
        
        # High CPU count (likely shared)
        mp.cpu_count() > 32,
        
        # Very high memory (likely shared)
        psutil.virtual_memory().total > 100 * (1024**3),  # >100GB
    ]
    
    return any(shared_indicators)


def get_dataloader_report(config: dict, model_type: str = "transformer") -> str:
    """Generate a report of DataLoader optimization settings."""
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    available_gb = psutil.virtual_memory().available / (1024**3)
    
    report = f"""
DataLoader Optimization Report ({model_type.upper()}):
{'=' * 50}
System Resources:
  - CPU cores: {cpu_count}
  - Total memory: {memory_gb:.1f} GB
  - Available memory: {available_gb:.1f} GB
  - Shared system detected: {detect_shared_system()}

Optimized Settings:
  - num_workers: {config['num_workers']} ({config['num_workers']/cpu_count*100:.1f}% of CPUs)
  - pin_memory: {config['pin_memory']}
  - persistent_workers: {config['persistent_workers']}
  - prefetch_factor: {config['prefetch_factor']}

Expected Benefits:
  - CPU utilization: {config['num_workers']/cpu_count*100:.1f}%
  - Memory efficiency: {'High' if config['pin_memory'] else 'Standard'}
  - I/O overlap: {'Yes' if config['num_workers'] > 0 else 'No'}
"""
    return report


# Convenience functions for each model type
def get_transformer_dataloader_config(**kwargs):
    """Get optimal DataLoader config for transformers."""
    return get_optimal_dataloader_config(model_type="transformer", **kwargs)


def get_lstm_dataloader_config(**kwargs):
    """Get optimal DataLoader config for LSTMs.""" 
    return get_optimal_dataloader_config(model_type="lstm", **kwargs)