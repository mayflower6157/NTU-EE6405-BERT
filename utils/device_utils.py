import torch
from loguru import logger


def get_device(gpu_id: int = 0) -> torch.device:
    """
    Get the appropriate device (CPU/GPU/MPS) for running computations.

    Args:
        gpu_id (int): GPU device ID to use if available. Defaults to 0.

    Returns:
        torch.device: Selected device object
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.success("Using MPS (Apple Silicon)")
        return device

    cuda_available = torch.cuda.is_available()
    valid_gpu_id = (
        isinstance(gpu_id, int)
        and gpu_id >= 0
        and cuda_available
        and gpu_id < torch.cuda.device_count()
    )

    if valid_gpu_id:
        try:
            device = torch.device(f"cuda:{gpu_id}")
            logger.success(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
            return device
        except Exception as exc:
            logger.warning(
                f"Failed to use cuda:{gpu_id} ({exc}); falling back to default CUDA device"
            )
            # Fall through to default CUDA selection below

    if cuda_available:
        if not valid_gpu_id and torch.cuda.device_count() > 0:
            logger.warning(
                f"GPU id {gpu_id} unavailable; defaulting to cuda:0"
            )
        device = torch.device("cuda")
        logger.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device

    device = torch.device("cpu")
    logger.success("Using CPU")
    return device


def move_to_device(data, device: torch.device):
    """
    Move data to specified device.

    Args:
        data: Input data (can be tensor, list, tuple, or dict)
        device: Target device to move data to

    Returns:
        Data moved to specified device
    """
    if isinstance(data, (list, tuple)):
        return [move_to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def get_memory_usage(device: torch.device) -> None:
    """
    Get current GPU memory usage if using CUDA device.

    Args:
        device: Current device

    Returns:
        str: Memory usage information
    """
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        cached = torch.cuda.memory_reserved(device) / 1024**2
        logger.info(
            f"GPU Memory: {allocated:.2f}MB allocated, {cached:.2f}MB cached | Logging this information"
        )
    elif device.type == "mps":
        logger.info("Memory usage tracking not available for MPS device")
    else:
        logger.info("Running on CPU")
