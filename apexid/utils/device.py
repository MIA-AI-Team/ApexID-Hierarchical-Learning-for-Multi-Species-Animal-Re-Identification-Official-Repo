from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    device_ids: list[int]
    cuda_available: bool
    cuda_device_count: int
    device_names: list[str]

    @property
    def is_multi_gpu(self):
        return len(self.device_ids) > 1


def get_cuda_device_ids():
    """Return all visible CUDA device ids."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def get_device(prefer_gpu=True, gpu_index=0):
    """Return the PyTorch device to use for training/inference."""
    if prefer_gpu and torch.cuda.is_available():
        if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested CUDA device {gpu_index}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
        return torch.device(f"cuda:{gpu_index}")
    return torch.device("cpu")


def get_device_info(prefer_gpu=True, gpu_index=0):
    """Return selected device plus visible CUDA metadata."""
    device_ids = get_cuda_device_ids()
    device_names = [torch.cuda.get_device_name(device_id) for device_id in device_ids]
    return DeviceInfo(
        device=get_device(prefer_gpu=prefer_gpu, gpu_index=gpu_index),
        device_ids=device_ids,
        cuda_available=torch.cuda.is_available(),
        cuda_device_count=len(device_ids),
        device_names=device_names,
    )


def print_device_info(prefer_gpu=True, gpu_index=0):
    """Print a concise device summary and return ``DeviceInfo``."""
    info = get_device_info(prefer_gpu=prefer_gpu, gpu_index=gpu_index)
    print(f"Selected device: {info.device}")
    if info.cuda_available:
        for device_id, name in zip(info.device_ids, info.device_names):
            print(f"CUDA:{device_id} - {name}")
    else:
        print("CUDA is not available. Using CPU.")
    return info
