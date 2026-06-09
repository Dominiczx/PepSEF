import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from typing import Tuple


def set_args_device(args, device: torch.device):
    """Ensure args has a `device` attribute usable by existing validation utilities."""
    args.device = device


def prepare_dataloaders_for_ddp(train_dataset, val_dataset, test_dataset, batch_size: int, world_size: int, rank: int, num_workers_train: int = 4, num_workers_val: int = 2):
    """Create DistributedSamplers and DataLoaders for DDP workers.

    Returns (train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler)
    """
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers_val)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers_val)
    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler


def run_validate_on_device(args, dataloader, model, phase='val', save_csv=False, device: torch.device = None):
    """Run the existing `validate` routine in a device-aware way.

    - Temporarily ensures `args.device` is set to `device` (or existing args.device).
    - Unwraps `DistributedDataParallel` to pass the underlying module to validation if necessary.
    """
    from utils.validation import validate

    if device is None:
        device = getattr(args, 'device', torch.device('cpu'))
    # set args.device for validate
    args.device = device

    # Unwrap DDP if wrapped
    if isinstance(model, DistributedDataParallel):
        core_model = model.module
    else:
        core_model = model

    return validate(args, dataloader, core_model, phase=phase, save_csv=save_csv)
