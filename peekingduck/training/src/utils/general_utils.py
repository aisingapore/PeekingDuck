
import torch
import gc

def free_gpu_memory(
    *args,
) -> None:
    """Delete all variables from the GPU. Clear cache.
    Args:
        model ([type], optional): [description]. Defaults to None.
        optimizer (torch.optim, optional): [description]. Defaults to None.
        scheduler (torch.optim.lr_scheduler, optional): [description]. Defaults to None.
    """

    if args is not None:
        # Delete all other variables
        # FIXME:TODO: Check my notebook on deleting global vars.
        for arg in args:
            del arg

    gc.collect()
    torch.cuda.empty_cache()