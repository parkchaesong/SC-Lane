import os
import pathlib
import torch


def save_model_dp(net, optimizer, save_path, name):
    """ save current models
    """
    if not os.path.exists(save_path):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(save_path, name)
    torch.save({
        "model_state": net.module.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
    }, model_path)
    print("Model saved as %s" % model_path)
