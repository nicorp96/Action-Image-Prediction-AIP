import torch
from collections import OrderedDict


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def unnormilize_action_torch(
    actions,
    pos_range=None,
    method="min-max",
):
    positions = actions[:, :3].clone()  # Clone to prevent modifying original data
    euler_angles = actions[:, 3:].clone()

    if method == "min-max":
        if pos_range is None:
            pos_min, pos_max = positions.min(dim=0)[0], positions.max(dim=0)[0]
        else:
            pos_min, pos_max = pos_range

        positions = (positions + 1) / 2  # De-normalize from [-1, 1] to [0, 1]
        positions = positions * (pos_max - pos_min + 1e-6) + pos_min

    # Concatenate positions and euler_angles back together
    unnorm_actions = torch.cat((positions, euler_angles), dim=1)

    return unnorm_actions


def unnormilize_action_seq__torch(
    actions,
    pos_range=None,
    method="min-max",
):
    positions = actions[:, :, :3].clone()  # Clone to prevent modifying original data
    euler_angles = actions[:, :, 3:6].clone()
    gripper = actions[:, :, 6:7]

    if method == "min-max":
        if pos_range is None:
            pos_min, pos_max = positions.min(dim=0)[0], positions.max(dim=0)[0]
        else:
            pos_min, pos_max = pos_range

        positions = (positions + 1) / 2  # De-normalize from [-1, 1] to [0, 1]
        positions = positions * (pos_max - pos_min + 1e-6) + pos_min

    # Concatenate positions and euler_angles back together
    unnorm_actions = torch.cat((positions, euler_angles, gripper), dim=1)

    return unnorm_actions


class NormalizeVideo:
    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return clip.float() / 255.0

    def __repr__(self) -> str:
        return self.__class__.__name__
