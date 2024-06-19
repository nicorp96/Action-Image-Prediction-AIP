import torch


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
