import torch
import torch.nn.functional as F

"""Some projection can be more efficient with knowledge of the previous
largets eigenvector and by being modified towards the end"""

def normalize(tensor):
    norm = float(torch.sqrt(torch.sum(tensor * tensor)))
    norm = max(norm, 1e-10)
    normalized_tensor = tensor / norm
    return normalized_tensor



def spectral_norm_conv(weights, lipschitz_goal, additional_parameters):
    """divides the conv layer by its L2 norm"""
    kernel_size = weights.shape[2]
    padding = kernel_size //2

    u = additional_parameters['largest_eigenvector']
    if additional_parameters['end_of_epoch']: n_steps = 5
    else: n_steps = 1

    with torch.no_grad():
        for _ in range(n_steps):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            v = normalize(F.conv2d(u.flip(2,3), weights.permute(1, 0, 2, 3), padding=padding)).flip(2, 3)
            u = normalize(F.conv2d(v, weights, padding=padding))
            u = u.clone()
            v = v.clone()

    current_lipschitz = torch.sum(u * F.conv2d(v, weights, padding=padding))
    new_weights = lipschitz_goal * weights / current_lipschitz
    additional_parameters['largest_eigenvector'] = u
    return new_weights, additional_parameters