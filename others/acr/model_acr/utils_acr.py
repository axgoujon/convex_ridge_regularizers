import os
from convex_models import *

clip_fbp = True
def load_model(device):
    ### load trained models
    model_path = os.path.dirname(os.path.realpath(__file__)) + "/../pretrained_models/"

    acr = ICNN(n_in_channels=1, n_filters=n_filters, kernel_size=kernel_size, n_layers=n_layers).to(device)
    sfb = SFB(n_in_channels=1, n_kernels=10, n_filters=32).to(device)
    l2_net = L2net().to(device) 

    acr.eval()
    sfb.eval()
    l2_net.eval()

    if not clip_fbp:
        acr.load_state_dict(torch.load(model_path + "icnn.pt"))
        sfb.load_state_dict(torch.load(model_path + "sfb.pt"))
        l2_net.load_state_dict(torch.load(model_path + "l2_net.pt"))
    else:
        acr.load_state_dict(torch.load(model_path + "icnn_clipped_fbp.pt"))
        sfb.load_state_dict(torch.load(model_path + "sfb_clipped_fbp.pt"))
        l2_net.load_state_dict(torch.load(model_path + "l2_net_clipped_fbp.pt"))


    model = Wrapper(acr, sfb, l2_net)
    model.to(device)

    return model
