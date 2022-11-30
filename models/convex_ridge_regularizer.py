import math
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
from models.multi_conv import MultiConv2d
from models.linear_spline import LinearSpline


class ConvexRidgeRegularizer(nn.Module):
    def __init__(self, channels=[1, 64], kernel_size=3, padding=1, activation_params={"activation_fn": "relu"}):
        
        super().__init__()

        self.padding = padding
        self.channels = channels

        # learnable regularization strength
        self.lmbd = nn.parameter.Parameter(data=torch.tensor(5.), requires_grad=True)
        # learnable regularization scaling
        self.mu = nn.parameter.Parameter(data=torch.tensor(1.), requires_grad=True)

        # linear layer, made of compositions of convolutions
        self.conv_layer = MultiConv2d(channels=channels, kernel_size=kernel_size, padding=padding)

        # activation functions
        self.activation_params = activation_params
        self.use_linear_spline = (activation_params["activation_fn"] == "linear_spline")

        if activation_params["activation_fn"] == "linear_spline":
            activation_params["n_channels"] = channels[-1]
            self.activation = LinearSpline(mode="conv", num_activations=channels[-1],
                                    size=activation_params["n_knots"],
                                    range_=activation_params["knots_range"],
                                    monotonic_constraint=activation_params["monotonic"],
                                    init=activation_params["spline_init"],
                                    differentiable_projection=activation_params["differentiable_projection"])
                                    
        elif activation_params["activation_fn"] == "relu":
            self.activation = nn.ReLU()
        elif activation_params["activation_fn"] == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError('Need to provide a valid activation function')

        self.num_params = sum(p.numel() for p in self.parameters())

        # initialize random image for caching an estimate of the largest eigen vector for Lipschitz bound computation
        # the size matters little compare to number of iterations, so small patches makes training more efficient
        # + the learning is carried on patches
        self.initializeEigen(size=20)
        
        # running estimate of Lipschitz
        self.L = nn.parameter.Parameter(data=torch.tensor(1.), requires_grad=False)

        print("---------------------")
        print(f"Building a CRR-NN model with \n - {channels} channels \n - {activation_params['activation_fn']} activation functions")
        if activation_params["activation_fn"] == "linear_spline":
            print(f"  ({self.activation})")
        print("---------------------")


    def initializeEigen(self, size=100):
        self.u = torch.empty((1,1,size, size)).uniform_()

    @property
    def lmbd_transformed(self):
        # ensure lmbd is nonzero positive
        return(torch.clip(self.lmbd, 0.01, None))

    @property
    def mu_transformed(self):
        # ensure mu is nonzero positive
        return(torch.clip(self.mu, 0.01, None))


    def forward(self, x):
        # linear layer (a multi-convolution)
        y = self.conv_layer(x)
        # activation
        y = self.activation(y)
        # transposed linear layer
        y = self.conv_layer.transpose(y)

        return(y)

    def grad(self, x):
        return(self.forward(x))

    def update_integrated_params(self):
        for ac in self.activation:
            ac.update_integrated_coeff()

    def cost(self, x):
        s = x.shape
        # first multi convolution layer
        y = self.conv_layer(x)
        # activation
        y = self.activation.integrate(y)

        return(torch.sum(y, dim=tuple(range(1, len(s)))))

    # regularization
    def TV2(self, include_weights=False):
        if self.activation_params["activation_fn"] == "linear_spline":
            return(self.activation.TV2())
        else:
            return(0)

    
    def precise_lipschitz_bound(self, n_iter=50, differentiable=False):
        with torch.no_grad():
            # vector with the max slope of each activation
            if self.use_linear_spline:
                slope_max = self.activation.slope_max
                if slope_max.max().item() == 0:
                    return(torch.tensor([0.], device = slope_max.device))
            # running eigen vector with largest eigen value estimate
            self.u = self.u.to(self.conv_layer.conv_layers[0].weight.device)
            u = self.u
            # power iterations
            for i in range(n_iter - 1):
                # normalization
                u = normalize(u)
                # W u
                u = self.conv_layer.convolutionNoBias(u)
                # D' W u
                if self.use_linear_spline:
                    u = u * slope_max.view(1,-1,1,1)
                # WT D' W u
                u = self.conv_layer.transpose(u)
                # norm of u
                sigma_estimate = norm(u)

        # embdding the computation in the forward
        if differentiable:
            u = normalize(u)
            # W u
            u = self.conv_layer.convolutionNoBias(u)
            # D' W u
            if self.use_linear_spline:
                slope_max = self.activation.slope_max
                u = u * slope_max.view(1,-1,1,1)
            # WT D' W u
            u = self.conv_layer.transpose(u)
            # norm of u
            sigma_estimate = norm(u)
            # update running estimate
            self.u = u
            return(sigma_estimate)
        else:
            # update running estimate
            self.u = u
            return(sigma_estimate)

    def prune(self, tol=1e-4):
        device = self.conv_layer.conv_layers[0].weight.device
        # 1. Convert multi-convolutions into single convolutions
        # 1.1 size of the single kernel
        new_padding = sum([conv.kernel_size[0]//2 for conv in self.conv_layer.conv_layers])
        new_kernel_size = 2*new_padding + 1

        # 1.2 Find new kernels <=> impulse responses
        impulse = torch.zeros((1, 1, new_kernel_size , new_kernel_size), device=device, requires_grad=False)
        impulse[0, 0, new_kernel_size//2, new_kernel_size//2] = 1

        new_kernel = self.conv_layer.convolutionNoBias(impulse)

        # 2. Determine the channels to prune, based on
        #     - impulse response magnitude
        kernel_norm = torch.sum(new_kernel**2, dim=(0, 2, 3))

        #     - TV2 of associated activation function
        coeff = self.activation.projected_coefficients
        slopes = (coeff[:,1:] - coeff[:,:-1])/self.activation.grid.item()
        tv2 = torch.sum(torch.abs(slopes[:,1:-1]), dim=1)
        
        # criterion to keep a (filter, activation) tuple
        weight = tv2 * kernel_norm

        l_keep = torch.where(weight > tol)[0]
        print("---------------------")
        print(f" PRUNNING \n Found {len(l_keep)} filters with non-vanishing potential functions")
        print("---------------------")


        # 3. Prune spline coefficients
        new_spline_coeff = torch.clone(self.activation.coefficients_vect.view(self.activation.num_activations, self.activation.size)[l_keep, :].contiguous().view(-1))
        self.activation.coefficients_vect.data = new_spline_coeff
        self.activation.num_activations = len(l_keep)

        self.activation.grid_tensor = torch.linspace(-self.activation.range_, self.activation.range_, self.activation.size).expand((self.activation.num_activations, self.activation.size))

        self.activation.init_zero_knot_indexes()

        # 4. Prune convolutions
        new_conv_layer = MultiConv2d(channels=[1, len(l_keep)], kernel_size=self.channels[-1], padding=new_padding)

        new_conv_layer.conv_layers[0].parametrizations.weight.original.data = new_kernel[:, l_keep, :, :].permute(1, 0, 2, 3)

        self.conv_layer = new_conv_layer
        self.channels = [1, len(l_keep)]
        self.padding = new_padding

        # 5. Update number of parameters
        self.num_params = sum(p.numel() for p in self.parameters())


def norm(u):
    return(torch.sqrt(torch.sum(u**2)))

def normalize(u):
    return(u/norm(u))