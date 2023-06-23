import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractproperty, abstractmethod
from models.quadratic_spline import Quadratic_Spline_Func


def monotonic_clipping(cs):
    """Simple projection of the spline coefficients to obtain a monotonic linear spline"""
    device = cs.device
    n = cs.shape[1]
    # get the projected slopes
    new_slopes = torch.clamp(cs[:,1:] - cs[:,:-1], 0, None)
    # clamping extension
    new_slopes[:,0] = 0
    new_slopes[:,-1] = 0
    
    # build new coefficients
    new_cs = torch.zeros(cs.shape, device=device)
    new_cs[:,1:] = torch.cumsum(new_slopes, dim=1)

    # set zero at zero zero
    new_cs = new_cs + (-new_cs[:,new_cs.shape[1]//2]).unsqueeze(1)

    return new_cs

def initialize_coeffs(init, grid_tensor, grid):
        """The coefficients are initialized with the value of the activation
        # at each knot (c[k] = f[k], since B1 splines are interpolators)."""
        
        if init == 'identity':
            coefficients = grid_tensor
        elif init == 'zero':
            coefficients = grid_tensor*0
        elif init == 'relu':
            coefficients = F.relu(grid_tensor)       
        else:
            raise ValueError('init should be in [identity, relu, absolute_value, maxmin, max_tv].')
        
        return coefficients

class LinearSpline_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """
    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, zero_knot_indexes, size, even, train=True):

        # The value of the spline at any x is a combination 
        # of at most two coefficients
        max_range = (grid.item() * (size // 2 - 1))
        if even:
            x = x - grid / 2
            max_range = (grid.item() * (size // 2 - 2))
        x_clamped = x.clamp(min=-(grid.item() * (size // 2)), max=max_range)

        floored_x = torch.floor(x_clamped / grid)  #left coefficient
        #fracs = x_clamped / grid - floored_x
        fracs = x / grid - floored_x  # distance to left coefficient

        # This gives the indexes (in coefficients_vect) of the left
        # coefficients
        indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()

        # Only two B-spline basis functions are required to compute the output
        # (through linear interpolation) for each input in the B-spline range.
        activation_output = coefficients_vect[indexes + 1] * fracs + \
            coefficients_vect[indexes] * (1 - fracs)
        if even:
            activation_output = activation_output + grid / 2

        ctx.save_for_backward(fracs, coefficients_vect, indexes, grid)
        ctx.results = (fracs, coefficients_vect, indexes, grid)
        return activation_output

    @staticmethod
    def backward(ctx, grad_out):

        fracs, coefficients_vect, indexes, grid = ctx.saved_tensors
        grad_x = (coefficients_vect[indexes + 1] -
                  coefficients_vect[indexes]) / grid * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).
        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        # right coefficients gradients
        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1) + 1,
                                            (fracs * grad_out).view(-1))
        # left coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
                                            ((1 - fracs) * grad_out).view(-1))

        return grad_x, grad_coefficients_vect, None, None, None, None

class LinearSplineDerivative_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """
    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, zero_knot_indexes, size, even):

        # The value of the spline at any x is a combination 
        # of at most two coefficients
        max_range = (grid.item() * (size // 2 - 1))
        if even:
            x = x - grid / 2
            max_range = (grid.item() * (size // 2 - 2))
        x_clamped = x.clamp(min=-(grid.item() * (size // 2)), max=max_range)

        floored_x = torch.floor(x_clamped / grid)  #left coefficient
        #fracs = x_clamped / grid - floored_x
        fracs = x / grid - floored_x  # distance to left coefficient

        # This gives the indexes (in coefficients_vect) of the left
        # coefficients
        indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()

        # Only two B-spline basis functions are required to compute the output
        # (through linear interpolation) for each input in the B-spline range.
        activation_output = (coefficients_vect[indexes + 1] - coefficients_vect[indexes]) / grid.item()
        if even:
            activation_output = activation_output + grid / 2

        ctx.save_for_backward(fracs, coefficients_vect, indexes, grid)
        return activation_output

    @staticmethod
    def backward(ctx, grad_out):

        fracs, coefficients_vect, indexes, grid = ctx.saved_tensors
        grad_x = 0 * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).
        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        # right coefficients gradients
        #grad_coefficients_vect.scatter_add_(0,
        #                                    indexes.view(-1) + 1,
        #                                    (fracs * grad_out).view(-1))
        # left coefficients gradients
        #grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
        #                                    ((1 - fracs) * grad_out).view(-1))

        return grad_x, grad_coefficients_vect, None, None, None, None


class LinearSpline(ABC, nn.Module):
    """
    Class for LinearSpline activation functions

    Args:
        mode (str): 'conv' (convolutional) or 'fc' (fully-connected).
        num_activations (int) : number of activation functions
        size (int): number of coefficients of spline grid; the number of knots K = size - 2.
        range_ (float) : positive range of the B-spline expansion. B-splines range = [-range_, range_].
        init (str): Function to initialize activations as ('relu', 'identity', 'zero').
        monotonic_constraint (bool): Constrain the actiation to be monotonic increasing
    """

    def __init__(self, mode, num_activations, size, range_, init="zero", monotonic_constraint=True, **kwargs):

        if mode not in ['conv', 'fc']:
            raise ValueError('Mode should be either "conv" or "fc".')
        if int(num_activations) < 1:
            raise TypeError('num_activations needs to be a '
                            'positive integer...')

        super().__init__()
        self.mode = mode
        self.size = int(size)
        self.even = self.size % 2 == 0
        self.num_activations = int(num_activations)
        self.init = init
        self.range_ = float(range_)
        grid = 2 * self.range_ / float(self.size-1)
        self.grid = torch.Tensor([grid])

        self.init_zero_knot_indexes()
        self.D2_filter = Tensor([1, -2, 1]).view(1, 1, 3).div(self.grid)
        self.monotonic_constraint = monotonic_constraint

        self.integrated_coeff = None

        # tensor with locations of spline coefficients
        self.grid_tensor = torch.linspace(-self.range_, self.range_, self.size).expand((self.num_activations, self.size))
        coefficients = initialize_coeffs(init, self.grid_tensor, self.grid)  # spline coefficients
        # Need to vectorize coefficients to perform specific operations
        # size: (num_activations*size)
        self.coefficients_vect = nn.Parameter(coefficients.contiguous().view(-1))


    def init_zero_knot_indexes(self):
        """ Initialize indexes of zero knots of each activation.
        """
        # self.zero_knot_indexes[i] gives index of knot 0 for filter/neuron_i.
        # size: (num_activations,)
        activation_arange = torch.arange(0, self.num_activations)
        self.zero_knot_indexes = (activation_arange * self.size +
                                  (self.size // 2))

    @property
    def coefficients(self):
        """ B-spline coefficients. """
        return self.coefficients_vect.view(self.num_activations, self.size)

    @property
    def projected_coefficients(self):
        """ B-spline coefficients projected to meet the constraint. """
        if self.monotonic_constraint:
            return self.monotonic_coefficients
        else:
            return self.coefficients

    @property
    def projected_coefficients_vect(self):
        """ B-spline coefficients projected to meet the constraint. """
        return self.projected_coefficients.contiguous().view(-1)

    @property
    def monotonic_coefficients(self):
        """Projection of B-spline coefficients such that the spline is increasing"""
        return monotonic_clipping(self.coefficients)

    @property
    def relu_slopes(self):
        """ Get the activation relu slopes {a_k},
        by doing a valid convolution of the coefficients {c_k}
        with the second-order finite-difference filter [1,-2,1].
        """
        D2_filter = self.D2_filter.to(device=self.coefficients.device)

        coeff = self.projected_coefficients

        slopes = F.conv1d(coeff.unsqueeze(1), D2_filter).squeeze(1)
        return slopes
    
    @property
    def monotonic_coefficients_vect(self):
        """Projection of B-spline coefficients such that they are increasing"""
        return self.monotonic_coefficients.contiguous().view(-1)


    def cache_constraint(self):
        """ Update the coeffcients to the constrained one, for post training """
        if self.monotonic_constraint:
            with torch.no_grad():
                self.coefficients_vect.data = self.monotonic_coefficients_vect.data
                self.monotonic_constraint = False


    def forward(self, x):
        """
        Args:
            input (torch.Tensor):
                2D or 4D, depending on weather the layer is
                convolutional ('conv') or fully-connected ('fc')

        Returns:
            output (torch.Tensor)
        """

        grid = self.grid.to(self.coefficients_vect.device)
        zero_knot_indexes = self.zero_knot_indexes.to(grid.device)

        coeff_vect = self.projected_coefficients_vect
        
        x = LinearSpline_Func.apply(x, coeff_vect, grid, zero_knot_indexes, \
                                        self.size, self.even)

        return x

    def derivative(self, x):
        """
        Args:
            input (torch.Tensor):
                2D or 4D, depending on weather the layer is
                convolutional ('conv') or fully-connected ('fc')

        Returns:
            output (torch.Tensor)
        """
        assert x.size(1) == self.num_activations, \
            'Wrong shape of input: {} != {}.'.format(input.size(1), self.num_activations)

        grid = self.grid.to(self.coefficients_vect.device)
        zero_knot_indexes = self.zero_knot_indexes.to(grid.device)

        coeff_vect = self.projected_coefficients_vect

        x = LinearSplineDerivative_Func.apply(x, coeff_vect, grid, zero_knot_indexes, \
                                        self.size, self.even)

        return x


    def update_integrated_coeff(self):
        print("-----------------------")
        print("Updating spline coefficients for the reg cost\n (the gradient-step model is trained and intergration is required to compute the regularization cost)")
        print("-----------------------")
        coeff = self.projected_coefficients
        
        
        # extrapolate assuming zero slopes of the linear spline at both ends
        coeff_int = torch.cat((coeff[:, 0:1], coeff, coeff[:, -1:]), dim=1)

        # integrate to obtain
        # the coefficents of the corresponding quadratic BSpline expansion
        self.integrated_coeff = torch.cumsum(coeff_int, dim = 1)*self.grid.to(coeff.device)
        
        # impose 0 at 0 and reshape
        # this is arbitray, as integration is up to a constant
        self.integrated_coeff = (self.integrated_coeff - self.integrated_coeff[:, (self.size + 2)//2].view(-1,1)).view(-1)

        # store once for all knots indexes
        # not the same as for the linear-spline as we have 2 more knots
        self.zero_knot_indexes_integrated = (torch.arange(0, self.num_activations) * (self.size + 2) +
                                  ((self.size + 2) // 2))

    def integrate(self, x):
        if self.integrated_coeff is None:
            self.update_integrated_coeff()

        if x.device != self.integrated_coeff.device:
            self.integrated_coeff = self.integrated_coeff.to(x.device)

        x = Quadratic_Spline_Func.apply(x - self.grid.to(x.device), self.integrated_coeff, self.grid.to(x.device), self.zero_knot_indexes_integrated.to(x.device), (self.size + 2))

        

        return(x)

    def extra_repr(self):
        """ repr for print(model) """

        s = ('mode={mode}, num_activations={num_activations}, '
             'init={init}, size={size}, grid={grid[0]:.3f}, '
             'monotonic_constraint={monotonic_constraint}.')

        return s.format(**self.__dict__)


    def TV2(self, ignore_tails=False, **kwargs):
        """
        Computes the second-order total-variation regularization.

        deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        The regularization term applied to this function is:
        TV(2)(deepsline) = ||a||_1.
        """
        if ignore_tails:
            return torch.sum(self.relu_slopes[:,1:-1].norm(1, dim=1))
        else:
            sl = self.relu_slopes
            return torch.sum(sl.norm(1, dim=1))


    def TV2_vec(self, ignore_tails=False,p=1, **kwargs):
        """
        Computes the second-order total-variation regularization.

        deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        The regularization term applied to this function is:
        TV(2)(deepsline) = ||a||_1.
        """
        if ignore_tails:
            return torch.sum(self.relu_slopes[:,1:-1].norm(p, dim=1))
        else:
            return self.relu_slopes.norm(1, dim=1)

    @property
    def slope_max(self):
        coeff = self.projected_coefficients
        slope = (coeff[:,1:] - coeff[:,:-1])/self.grid.item()
        slope_max = torch.max(slope, dim=1)[0]
        return(slope_max)

    # tranform the splines into clip functions
    def get_clip_equivalent(self):
        """ Express the splines as sum of two ReLUs
         Only relevant for splines that look alike the cpli function """
        coeff_proj = self.projected_coefficients.clone().detach()
        slopes = (coeff_proj[:,1:] - coeff_proj[:,:-1])
        slopes_change = slopes[:,1:] - slopes[:,:-1]

        i1 = torch.max(slopes_change, dim=1)
        i2 = torch.min(slopes_change, dim=1)

        i0 = torch.arange(0, coeff_proj.shape[0]).to(coeff_proj.device)

        self.grid_tensor = self.grid_tensor.to(coeff_proj.device)
        x1 = self.grid_tensor[i0, i1[1] + 1].view(1,-1,1,1)
        y1 = coeff_proj[i0, i1[1] + 1].view(1,-1,1,1)

        x2 = self.grid_tensor[i0, i2[1] + 1].view(1,-1,1,1)
        y2 = coeff_proj[i0, i2[1] + 1].view(1,-1,1,1)

        slopes = ((y2 - y1)/(x2 - x1)).view(1,-1,1,1)

        cl = clip_activation(x1, x2, y1, slopes)

        return(cl)


class clip_activation(nn.Module):
    def __init__(self, x1, x2, y1, slopes):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.slopes = torch.nan_to_num(slopes)
        self.y1 = y1

    def forward(self, x):
        return(self.slopes * (torch.relu(x - self.x1) - torch.relu(x - self.x2)) + self.y1)

    def integrate(self, x):
        return(self.slopes/2 * ((torch.relu(x - self.x1)**2 - torch.relu(x - self.x2)**2) + self.y1 * x))
    
    @property
    def slope_max(self):
        slope_max = torch.max(self.slopes, dim=1)[0]
        return(slope_max)
    
    


