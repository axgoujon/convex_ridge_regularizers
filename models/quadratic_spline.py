import torch
from torch import nn


class Quadratic_Spline_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """
    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, zero_knot_indexes, size):

        x_clamped = x.clamp(min=-(grid.item() * (size // 2)),
                            max=(grid.item() * (size // 2 - 2)))

        floored_x = torch.floor(x_clamped / grid)  # left coefficient


        
        # This gives the indexes (in coefficients_vect) of the left
        # coefficients
        indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()
        # B-Splines evaluation
        shift1 = (x - floored_x*grid)/grid


        frac1 = ((shift1 - 1)**2)/2
        frac2 = (-2*(shift1)**2 + 2*shift1 + 1)/2 
        frac3 = (shift1)**2/2



        activation_output = coefficients_vect[indexes + 2] * frac3 + \
            coefficients_vect[indexes + 1] * frac2 + \
            coefficients_vect[indexes] * frac1


        grad_x = coefficients_vect[indexes + 2] * (shift1) + \
            coefficients_vect[indexes + 1] * (1 - 2*shift1) + \
            coefficients_vect[indexes] * ((shift1 - 1))

        grad_x = grad_x / grid

        ctx.save_for_backward(grad_x, frac1, frac2, frac3, coefficients_vect, indexes, grid)
        
        return activation_output

    @staticmethod
    def backward(ctx, grad_out):

        grad_x, frac1, frac2, frac3, coefficients_vect, indexes, grid = ctx.saved_tensors

        grad_x = grad_x * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).

        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        # coefficients gradients
        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1) + 2,
                                            (frac3 * grad_out).view(-1))

        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1) + 1,
                                            (frac2 * grad_out).view(-1))

        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1),
                                            (frac1 * grad_out).view(-1))
       

        return grad_x, grad_coefficients_vect, None, None, None, None



class quadratic_spline(nn.Module):
    '''
    
    '''
    def __init__(self, n_channels, n_knots, T):
        '''

        '''
        super().__init__()
        self.n_knots = n_knots
        self.n_channels = n_channels
        self.T = self.spline_grid_from_range(n_knots,T)
        print("grid step ",self.T.item())
        # initialize the coefficients
        self.coefficients = torch.zeros((n_channels,n_knots))
        
        self.coefficients.requiresGrad = False
        activation_arange = torch.arange(0, self.n_channels)
        self.zero_knot_indexes = (activation_arange * self.n_knots +
                                  (self.n_knots // 2))
        self.mode = "conv"
        

    def forward(self, input):
        """
        Args:
            input (torch.Tensor):
                2D or 4D, depending on weather the layer is
                convolutional ('conv') or fully-connected ('fc')

        Returns:
            output (torch.Tensor)
        """
        input_size = input.size()

        x = input


        assert x.size(1) == self.n_channels, \
            f'{input.size(1)} != {self.n_channels}.'
        self.coefficients_vect = self.coefficients.view(-1)
        grid = self.T.to(self.coefficients_vect.device)
        zero_knot_indexes = self.zero_knot_indexes.to(grid.device)

        coefficients_proj_vect = self.coefficients.view(-1)

            

        output = Quadratic_Spline_Func.apply(x, coefficients_proj_vect, grid,
                                        zero_knot_indexes, self.n_knots)

        return output


    def spline_grid_from_range(self,spline_size, spline_range, round_to=1e-6):
        """
        Compute spline grid spacing from desired one-sided range
        and the number of activation coefficients.

        Args:
            spline_size (odd int):
                number of spline coefficients
            spline_range (float):
                one-side range of spline expansion.
            round_to (float):
                round grid to this value
        """
        if int(spline_size) % 2 == 0:
            raise TypeError('size should be an odd number.')
        if float(spline_range) <= 0:
            raise TypeError('spline_range needs to be a positive float...')

        spline_grid = ((float(spline_range) /
                        (int(spline_size) // 2)) // round_to) * round_to

        return torch.tensor(spline_grid)
