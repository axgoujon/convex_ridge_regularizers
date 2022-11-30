import json
from models.linear_spline import LinearSpline

def get_spline_coefficients(model):
    coeffs_list = []
    for module in model.modules():
        if isinstance(module, LinearSpline):
            coeffs_list.append(module.coefficients_vect)
    return coeffs_list
    
def get_spline_scaling_coeffs(model):
    coeffs_list = []
    for module in model.modules():
        if isinstance(module, LinearSpline):
            coeffs_list.append(module.scaling_coeffs_vect)
    return coeffs_list

def get_spline_bn_coeffs(model):
    coeffs_list = []
    for module in model.modules():
        if isinstance(module, LinearSpline):
            coeffs_list.append(module.gamma)
            coeffs_list.append(module.beta)
    return coeffs_list

def get_no_spline_coefficients(model):
    coeffs_list = set(model.parameters())
    coeffs_list = coeffs_list - set(get_spline_coefficients(model)) - set(get_spline_scaling_coeffs(model))
    coeffs_list = list(coeffs_list)
    return coeffs_list