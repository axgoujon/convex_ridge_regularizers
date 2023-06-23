#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 12:37:25 2021

@author: subhadip
"""
import numpy as np
import torch
from odl import Operator
import odl



class OperatorFunction(torch.autograd.Function):

    """Wrapper of an ODL operator as a ``torch.autograd.Function``.
    """

    @staticmethod
    def forward(ctx, operator, input):
        """Evaluate forward pass on the input.
        Parameters
        ----------
        ctx : context object
            Object to communicate information between forward and backward
            passes.
        operator : `Operator`
            ODL operator to be wrapped. For gradient computations to
            work, ``operator.derivative(x).adjoint`` must be implemented.
        input : `torch.Tensor`
            Point at which to evaluate the operator.
        Returns
        -------
        result : `torch.Tensor`
            Tensor holding the result of the evaluation.
        """
        if not isinstance(operator, Operator):
            raise TypeError(
                "`operator` must be an `Operator` instance, got {!r}"
                "".format(operator)
            )

        # Save operator for backward; input only needs to be saved if
        # the operator is nonlinear (for `operator.derivative(input)`)
        ctx.operator = operator

        if not operator.is_linear:
            # Only needed for nonlinear operators
            ctx.save_for_backward(input)

        # TODO(kohr-h): use GPU memory directly when possible
        # TODO(kohr-h): remove `copy_if_zero_strides` when NumPy 1.16.0
        # is required
        input_arr = copy_if_zero_strides(input.cpu().detach().numpy())

        # Determine how to loop over extra shape "left" of the operator
        # domain shape
        in_shape = input_arr.shape
        op_in_shape = operator.domain.shape
        if operator.is_functional:
            op_out_shape = ()
            op_out_dtype = operator.domain.dtype
        else:
            op_out_shape = operator.range.shape
            op_out_dtype = operator.range.dtype

        extra_shape = in_shape[:-len(op_in_shape)]
        if in_shape[-len(op_in_shape):] != op_in_shape:
            shp_str = str(op_in_shape).strip('(,)')
            raise ValueError(
                'input tensor has wrong shape: expected (*, {}), got {}'
                ''.format(shp_str, in_shape)
            )

        # Store some information on the context object
        ctx.op_in_shape = op_in_shape
        ctx.op_out_shape = op_out_shape
        ctx.extra_shape = extra_shape
        ctx.op_in_dtype = operator.domain.dtype
        ctx.op_out_dtype = op_out_dtype

        # Evaluate the operator on all inputs in a loop
        if extra_shape:
            # Multiple inputs: flatten extra axes, then do one entry at a time
            input_arr_flat_extra = input_arr.reshape((-1,) + op_in_shape)
            results = []
            for inp in input_arr_flat_extra:
                results.append(operator(inp))

            # Stack results, reshape to the expected output shape and enforce
            # correct dtype
            result_arr = np.stack(results).astype(op_out_dtype, copy=False)
            result_arr = result_arr.reshape(extra_shape + op_out_shape)
        else:
            # Single input: evaluate directly
            result_arr = np.asarray(
                operator(input_arr)
            ).astype(op_out_dtype, copy=False)

        # Convert back to tensor
        tensor = torch.from_numpy(result_arr).to(input.device)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        
        # Return early if there's nothing to do
        if not ctx.needs_input_grad[1]:
            return None, None

        operator = ctx.operator

        # Get `operator` and `input` from the context object (the input
        # is only needed for nonlinear operators)
        if not operator.is_linear:
            # TODO: implement directly for GPU data
            # TODO(kohr-h): remove `copy_if_zero_strides` when NumPy 1.16.0
            # is required
            input_arr = copy_if_zero_strides(
                ctx.saved_tensors[0].detach().cpu().numpy()
            )

        # ODL weights spaces, pytorch doesn't, so we need to handle this
        try:
            dom_weight = operator.domain.weighting.const
        except AttributeError:
            dom_weight = 1.0
        try:
            ran_weight = operator.range.weighting.const
        except AttributeError:
            ran_weight = 1.0
        scaling = dom_weight / ran_weight

        # Convert `grad_output` to NumPy array
        grad_output_arr = copy_if_zero_strides(
            grad_output.detach().cpu().numpy()
        )

        # Get shape information from the context object
        op_in_shape = ctx.op_in_shape
        op_out_shape = ctx.op_out_shape
        extra_shape = ctx.extra_shape
        op_in_dtype = ctx.op_in_dtype

        # Check if `grad_output` is consistent with `extra_shape` and
        # `op_out_shape`
        if grad_output_arr.shape != extra_shape + op_out_shape:
            raise ValueError(
                'expected tensor of shape {}, got shape {}'
                ''.format(extra_shape + op_out_shape, grad_output_arr.shape)
            )

        # Evaluate the (derivative) adjoint on all inputs in a loop
        if extra_shape:
            # Multiple gradients: flatten extra axes, then do one entry
            # at a time
            grad_output_arr_flat_extra = grad_output_arr.reshape(
                (-1,) + op_out_shape
            )

            results = []
            if operator.is_linear:
                for ograd in grad_output_arr_flat_extra:
                    results.append(np.asarray(operator.adjoint(ograd)))
            else:
                # Need inputs, flattened in the same way as the gradients
                input_arr_flat_extra = input_arr.reshape((-1,) + op_in_shape)
                for ograd, inp in zip(
                    grad_output_arr_flat_extra, input_arr_flat_extra
                ):
                    results.append(
                        np.asarray(operator.derivative(inp).adjoint(ograd))
                    )

            # Stack results, reshape to the expected output shape and enforce
            # correct dtype
            result_arr = np.stack(results).astype(op_in_dtype, copy=False)
            result_arr = result_arr.reshape(extra_shape + op_in_shape)
        else:
            # Single gradient: evaluate directly
            if operator.is_linear:
                result_arr = np.asarray(
                    operator.adjoint(grad_output_arr)
                ).astype(op_in_dtype, copy=False)
            else:
                result_arr = np.asarray(
                    operator.derivative(input_arr).adjoint(grad_output_arr)
                ).astype(op_in_dtype, copy=False)

        # Apply scaling, convert to tensor and return
        if scaling != 1.0:
            result_arr *= scaling
        grad_input = torch.from_numpy(result_arr).to(grad_output.device)
        return None, grad_input  # return `None` for the `operator` part


class OperatorModule(torch.nn.Module):

    """Wrapper of an ODL operator as a ``torch.nn.Module``.
    """

    def __init__(self, operator):
        """Initialize a new instance."""
        super(OperatorModule, self).__init__()
        self.operator = operator

    def forward(self, x):
        """Compute forward-pass of this module on ``x``.
        """
        in_shape = tuple(x.shape)
        in_ndim = len(in_shape)
        op_in_shape = self.operator.domain.shape
        op_in_ndim = len(op_in_shape)
        if in_ndim <= op_in_ndim or in_shape[-op_in_ndim:] != op_in_shape:
            shp_str = str(op_in_shape).strip('()')
            raise ValueError(
                'input tensor has wrong shape: expected (N, *, {}), got {}'
                ''.format(shp_str, in_shape)
            )
        return OperatorFunction.apply(self.operator, x)

    def __repr__(self):
        """Return ``repr(self)``."""
        op_name = self.operator.__class__.__name__
        op_in_shape = self.operator.domain.shape
        if len(op_in_shape) == 1:
            op_in_shape = op_in_shape[0]
        op_out_shape = self.operator.range.shape
        if len(op_out_shape) == 1:
            op_out_shape = op_out_shape[0]

        return '{}({}) ({} -> {})'.format(
            self.__class__.__name__, op_name, op_in_shape, op_out_shape
        )


def copy_if_zero_strides(arr):
    """Workaround for NumPy issue #9165 with 0 in arr.strides."""
    assert isinstance(arr, np.ndarray)
    return arr.copy() if 0 in arr.strides else arr

