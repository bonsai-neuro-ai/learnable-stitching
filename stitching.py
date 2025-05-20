from typing import Self

import nn_lib.models.fancy_layers as fl
import torch
from nn_lib.models.utils import conv2d_shape_inverse
from torch import nn


class ConvStitchingLayer(nn.Module, fl.Regressable):

    def __init__(
        self,
        in_shape: tuple[int, int, int],
        out_shape: tuple[int, int, int],
        conv_part: fl.RegressableConv2d,
    ):
        super(ConvStitchingLayer, self).__init__()
        c_in, h_in, w_in = in_shape
        c_out, h_out, w_out = out_shape

        assert c_in == conv_part.in_channels
        assert c_out == conv_part.out_channels

        if in_shape != out_shape:
            self.maybe_resize = fl.Interpolate2d(
                size=conv2d_shape_inverse((h_out, w_out), **conv_part.conv_params),
                mode="bilinear",
            )
        else:
            self.maybe_resize = nn.Identity()

        self.conv = conv_part

    def forward(self, x):
        return self.conv(self.maybe_resize(x))

    def init_by_regression(
        self, from_data: torch.Tensor, to_data: torch.Tensor, transpose: bool = False
    ) -> Self:
        """
        Initialize the convolutional layer by regression.
        :param from_data: The input data.
        :param to_data: The target data.
        :param transpose: Whether to transpose the regression (optimize to->from).
        :return: Self.
        """
        self.conv.init_by_regression(self.maybe_resize(from_data), to_data, transpose=transpose)
        return self


def create_stitching_layer(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    stitching_family: str
) -> ConvStitchingLayer:
    """
    Create a ConvStitchingLayer.
    :param in_shape: The input shape.
    :param out_shape: The output shape.
    :param stitching_family: String name of the type of stitching layer to create.
    :return: The ConvStitchingLayer.
    """
    if stitching_family == "1x1Conv":
        conv_part = fl.RegressableConv2d(
            in_channels=in_shape[0],
            out_channels=out_shape[0],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        assert len(in_shape) == 3, "1x1Conv requires in_shape (c, h, w)"
        assert len(out_shape) == 3, "1x1Conv requires out_shape (c, h, w)"

        return ConvStitchingLayer(in_shape, out_shape, conv_part)
    else:
        raise NotImplementedError("Other stitching families still to-do")
