# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause
import typing as ty


class ReshapeError(Exception):
    """Raised when new port shape is incompatible with old shape."""

    def __init__(self, old_shape: ty.Tuple, new_shape: ty.Tuple):
        msg = (
            "Cannot reshape 'old_shape'={} to 'new_shape={}. "
            "Total number of elements must not change during "
            "reshaping.".format(old_shape, new_shape)
        )
        super().__init__(msg)


class DuplicateConnectionError(Exception):
    """Raised when an attempt is made to create more than one connection
    between source and destination port."""

    def __init__(self):
        super().__init__(self, "Cannot connect same ports twice.")


class ConcatShapeError(Exception):
    """Raised when incompatible ports are tried to be concatenated."""

    def __init__(self, shapes, axis):
        msg = (
            "Shapes {} outside of concatenation axis {} of ports to "
            "be concatenated are incompatible.".format(shapes, axis)
        )
        super().__init__(self, msg)


class VarNotSharableError(Exception):
    """Raised when an attempt is made to connect a RefPort or VarPort to a
    non-sharable Var."""

    def __init__(self, var_name: str):
        msg = (
            "Var '{}' is not shareable. Cannot connect RefPort or "
            "VarPort.".format(var_name)
        )
        super().__init__(msg)
