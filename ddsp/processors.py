# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch implementation of DDSP following closely the original code
# https://github.com/magenta/ddsp/blob/8955dc683e7f6aa03dd9fcb16987c1f8d20ae33f/ddsp/processors.py

import torch
import torch.nn as nn


# Processor Base Class ---------------------------------------------------------
class Processor(nn.Module):
    """Abstract base class for signal processors.
    Since most effects / synths require specificly formatted control signals
    (such as amplitudes and frequenices), each processor implements a
    get_controls(inputs) method, where inputs are a variable number of tensor
    arguments that are typically neural network outputs. Check each child class
    for the class-specific arguments it expects. This gives a dictionary of
    controls that can then be passed to get_signal(controls). The
    get_outputs(inputs) method calls both in succession and returns a nested
    output dictionary with all controls and signals.
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        """Convert input tensors arguments into a signal tensor."""

        controls = self.get_controls(*args, **kwargs)
        signal = self.get_signal(**controls)
        return signal

    def get_controls(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> dict:
        """Convert input tensor arguments into a dict of processor controls."""
        raise NotImplementedError

    def get_signal(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        """Convert control tensors into a signal tensor."""
        raise NotImplementedError
