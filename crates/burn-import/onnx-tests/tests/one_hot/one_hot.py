#!/usr/bin/env python3

# used to generate model: one_hot.onnx

import torch
import torch.nn as nn
import torch.nn.functional as F

class OneHotModel(nn.Module):
    def __init__(self, axis=-1):
        super(OneHotModel, self).__init__()
        self.axis = axis

    def forward(self, indices, depth, values):
        # depth: number of classes (scalar)
        # values: [off_value, on_value]
        off_value, on_value = values[0], values[1]

        # Perform one-hot encoding
        one_hot = F.one_hot(indices, num_classes=depth)

        # Scale by on_value and add off_value
        one_hot = one_hot * (on_value - off_value) + off_value

        if self.axis != -1:
            # Rearrange the dimensions to match the desired axis
            one_hot = one_hot.permute(
                *range(self.axis),
                len(indices.shape),  # The new one-hot axis
                *range(self.axis, len(indices.shape))
            )
        return one_hot


def main():
    # Set reproducibility and precision
    torch.manual_seed(42)
    torch.set_printoptions(precision=8)

    # Initialize the model
    model = OneHotModel(axis=-1)  # Default axis = -1
    model.eval()
    device = torch.device("cpu")

    # Inputs
    indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64, device=device)  # Input indices
    depth = 4  # Number of classes
    values = torch.tensor([0.0, 1.0], device=device)  # [off_value, on_value]

    # Export to ONNX
    file_name = "one_hot.onnx"
    torch.onnx.export(model,
                      (indices, depth, values),
                      file_name,
                      input_names=["indices", "depth", "values"],
                      output_names=["output"],
                      opset_version=16,
                      dynamic_axes={"indices": {0: "batch", 1: "features"}})

    print("Finished exporting model to {}".format(file_name))

    # Output test data for verification
    output = model.forward(indices, depth, values)
    print("Test input indices:\n{}".format(indices))
    print("Test output one-hot tensor:\n{}".format(output))


if __name__ == '__main__':
    main()
