#!/usr/bin/env python3

# used to generate model: one_hot.onnx
#!/usr/bin/env python3

import torch
import torch.nn as nn


class OneHotModel(nn.Module):
    def __init__(self, axis=-1):
        super(OneHotModel, self).__init__()
        self.axis = axis

    def forward(self, indices, depth, values):
        off_value, on_value = values[0], values[1]

        # Compute one-hot tensor
        one_hot = torch.nn.functional.one_hot(
            indices, num_classes=depth
        ).to(dtype=torch.float32)

        # Apply on_value and off_value
        one_hot = one_hot * (on_value - off_value) + off_value

        # Permute dimensions if axis is not -1
        if self.axis != -1:
            rank = len(indices.shape) + 1
            axis = self.axis if self.axis >= 0 else rank + self.axis
            one_hot = one_hot.permute(
                *range(0, axis), rank - 1, *range(axis, rank - 1)
            )
        return one_hot


def main():
    # Set reproducibility
    torch.manual_seed(42)

    # Initialize model
    axis = -1  # Change axis if needed
    model = OneHotModel(axis=axis)
    model.eval()

    # Test inputs
    indices = torch.tensor([0, 2, 1, 4], dtype=torch.int64)  # Example indices
    depth = 6
    values = torch.tensor([0.0, 1.0], dtype=torch.float32)  # [off_value, on_value]

    # Export the model to ONNX
    file_name = "one_hot.onnx"
    torch.onnx.export(
        model,
        (indices, depth, values),
        file_name,
        input_names=["indices", "depth", "values"],
        output_names=["output"],
        opset_version=16,
    )

    # Test model output
    output = model(indices, depth, values)
    print("Test input (indices):", indices)
    print("Test input (depth):", depth)
    print("Test input (values):", values)
    print("Model output shape:", output.shape)
    print("Model output:", output)
    print(f"Model exported to {file_name}")


if __name__ == "__main__":
    main()
