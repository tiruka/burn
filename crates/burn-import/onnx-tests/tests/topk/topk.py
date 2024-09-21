#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/topk/topk.onnx

import torch
import torch.nn as nn
import torch.onnx


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc(x)
        values, indices = torch.topk(x, k=3, dim=1)
        return values, indices


def main() -> None:
    model = SimpleModel()
    model.eval()

    # input data (batchsize: 1, feature dims: 10)
    device = torch.device("cpu")
    dummy_input = torch.randn(1, 10, device=device)

    # Model infers for sure.
    output_values, output_indices = model(dummy_input)
    print("Output values:", output_values)
    print("Output indices:", output_indices)
    
    # Turn into ONNX
    torch.onnx.export(model,
                  dummy_input,
                  "topk.onnx",
                  export_params=True,
                  opset_version=11,
                  input_names=['input'],
                  output_names=['values', 'indices'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'values': {0: 'batch_size'}, 
                                'indices': {0: 'batch_size'}})


if __name__ == "__main__":
    main()