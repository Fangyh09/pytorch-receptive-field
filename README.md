# pytorch-receptive-field
Compute CNN receptive field size in pytorch

## Usage
`git clone https://github.com/Fangyh09/pytorch-receptive-field.git`

```python
from torch_receptive_field import receptive_field
receptive_field(model, input_size=(channels, H, W))
```

## Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_receptive_field import receptive_field

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        y = self.maxpool(y)
        return y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Net().to(device)

receptive_field(model, (3, 256, 256))
```
```
------------------------------------------------------------------------------
        Layer (type)    map size      start       jump receptive_field
==============================================================================
        0             [256, 256]        0.5        1.0             1.0
        1             [128, 128]        0.5        2.0             7.0
        2             [128, 128]        0.5        2.0             7.0
        3             [128, 128]        0.5        2.0             7.0
        4               [64, 64]        0.5        4.0            11.0
==============================================================================
```

## More
`start` is the center of first item in the map grid .

`jump` is the distance of the adjacent item in the map grid.

`receptive_field` is the field size of the item in the map grid.

## Related
Thanks @pytorch-summary

https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
