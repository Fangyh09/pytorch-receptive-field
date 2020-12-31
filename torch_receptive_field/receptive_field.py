import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

def check_same(stride):
    if isinstance(stride, (list, tuple)):
        assert len(stride) == 2 and stride[0] == stride[1]
        stride = stride[0]
    return stride

def receptive_field(model, input_size, batch_size=-1, device="cuda"):
    '''
    :parameter
    'input_size': tuple of (Channel, Height, Width)

    :return  OrderedDict of `Layername`->OrderedDict of receptive field stats {'j':,'r':,'start':,'conv_stage':,'output_shape':,}
    'j' for "jump" denotes how many pixels do the receptive fields of spatially neighboring units in the feature tensor
        do not overlap in one direction.
        i.e. shift one unit in this feature map == how many pixels shift in the input image in one direction.
    'r' for "receptive_field" is the spatial range of the receptive field in one direction.
    'start' denotes the center of the receptive field for the first unit (start) in on direction of the feature tensor.
        Convention is to use half a pixel as the center for a range. center for `slice(0,5)` is 2.5.
    '''
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(receptive_field)
            m_key = "%i" % module_idx
            p_key = "%i" % (module_idx - 1)
            receptive_field[m_key] = OrderedDict()

            if not receptive_field["0"]["conv_stage"]:
                print("Enter in deconv_stage")
                receptive_field[m_key]["j"] = 0
                receptive_field[m_key]["r"] = 0
                receptive_field[m_key]["start"] = 0
            else:
                p_j = receptive_field[p_key]["j"]
                p_r = receptive_field[p_key]["r"]
                p_start = receptive_field[p_key]["start"]
                
                if class_name == "Conv2d" or class_name == "MaxPool2d":
                    kernel_size = module.kernel_size
                    stride = module.stride
                    padding = module.padding
                    dilation = module.dilation
       
                    kernel_size, stride, padding, dilation = map(check_same, [kernel_size, stride, padding, dilation])
                    receptive_field[m_key]["j"] = p_j * stride
                    receptive_field[m_key]["r"] = p_r + ((kernel_size - 1) * dilation) * p_j
                    receptive_field[m_key]["start"] = p_start + ((kernel_size - 1) / 2 - padding) * p_j
                elif class_name == "BatchNorm2d" or class_name == "ReLU" or class_name == "Bottleneck":
                    receptive_field[m_key]["j"] = p_j
                    receptive_field[m_key]["r"] = p_r
                    receptive_field[m_key]["start"] = p_start
                elif class_name == "ConvTranspose2d":
                    receptive_field["0"]["conv_stage"] = False
                    receptive_field[m_key]["j"] = 0
                    receptive_field[m_key]["r"] = 0
                    receptive_field[m_key]["start"] = 0
                else:
                    raise ValueError("module not ok")
                    pass
            receptive_field[m_key]["input_shape"] = list(input[0].size()) # only one
            receptive_field[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                # list/tuple
                receptive_field[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                # tensor
                receptive_field[m_key]["output_shape"] = list(output.size())
                receptive_field[m_key]["output_shape"][0] = batch_size

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(2, *in_size)).type(dtype) for in_size in input_size]
    else:
        x = Variable(torch.rand(2, *input_size)).type(dtype)

    # create properties
    receptive_field = OrderedDict()
    receptive_field["0"] = OrderedDict()
    receptive_field["0"]["j"] = 1.0
    receptive_field["0"]["r"] = 1.0
    receptive_field["0"]["start"] = 0.5
    receptive_field["0"]["conv_stage"] = True
    receptive_field["0"]["output_shape"] = list(x.size())
    receptive_field["0"]["output_shape"][0] = batch_size
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("------------------------------------------------------------------------------")
    line_new = "{:>20}  {:>10} {:>10} {:>10} {:>15} ".format("Layer (type)", "map size", "start", "jump", "receptive_field")
    print(line_new)
    print("==============================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in receptive_field:
        # input_shape, output_shape, trainable, nb_params
        assert "start" in receptive_field[layer], layer
        assert len(receptive_field[layer]["output_shape"]) == 4
        line_new = "{:7} {:12}  {:>10} {:>10} {:>10} {:>15} ".format(
            "",
            layer,
            str(receptive_field[layer]["output_shape"][2:]),
            str(receptive_field[layer]["start"]),
            str(receptive_field[layer]["j"]),
            format(str(receptive_field[layer]["r"]))
        )
        print(line_new)

    print("==============================================================================")
    # add input_shape
    receptive_field["input_size"] = input_size
    return receptive_field


def receptive_field_for_unit(receptive_field_dict, layer, unit_position):
    """Utility function to calculate the receptive field for a specific unit in a layer
        using the dictionary calculated above
    :parameter
        'layer': layer name, should be a key in the result dictionary
        'unit_position': spatial coordinate of the unit (H, W)

    ```
    alexnet = models.alexnet()
    model = alexnet.features.to('cuda')
    receptive_field_dict = receptive_field(model, (3, 224, 224))
    receptive_field_for_unit(receptive_field_dict, "8", (6,6))
    ```
    Out: [(62.0, 161.0), (62.0, 161.0)]
    """
    input_shape = receptive_field_dict["input_size"]
    if layer in receptive_field_dict:
        rf_stats = receptive_field_dict[layer]
        assert len(unit_position) == 2
        feat_map_lim = rf_stats['output_shape'][2:]
        if np.any([unit_position[idx] < 0 or
                   unit_position[idx] >= feat_map_lim[idx]
                   for idx in range(2)]):
            raise Exception("Unit position outside spatial extent of the feature tensor ((H, W) = (%d, %d)) " % tuple(feat_map_lim))
        # X, Y = tuple(unit_position)
        rf_range = [(rf_stats['start'] + idx * rf_stats['j'] - rf_stats['r'] / 2,
            rf_stats['start'] + idx * rf_stats['j'] + rf_stats['r'] / 2) for idx in unit_position]
        if len(input_shape) == 2:
            limit = input_shape
        else:  # input shape is (channel, H, W)
            limit = input_shape[1:3]
        rf_range = [(max(0, rf_range[axis][0]), min(limit[axis], rf_range[axis][1])) for axis in range(2)]
        print("Receptive field size for layer %s, unit_position %s,  is \n %s" % (layer, unit_position, rf_range))
        return rf_range
    else:
        raise KeyError("Layer name incorrect, or not included in the model.")
