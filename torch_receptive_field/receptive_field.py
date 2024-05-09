import torch
import torch.nn as nn
try:
    import cv2
    cv2_available = True
except ImportError:
    cv2_available = False
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from matplotlib.animation import FuncAnimation
from collections import OrderedDict


def check_same(stride):
    if isinstance(stride, (list, tuple)):
            assert (len(stride) == 2 and stride[0] == stride[1]) or (len(stride) == 3 and stride[0] == stride[1] and stride[1] == stride[2])
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
        pointwise_operations = ['ReLU', 'LeakyReLU',
                                'ELU', 'Hardshrink', 'Hardsigmoid', 'Hardtanh', 'LogSigmoid', 'PReLU',
                                'ReLU6', 'RReLU', 'SELU', 'CELU', 'GELU', 'Sigmoid', 'SiLU', 'Mish',
                                'Softplus', 'Softshrink', 'Softsign', 'Tanh', 'Tanhshrink', 'Threshold', 'GLU']

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
                
                if class_name == "Conv2d" or class_name == "MaxPool2d" or class_name == "AvgPool2d" or class_name == "Conv3d" or class_name == "MaxPool3d":
                    kernel_size = module.kernel_size
                    stride = module.stride
                    padding = module.padding

                    if class_name == "AvgPool2d":
                        # Avg Pooling does not have dilation, set it to 1 (no dilation)
                        dilation = 1
                    else:
                        dilation = module.dilation

                    kernel_size, stride, padding, dilation = map(check_same, [kernel_size, stride, padding, dilation])
                    receptive_field[m_key]["j"] = p_j * stride
                    receptive_field[m_key]["r"] = p_r + ((kernel_size - 1) * dilation) * p_j
                    receptive_field[m_key]["start"] = p_start + ((kernel_size - 1) / 2 - padding) * p_j
                elif class_name in pointwise_operations or class_name == "BatchNorm2d" or class_name == "Bottleneck" or class_name == "BatchNorm3d":
                    receptive_field[m_key]["j"] = p_j
                    receptive_field[m_key]["r"] = p_r
                    receptive_field[m_key]["start"] = p_start
                elif class_name == "ConvTranspose2d" or class_name == "ConvTranspose3d":
                    receptive_field["0"]["conv_stage"] = False
                    receptive_field[m_key]["j"] = 0
                    receptive_field[m_key]["r"] = 0
                    receptive_field[m_key]["start"] = 0
                else:
                    raise ValueError("module {} not supported yet".format(class_name))
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
            and not isinstance(module, nn.Linear)
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
    for layer in receptive_field:
        # input_shape, output_shape, trainable, nb_params
        assert "start" in receptive_field[layer], layer
        assert len(receptive_field[layer]["output_shape"]) == 4 or len(receptive_field[layer]["output_shape"]) == 5
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
        assert len(unit_position) == 2 or len(unit_position) == 3
        feat_map_lim = rf_stats['output_shape'][2:]
        if np.any([unit_position[idx] < 0 or
                   unit_position[idx] >= feat_map_lim[idx]
                   for idx in range(len(unit_position))]):
            if len(unit_position) == 2:
                raise Exception("Unit position outside spatial extent of the feature tensor ((H, W) = (%d, %d)) " % tuple(feat_map_lim))
            else:
                raise Exception("Unit position outside spatial extent of the feature tensor ((D, H, W) = (%d, %d, %d)) " % tuple(feat_map_lim))
        # X, Y = tuple(unit_position)
        rf_range = [(rf_stats['start'] + idx * rf_stats['j'] - rf_stats['r'] / 2,
            rf_stats['start'] + idx * rf_stats['j'] + rf_stats['r'] / 2) for idx in unit_position]
        if len(unit_position) == 2:
            if len(input_shape) == 2:
                limit = input_shape
            else:  # input shape is (channel, H, W)
                limit = input_shape[1:3]
            rf_range = [(max(0, rf_range[axis][0]), min(limit[axis], rf_range[axis][1])) for axis in range(2)]
        else:
            if len(input_shape) == 3:
                limit = input_shape
            else:  # input shape is (channel, D, H, W)
                limit = input_shape[1:4]
            rf_range = [(max(0, rf_range[axis][0]), min(limit[axis], rf_range[axis][1])) for axis in range(3)]

        print("Receptive field size for layer %s, unit_position %s,  is \n %s" % (layer, unit_position, rf_range))
        return rf_range
    else:
        raise KeyError("Layer name incorrect, or not included in the model.")


def read_image(image):
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, str):
        if not cv2_available:
            raise ValueError(f"Image loading requires the cv2 module.")
        try:
            return cv2.imread(image)
        except Exception as e:
            raise ValueError(f"Error reading image: {e}")
    else:
        raise ValueError("Unsupported image type. Supported types: numpy array, file path")


def receptive_field_visualization_2d(receptive_field_dict, image, save_name="receptive_field_visualization_2d"):
    if not cv2_available:
        raise ValueError(f"Visualization requires the cv2 module.")
    
    ordered_key_list = list(receptive_field_dict.keys())[:-1]
    image_size = receptive_field_dict["input_size"][1]
    image = read_image(image)
    image = cv2.resize(image, (image_size, image_size))
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = (gray_image * 0.5).astype(np.uint8)
    fig, ax = plt.subplots()

    def update_frame(frame_number):
        layer = ordered_key_list[frame_number]
        rf_size = int(receptive_field_dict[layer]["r"])
        image_with_rf = image.copy()
        top_left = (center_x - rf_size // 2, center_y - rf_size // 2)
        bottom_right = (center_x + rf_size // 2, center_y + rf_size // 2)

        # Set the pixels outside of the receptive field to grayscale
        if top_left[1] > 0:
            image_with_rf[:int(top_left[1]), :] = cv2.cvtColor(gray_image[:int(top_left[1]), :], cv2.COLOR_GRAY2BGR)
        if bottom_right[1] < image.shape[0]:
            image_with_rf[int(bottom_right[1]):, :] = cv2.cvtColor(gray_image[int(bottom_right[1]):, :], cv2.COLOR_GRAY2BGR)
        if top_left[0] > 0:
            image_with_rf[:, :int(top_left[0])] = cv2.cvtColor(gray_image[:, :int(top_left[0])], cv2.COLOR_GRAY2BGR)
        if bottom_right[0] < image.shape[1]:
            image_with_rf[:, int(bottom_right[0]):] = cv2.cvtColor(gray_image[:, int(bottom_right[0]):], cv2.COLOR_GRAY2BGR)

        ax.clear()
        ax.imshow(cv2.cvtColor(image_with_rf, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Layer {layer: <3} | Receptive Field {rf_size: <3}")

    ani = FuncAnimation(fig, update_frame, frames=len(ordered_key_list))
    ani.save(f'{save_name}.gif', writer='imagemagick', fps=2)
