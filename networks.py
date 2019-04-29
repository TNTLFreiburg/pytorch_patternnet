import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import patterns
import time


class MnistNet(nn.Module):

    def __init__(self):
        super(MnistNet, self).__init__()

        # 32 for cifar 10, 28 for mnist
        input_height = 28
        num_channels = 1
        self.map_width = int(((input_height - 4) / 2 - 4) / 2)
        # define layer weights as attributes of the net
        # all attributes are later parameters of the net
        self.conv1 = nn.Conv2d(num_channels, 16, 5)
        # definition for pool, to save the indices of the
        # maximal values, add: return_indices=True
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 64, 5)
        self.fc1 = nn.Linear(64 * self.map_width * self.map_width, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

        self.layers = [self.conv1, self.relu, self.pool, 
                       self.conv2, self.relu, self.pool,
                       self.fc1, self.relu, self.fc2, self.relu, self.fc3]

    def forward(self, x):

        # define computations during forward pass
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * self.map_width * self.map_width)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PatternNet(torch.nn.Module):

    def __init__(self, layers_lst):
        super(PatternNet, self).__init__()

        # initialize variables for later use
        self.layer_stats = None
        self.patterns_lst = None
        self.lst_layers = layers_lst
        # save change from conv to linear layers to know when to reshape
        # get first conv or linear layer
        cur_layer_ind = 0
        while not (self.lst_layers[cur_layer_ind].__class__.__name__ in \
                ["Conv2d","Linear"]) and cur_layer_ind < len(self.lst_layers):

            cur_layer_ind+=1
        # if only the last layer is conv or linear, do not need to find next layer
        if cur_layer_ind < len(self.lst_layers) -1:
            cur_layer = self.lst_layers[cur_layer_ind].__class__.__name__
            next_layer_ind = cur_layer_ind + 1
            while cur_layer_ind < len(self.lst_layers) -1:
                while not (self.lst_layers[next_layer_ind].__class__.__name__ in \
                        ["Conv2d","Linear"]) and next_layer_ind < len(self.lst_layers):
                    next_layer_ind+=1
                next_layer = self.lst_layers[next_layer_ind].__class__.__name__
                # now check if there's a change between conv and linear layers
                # print('Current layer:', cur_layer, 'Next layer:', next_layer)
                # print('Indices:',cur_layer_ind, next_layer_ind)
                if cur_layer == "Conv2d" and next_layer == "Linear":
                    self.reshape_ind = next_layer_ind
                cur_layer_ind = next_layer_ind
                cur_layer = next_layer
                next_layer_ind = cur_layer_ind +1
        print('Linear layer for reshape:', self.reshape_ind)

        self.weights_lst = []
        for layer in layers_lst:
            if layer.__class__.__name__ in ['Conv2d', 'Linear']:
                self.weights_lst.append(layer.weight.data)
                self.weights_lst.append(layer.bias.data)

        # initialize the backward layers
        self.backward_layers = []
        for layer in self.lst_layers[::-1]:
            if layer.__class__.__name__ == "Conv2d":
                self.backward_layers.append(deconv_from_conv(layer))
            if layer.__class__.__name__ == "Linear":
                self.backward_layers.append(backlin_from_linear(layer))
            if layer.__class__.__name__ == "MaxPool2d":
                self.backward_layers.append(unpool_from_pool(layer))
            if layer.__class__.__name__ == "ReLU":
                self.backward_layers.append(BackwardReLU())

    def compute_signal(self, img, only_biggest_value=True):
        """ Additional method to compute the signal for a given image. """
        return self.forward(img, only_biggest_value)

    def forward(self, img, only_biggest_value=True):

        output, _, indices, switches = self.forward_with_extras(img)

        if not only_biggest_value:
            y = output
        # use only highest valued output
        else:
            # atm only works for single images, make it work with several 
            # images as well
            y = torch.zeros(output.size(), requires_grad=False)
            max_v, max_i = torch.max(output.data, dim=1)
            # max_i = int(max_i.numpy())
            # max_v = float(max_v.numpy())
            y[range(y.shape[0]), max_i] = max_v

        ind_cnt = 0
        switch_cnt = 0
        pool_cnt = 0

        # go through all layers and apply their backward pass functionality
        for ind, layer in enumerate(self.backward_layers):
            if layer.__class__.__name__ == "BackwardReLU":
                mask = indices[ind_cnt]
                y = layer(y, mask)
                ind_cnt += 1
            elif layer.__class__.__name__ == "MaxUnpool2d":
                y = layer(y, switches[switch_cnt])
                switch_cnt += 1
                pool_cnt += 1
            else:
                # if other layer than linear or conv, could theoretically
                # be applied here without noticing
                y = layer(y)
                # scale output to be between -1 and 1
                absmax = torch.abs(y.data).max()
                if absmax > 0.000001:
                    y.data = y.data / absmax
                y.data[y.data > 1] = 1
                y.data[y.data < -1] = -1

                # check if reshape is necessary
                if len(self.lst_layers) - ind == self.reshape_ind + 1:

                    # s = self._reshape_size_in
                    # res = torch.zeros(y.data.size(0), s[1], s[2], s[3])

                    # # backward view as computed by keras
                    # for i in range(y.data.size(0)):
                    #     for j in range(y.data.size(1)):
                    #         k = int((j / s[1]) / s[2])
                    #         l = int(j / s[1]) % s[2]
                    #         res[i, j % s[1], k, l] = y.data[i, j]
                    # shape_before_reshape = y.shape
                    # y.data = res

                    s = self._reshape_size_in
                    shape_before_reshape = y.shape
                    y.data = y.data.view(-1, s[1], s[2], s[3])
                    print('Reshaping during backward pass:')
                    print('Shape before reshape:', shape_before_reshape, 
                          'Shape after reshape:', y.shape)

        return y

    def forward_with_extras(self, imgs):
        """
        Performs one forward pass through the network given at initialization
        (only convolutional, linear, pooling and ReLU layer). Additionally to
        the final output the input and output to each convolutional and linear
        layer, the switches of pooling layers and the indices of the values
        that are set to zero of each ReLU layer, are returned.
        """

        output = Variable(imgs, requires_grad=False)

        layers = {}
        layers_wo_bias = {}
        conv_cnt = 1
        lin_cnt = 1
        indices = []
        switches = []

        def expand_bias(bias, size):
            new_tensor = torch.zeros((size))
            for i in range(bias.shape[0]):
                if len(size) == 4:
                    new_tensor[:, i, :, :] = bias[i]
                else:
                    new_tensor[:, i] = bias[i]

            return new_tensor

        for ind, layer in enumerate(self.lst_layers):
            if layer.__class__.__name__ == "Conv2d":
                # save input to layer
                layers["Conv%s" % str(conv_cnt).zfill(3)] = {}
                layers["Conv%s" % str(conv_cnt).zfill(3)][
                    "inputs"
                ] = output.data
                layers["Conv%s" % str(conv_cnt).zfill(3)][
                    "kernel_size"
                ] = layer.kernel_size
                # apply layer
                output = layer(output)
                # save output of layer
                layers["Conv%s" % str(conv_cnt).zfill(3)][
                    "outputs"
                ] = output.data
                # save output without bias
                bias = layer.bias.data
                bias = expand_bias(bias, output.data.shape)
                output_wo_bias = output.data - bias
                layers_wo_bias[
                    "Conv%s" % str(conv_cnt).zfill(3)
                ] = output_wo_bias
                conv_cnt += 1
            elif layer.__class__.__name__ == "Linear":
                # save input to layer
                layers["Linear%s" % str(lin_cnt).zfill(3)] = {}
                layers["Linear%s" % str(lin_cnt).zfill(3)][
                    "inputs"
                ] = output.data
                # apply layer
                output = layer(output)
                # save output of layer
                layers["Linear%s" % str(lin_cnt).zfill(3)][
                    "outputs"
                ] = output.data
                # save output without bias
                bias = layer.bias.data
                bias = expand_bias(bias, output.data.shape)
                output_wo_bias = output.data - bias
                layers_wo_bias[
                    "Linear%s" % str(lin_cnt).zfill(3)
                ] = output_wo_bias
                lin_cnt += 1
            elif layer.__class__.__name__ == "MaxPool2d":
                # set return indices to true to get the switches
                layer.return_indices = True
                # apply layer
                output, switch = layer(output)
                # save switches
                switches.append(switch)
            elif layer.__class__.__name__ == "ReLU":
                # save indices smaller zero
                indices.append(output <= 0)
                # apply layer
                output = layer(output)

            # add view between convolutional and linear sequential
            if ind == self.reshape_ind-1:  # layer before the first linear layer
                    self._reshape_size_in = output.shape
                    output = output.view(-1,self.lst_layers[ind+1].in_features)
                    print('Reshaping during forward pass:')
                    print('Shape before reshape:', self._reshape_size_in, 
                          'Shape after reshape:', output.shape)

        return (
            output,
            (layers, layers_wo_bias),
            indices[::-1],
            switches[::-1],
        )

    def compute_statistics(self, imgs, use_bias=False):
        """ Initializes statistics if no statistics were computed
            before. Otherwise updates the already computed statistics.
        """

        # get the layer outputs
        _, outputs, _, _ = self.forward_with_extras(imgs)
        layer_outputs = outputs[0]
        layer_outputs_wo_bias = outputs[1]

        # initialize statistics
        if self.layer_stats is None:
            self.layer_stats = {}
            for key in sorted(layer_outputs):
                inp = layer_outputs[key]["inputs"]
                if not use_bias:
                    out = layer_outputs_wo_bias[key]
                else:
                    out = layer_outputs[key]["outputs"]

                if "Conv" in key:
                    ks = layer_outputs[key]["kernel_size"]
                    inp_dense, out_dense = patterns._conv_maps_to_dense(
                        inp, out, ks
                    )
                    ### added to get right counts
                    _, out_dense_wb = patterns._conv_maps_to_dense(
                        inp, layer_outputs[key]["outputs"], ks
                    )
                    self.layer_stats[key] = patterns.compute_statistics(
                        inp_dense, out_dense, out_dense_wb
                    )
                else:
                    ### added to get right counts
                    self.layer_stats[key] = patterns.compute_statistics(
                        inp, out, layer_outputs[key]["outputs"]
                    )

        # update statistics
        else:

            for key in sorted(layer_outputs):
                inp = layer_outputs[key]["inputs"]
                if not use_bias:
                    out = layer_outputs_wo_bias[key]
                else:
                    out = layer_outputs[key]["outputs"]
                if "Conv" in key:
                    ks = layer_outputs[key]["kernel_size"]
                    inp_dense, out_dense = patterns._conv_maps_to_dense(
                        inp, out, ks
                    )
                    ### added to get right counts
                    _, out_dense_wb = patterns._conv_maps_to_dense(
                        inp, layer_outputs[key]["outputs"], ks
                    )
                    self.layer_stats[key] = patterns.update_statistics(
                        inp_dense, out_dense, out_dense_wb, 
                        self.layer_stats[key])
                else:
                    ### added to get right counts
                    self.layer_stats[key] = patterns.update_statistics(
                        inp, out, layer_outputs[key]["outputs"],
                        self.layer_stats[key])

    def compute_patterns(self):

        if self.layer_stats is None:
            print(
                "No statistics computed yet, therefore no pattern computation",
                " possible!",
            )

        else:
            self.patterns_lst = []
            for ind, key in enumerate(sorted(self.layer_stats)):
                if "Conv" in key:
                    # since conv layers come before linear layers the
                    # order of layers is as in weights_lst
                    pattern = patterns.compute_patterns_conv(
                        self.layer_stats[key], self.weights_lst[ind * 2]
                    )
                    self.patterns_lst.append(pattern)
                else:
                    pattern = patterns.compute_patterns_linear(
                        self.layer_stats[key], self.weights_lst[ind * 2]
                    )
                    self.patterns_lst.append(pattern)

    def set_patterns(self, pattern_type="A_plus"):
        """ pattern_type can either be A_plus or A_linear
        """
        if self.patterns_lst is None:
            print("No patterns computed yet!")

        else:
            pattern_cnt = 0

            # go through the backward layer list in reverse order, so that
            # the order is the same as in the patterns list
            for layer in self.backward_layers[::-1]:
                layer_class = layer.__class__.__name__
                if layer_class == "Conv2d":
                    layer.parameters().__next__().data = self.patterns_lst[
                        pattern_cnt
                    ][pattern_type]
                    pattern_cnt += 1
                elif layer_class == "Linear":
                    layer.parameters().__next__().data = self.patterns_lst[
                        pattern_cnt
                    ][pattern_type].permute(1, 0)
                    pattern_cnt += 1


def unpool_from_pool(pooling_layer):
    return nn.MaxUnpool2d(
        pooling_layer.kernel_size, pooling_layer.stride, pooling_layer.padding
    )


def deconv_from_conv(conv_layer):
    padding_f = conv_layer.padding[0]
    ks = conv_layer.kernel_size[0]
    padding_b = -padding_f + ks - 1
    return nn.Conv2d(
        conv_layer.out_channels,
        conv_layer.in_channels,
        conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=padding_b,
        bias=False,
    )


def backlin_from_linear(lin_layer):
    return nn.Linear(lin_layer.out_features, lin_layer.in_features, bias=False)


class BackwardReLU(torch.nn.Module):

    def __init__(self):
        super(BackwardReLU, self).__init__()

    def forward(self, inp, indices):
        inp[indices] = 0

        return inp