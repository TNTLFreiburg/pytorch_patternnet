import torch
import torch.nn as nn
from torch.autograd import Variable
import patterns
import time
import layers


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
                if cur_layer == "Conv2d" and next_layer == "Linear":
                    self.reshape_ind = next_layer_ind
                cur_layer_ind = next_layer_ind
                cur_layer = next_layer
                next_layer_ind = cur_layer_ind +1


        # initialize the backward layers
        self.backward_layers = []
        for layer in self.lst_layers[::-1]:
            if layer.__class__.__name__ == "Conv2d":
                self.backward_layers.append(layers.PatternConv2d(layer))
            if layer.__class__.__name__ == "Linear":
                self.backward_layers.append(layers.PatternLinear(layer))
            if layer.__class__.__name__ == "MaxPool2d":
                self.backward_layers.append(layers.PatternMaxPool2d(layer))
            if layer.__class__.__name__ == "ReLU":
                self.backward_layers.append(layers.PatternReLU())

    def compute_signal(self, img, only_biggest_value=True):
        """ Additional method to compute the signal for a given image. """
        return self.forward(img, only_biggest_value)

    def forward(self, img, only_biggest_value=True):

        output, _, indices, switches = self.forward_with_extras(img)

        if not only_biggest_value:
            y = output
        # use only highest valued output
        else:
            y = torch.zeros(output.size(), requires_grad=False)
            max_v, max_i = torch.max(output.data, dim=1)
            y[range(y.shape[0]), max_i] = max_v

        ind_cnt = 0
        switch_cnt = 0

        # go through all layers and apply their backward pass functionality
        for ind, layer in enumerate(self.backward_layers):
            if layer.__class__.__name__ == "PatternReLU":
                mask = indices[ind_cnt]
                y = layer.backward(y, mask)
                ind_cnt += 1
            elif layer.__class__.__name__ == "PatternMaxPool2d":
                y = layer.backward(y, switches[switch_cnt])
                switch_cnt += 1
            else:
                # if other layer than linear or conv, could theoretically
                # be applied here without noticing
                y = layer.backward(y)

                # check if reshape is necessary
                if len(self.lst_layers) - ind == self.reshape_ind + 1:
                    s = self._reshape_size_in
                    y.data = y.data.view(-1, s[1], s[2], s[3])
 
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

        layers = []
        layers_wo_bias = []
        cnt = 0
        indices = []
        switches = []

        for ind, layer in enumerate(self.backward_layers[::-1]):
            # print(layer.forward_layer)
            if layer.__class__.__name__ == "PatternConv2d":
                # save input to layer
                layers.append({})
                layers[cnt]["inputs"] = output.data
                # apply forward layer
                output, output_wo_bias = layer(output)
                # save output of layer
                layers[cnt]["outputs"] = output.data
                # save output without bias
                layers_wo_bias.append(output_wo_bias)
                cnt += 1
            elif layer.__class__.__name__ == "PatternLinear":
                # save input to layer
                layers.append({})
                layers[cnt]["inputs"] = output.data
                # apply layer
                output, output_wo_bias = layer(output)
                # save output of layer
                layers[cnt]["outputs"] = output.data
                # save output without bias
                layers_wo_bias.append(output_wo_bias)
                cnt += 1
            elif layer.__class__.__name__ == "PatternMaxPool2d":
                # set return indices to true to get the switches
                # apply layer
                output, switch = layer(output)
                # save switches
                switches.append(switch)
            elif layer.__class__.__name__ == "PatternReLU":
                # save indices smaller zero
                output, inds = layer(output)
                indices.append(inds)

            # add view between convolutional and linear sequential
            if ind == self.reshape_ind-1:  # layer before the first linear layer
                    self._reshape_size_in = output.shape
                    output = output.view(-1,self.lst_layers[ind+1].in_features)


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

        # cnt for layers with params
        cnt = 0
        for layer in self.backward_layers[::-1]:
            if layer.__class__.__name__ in ["PatternConv2d", "PatternLinear"]:
                layer.compute_statistics(layer_outputs[cnt]["inputs"],
                                         layer_outputs[cnt]["outputs"],
                                         layer_outputs_wo_bias[cnt])
                cnt += 1


    def compute_patterns(self):

            for layer in self.backward_layers[::-1]:
                if layer.__class__.__name__ in ["PatternConv2d", "PatternLinear"]:
                    layer.compute_patterns()


    def set_patterns(self, pattern_type="relu"):
        """ pattern_type can either be A_plus or A_linear
        """
        for layer in self.backward_layers[::-1]:
            if layer.__class__.__name__ in ["PatternConv2d", "PatternLinear"]:
                layer.set_patterns(pattern_type=pattern_type)