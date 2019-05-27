import torch
import numpy as np
import torch.nn as nn
import patterns

class PatternConv2d(nn.Module):

    def __init__(self, conv_layer):
        super(PatternConv2d, self).__init__()

        self.forward_layer = conv_layer  # kernels size of forward layer: 
                                         # self.forward_layer.kernel_size
        padding_f = conv_layer.padding[0]
        ks = conv_layer.kernel_size[0]
        padding_b = -padding_f + ks - 1
        self.backward_layer =  nn.Conv2d(
            conv_layer.out_channels,
            conv_layer.in_channels,
            conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=padding_b,
            bias=False,
        )

        self.statistics = None
        self.patterns = None

    def forward(self, input):
        ''' perform forward computations of forward_layer, return not only new
            output, but also output without bias, if the forward layer has a 
            bias parameter.
        '''
        
        def expand_bias(bias, size):
            new_tensor = torch.zeros((size))
            for i in range(bias.shape[0]):
                new_tensor[:, i, :, :] = bias[i]

            return new_tensor

        
        output = self.forward_layer(input)
        # TODO: what if the layer does not have a bias?
        bias = expand_bias(self.forward_layer.bias.data)
        output_wo_bias = output - bias

        return output, output_wo_bias


    def backward(self, input):
        ''' compute a backward step (for signal computation).
        '''
        output = self.backward_layer(input)
        # rescale output to be between -1 and 1
        absmax = torch.abs(output.data).max()
        if absmax > 0.000001:
            output.data /= absmax
        output.data[output.data > 1] = 1
        output.data[output.data < -1] = -1

        return output


    def compute_statistics(self, input, output, output_wo_bias=None):
        ''' compute statistics for this layer given the input, output and 
            output without bias. Initialize statistics if none there yet,
            otherwise update statistics with new values.

            If the forward layer does not use a bias term, then the output
            without bias, i.e. the layer's output, is in output and there is 
            no tensor in output_wo_bias.
        '''
        if output_wo_bias is None:
            inp_dense, out_dense = patterns._conv_maps_to_dense(input, output,
                                        self.forward_layer.kernel_size)
            if self.statistics is None:
                self.statistics = patterns.compute_statistics(inp_dense, 
                                                              out_dense, 
                                                              out_dense)
            else:
                self.statistics = patterns.update_statistics(inp_dense,
                                                             out_dense,
                                                             out_dense,
                                                             self.statistics)

        else:
            inp_dense, out_wo_bias_dense = patterns._conv_maps_to_dense(input, 
                                            output_wo_bias,
                                            self.forward_layer.kernel_size)
            _, out_dense = patterns._conv_maps_to_dense(input, output,
                                        self.forward_layer.kernel_size)
            if self.statistics is None:
                self.statistics = patterns.compute_statistics(inp_dense, 
                                                            out_wo_bias_dense, 
                                                            out_dense)
            else:
                self.statistics = patterns.update_statistics(inp_dense,
                                                             out_wo_bias_dense,
                                                             out_dense,
                                                             self.statistics)
        

    def compute_patterns(self):
        ''' Compute patterns from the computed statistics. 
        '''
        kernel = next(self.forward_layer.parameters())
        self.patterns = patterns.compute_patterns_conv(self.statistics, 
                                                       kernel)

    def set_patterns(self, pattern_type='relu'):
        ''' Sets the computed patterns as the kernel of the backward layer.
            pattern_type can be 'relu' or 'linear'
        '''
        if pattern_type == 'relu':
            next(self.backward_layer.parameters()) = self.patterns['A_plus']
        elif pattern_type == 'linear':
            next(self.backward_layer.parameters()) = self.patterns['A_linear']


class PatternLinear(nn.Module):

    def __init__(self, linear_layer):
        super(PatternLinear, self).__init__()

        self.forward_layer = linear_layer 

        self.backward_layer = nn.Linear(linear_layer.out_features, 
                                        linear_layer.in_features, 
                                        bias=False)

        self.statistics = None
        self.patterns = None

    def forward(self, input):
        ''' perform forward computations of forward_layer, return not only new
            output, but also output without bias, if the forward layer has a 
            bias parameter.
        '''

        def expand_bias(bias, size):
            new_tensor = torch.zeros((size))
            for i in range(bias.shape[0]):
                new_tensor[:, i] = bias[i]

            return new_tensor

        output = self.forward_layer(input)
        # TODO: what if the layer does not have a bias?
        bias = expand_bias(self.forward_layer.bias.data)
        output_wo_bias = output - bias

        return output, output_wo_bias


    def backward(self, input):
        ''' compute a backward step (for signal computation).
        '''
        output = self.backward_layer(input)
        # rescale output to be between -1 and 1
        absmax = torch.abs(output.data).max()
        if absmax > 0.000001:
            output.data /= absmax
        output.data[output.data > 1] = 1
        output.data[output.data < -1] = -1

        return output


    def compute_statistics(self, input, output, output_wo_bias=None):
        ''' compute statistics for this layer given the input, output and 
            output without bias. Initialize statistics if none there yet,
            otherwise update statistics with new values.

            If the forward layer does not use a bias term, then the output
            without bias, i.e. the layer's output, is in output and there is 
            no tensor in output_wo_bias.
        '''
        if output_wo_bias is None:
            if self.statistics is None:
                self.statistics = patterns.compute_statistics(input, 
                                                              output, 
                                                              output)
            else:
                self.statistics = patterns.update_statistics(input,
                                                             output,
                                                             output,
                                                             self.statistics)

        else:
            if self.statistics is None:
                self.statistics = patterns.compute_statistics(input, 
                                                            output_wo_bias, 
                                                            output)
            else:
                self.statistics = patterns.update_statistics(input,
                                                             output_wo_bias,
                                                             output,
                                                             self.statistics)
        

    def compute_patterns(self):
        ''' Compute patterns from the computed statistics. 
        '''
        w = next(self.forward_layer.parameters())
        self.patterns = patterns.compute_patterns_linear(self.statistics, w)


    def set_patterns(self, pattern_type='relu'):
        ''' Sets the computed patterns as the kernel of the backward layer.
            pattern_type can be 'relu' or 'linear'
        '''
        if pattern_type == 'relu':
            next(self.backward_layer.parameters()) = self.patterns['A_plus']
        elif pattern_type == 'linear':
            next(self.backward_layer.parameters()) = self.patterns['A_linear']


class PatternReLU(nn.Module):

    def __init__(self):
        super(PatternReLU, self).__init__()

        self.forward_layer = nn.ReLU()

    def forward(self, input):
        indices = input <= 0
        output = self.forward_layer(input)

        return output, indices

    def backward(self, input, indices):
        # copy the input
        input = torch.new_tensor(input)
        input[indices] = 0

        return input


class PatternMaxPool2d(nn.Module):

    def __init__(self, pool_layer):
        super(PatternMaxPool2d, self).__init__()

        # create a new pooling layer to use a different instance with 
        # return_indices=True without changing the original layer's
        # settings
        self.forward_layer = nn.MaxPool2d(pool_layer.kernel_size,
                                          pool_layer.stride,
                                          pool_layer.padding,
                                          pool_layer.dilation,
                                          return_indices=True)
        self.backward_layer = nn.MaxUnpool2d(pool_layer.kernel_size, 
                                             pool_layer.stride, 
                                             pool_layer.padding)

    def forward(self, input):
        return self.forward_layer(input)

    def backward(self, input, switches):
        return self.backward_layer(input, switches)
    