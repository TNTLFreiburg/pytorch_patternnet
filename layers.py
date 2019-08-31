import torch
import numpy as np
import torch.nn as nn
import patterns

class PatternConv2d(nn.Module):

    def __init__(self, conv_layer):
        super(PatternConv2d, self).__init__()

        if conv_layer.dilation != (1,1):

            def dilation_mask(kernel_size, dilation):
    
                mask = torch.zeros(kernel_size[0], kernel_size[1], 
                                kernel_size[2]+(dilation[0]-1)*(kernel_size[2]-1),
                                kernel_size[3]+(dilation[1]-1)*(kernel_size[3]-1))
                
                locs_x = np.arange(0, mask.shape[2], dilation[0])
                locs_y = np.arange(0, mask.shape[3], dilation[1])
                inds_x, inds_y = np.meshgrid(locs_x, locs_y)
                
                mask[:,:,inds_x, inds_y] = 1
                
                return mask

            self.dil_mask = lambda ks: dilation_mask(ks, conv_layer.dilation)
            
            
        
        self.forward_layer = conv_layer  # kernels size of forward layer: 
                                         # self.forward_layer.kernel_size
        padding_f = np.array(conv_layer.padding)
        ks = np.array(self.forward_layer.kernel_size) 
        padding_b = tuple(-padding_f + ks - 1)
        if conv_layer.dilation != (1,1):
            ks_dil = np.array(self.dil_mask(conv_layer.weight.shape).shape[-2:])
            padding_b = tuple(-padding_f + ks_dil - 1)
        self.backward_layer =  nn.Conv2d(
            conv_layer.out_channels,
            conv_layer.in_channels,
            ks,
            stride=conv_layer.stride,
            dilation=conv_layer.dilation,
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
        # what if the layer does not have a bias?
        if self.forward_layer.bias is None:
            return output
        bias = expand_bias(self.forward_layer.bias.data, output.data.shape)
        output_wo_bias = output - bias

        return output, output_wo_bias


    def backward(self, input, normalize_output=True):
        ''' compute a backward step (for signal computation).
        '''
        output = self.backward_layer(input)
        # if the dilation is not none the output has to be 
        # dilated to the original input size
        # if self.forward_layer.dilation != (1,1):
        #     output_mask = self.dil_mask(output.shape)
        #     print('Dilaten output shape:',output_mask.shape, 'Current output shape:', output.shape)
        #     output_dilated = torch.zeros(output_mask.shape)
        #     output_dilated[output_mask == 1] = torch.flatten(output)
        #     output = output_dilated
        if normalize_output:
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
        kernel_size = self.forward_layer.kernel_size
        if self.forward_layer.dilation != (1,1):
            dilation = self.forward_layer.dilation
            kernel_size = tuple((kernel_size[0]+(dilation[0]-1)*(kernel_size[0]-1),
                                kernel_size[1]+(dilation[1]-1)*(kernel_size[1]-1)))
        # print(output)

        if output_wo_bias is None:
            inp_dense, out_dense = patterns._conv_maps_to_dense(input, output,
                                                                kernel_size)
            if self.forward_layer.dilation != (1,1):
                inp_mask = torch.flatten(self.dil_mask(self.forward_layer.weight.shape)[0])
                inp_dense = inp_dense[:, inp_mask==1]

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
                                            kernel_size)
            _, out_dense = patterns._conv_maps_to_dense(input, output,
                                        kernel_size)
            if self.forward_layer.dilation != (1,1):
                inp_mask = torch.flatten(self.dil_mask(self.forward_layer.weight.shape)[0])
                inp_dense = inp_dense[:, inp_mask==1]
 
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
        kernel = self.forward_layer.weight.data
        self.patterns = patterns.compute_patterns_conv(self.statistics, 
                                                       kernel)


    def set_patterns(self, pattern_type='relu'):
        ''' Sets the computed patterns as the kernel of the backward layer.
            pattern_type can be 'relu' or 'linear'
        '''
        if pattern_type == 'relu':
            self.backward_layer.parameters().__next__().data = self.patterns['A_plus']
        elif pattern_type == 'linear':
            self.backward_layer.parameters().__next__().data = self.patterns['A_linear']


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
        if self.forward_layer.bias is None:
            return output
        bias = expand_bias(self.forward_layer.bias.data, output.data.shape)
        output_wo_bias = output - bias

        return output, output_wo_bias


    def backward(self, input, normalize_output=True):
        ''' compute a backward step (for signal computation).
        '''
        output = self.backward_layer(input)
        if normalize_output:
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
        w = self.forward_layer.weight.data
        self.patterns = patterns.compute_patterns_linear(self.statistics, w)


    def set_patterns(self, pattern_type='relu'):
        ''' Sets the computed patterns as the kernel of the backward layer.
            pattern_type can be 'relu' or 'linear'
        '''
        if pattern_type == 'relu':
            # self.backward_layer.weight.data = self.patterns['A_plus'].permute(1,0)
            self.backward_layer.parameters().__next__().data = self.patterns['A_plus'].permute(1,0)
        elif pattern_type == 'linear':
            self.backward_layer.parameters().__next__().data = self.patterns['A_linear'].permute(1,0)


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
        input = input.clone().detach()
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
        # kernel_size[0]+(dilation[0]-1)*(kernel_size[0]-1)
        ks = np.array(pool_layer.kernel_size)
        dil = np.array(pool_layer.dilation)
        ks_dil = ks + (dil - 1) * (ks - 1)
        self.backward_layer = nn.MaxUnpool2d(ks_dil, 
                                             pool_layer.stride, 
                                             pool_layer.padding)

    def forward(self, input):
        return self.forward_layer(input)

    def backward(self, input, switches, output_size=None):
        if output_size is None:
            return self.backward_layer(input, switches)
        else:
            return self.backward_layer(input, switches, output_size=output_size)


class PatternBatchNorm2d(torch.nn.Module):
    def __init__(self, bn_layer):
        super(PatternBatchNorm2d, self).__init__()
        
        self.forward_layer = bn_layer
        # TODO: use running mean and running var if existant
        # else need to compute new mean and var for each call
        if bn_layer.track_running_stats:
            self.mean = bn_layer.running_mean
            self.var = bn_layer.running_var
        else:
            self.mean = None
            self.var = None
        
    def forward(self, inp):
        # TODO: check if my own computation (if no running stats) is really the same as from the layer
        # --> did not check, but backward is just reversed computation and leads to same result as 
        # what's passed to the layer (up to 10^-5)
        if self.forward_layer.track_running_stats:
#             print('Using original layer for forward pass')
            return self.forward_layer(inp)
        
        else:
            self.mean = torch.mean(inp, dim=(0,2,3), keepdim=True)
            self.var = torch.var(inp, dim=(0,2,3), unbiased=False, keepdim=True)
            
            return (inp - self.mean) / (torch.sqrt(self.var + self.forward_layer.eps)) * \
                    self.forward_layer.weight.data + self.forward_layer.bias.data
        
    def backward(self, out):
        # forward computation: (x - mean) /(sqrt(var + eps))*gamma + beta
        # --> backward computation: (out - beta) / gamma * sqrt(var+eps) + mean
        
        tmp = out - self.forward_layer.bias.data[None,:,None,None]
        tmp /= self.forward_layer.weight.data[None,:,None,None]
        tmp *= torch.sqrt(self.var + self.forward_layer.eps)[None,:,None,None]
        tmp += self.mean[None,:,None,None]
        result = tmp
        
        return result
        
    