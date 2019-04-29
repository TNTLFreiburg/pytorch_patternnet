"""Perform pattern computation."""
import torch
import numpy as np

DTYPE = torch.float32
# statistics computation
def compute_statistics(X, Y, Y_wb, device=torch.device("cpu")):
    """
    Compute statistics of the given data e.g. mean over x and y, covariance of
    x and y.

    Statistics needed:
    for a linear:
        covariance of x and y for each neuron in y
        variance of y for each neuron in y
    for a+/a-:
        mean_y+ for each neuron in y
        mean_y- for each neuron in y
        means_x+ for each neuron in y
        means_x- for each neuron in y
        means_yx+ for each neuron in y
        means_yx- for each neuron in y

    Parameters
    ----------
    X: matrix with inputs to the layer
       shape: sxn with n number of neurons, s number
       samples
    Y: matrix with outputs of the layer, before relu
       or other functions applied
       shape: sxm with m number of neurons, s number
       of samples

    Returns
    -------
    dict with statistics

    """
    num_samples = X.size()[0]
    num_neurons_x = X.size()[1]
    num_neurons_y = Y.size()[1]
    # create a dictionary with three subdictionaries
    # one for the linear estimators and two for the two component estimator

    lin_dict = {
        "e_y": torch.zeros((1, num_neurons_y), device=device, dtype=DTYPE),
        "e_yy": torch.zeros((1, num_neurons_y), device=device, dtype=DTYPE),
        "e_x": torch.zeros(
            (num_neurons_y, num_neurons_x), device=device, dtype=DTYPE
        ),
        "e_xy": torch.zeros(
            (num_neurons_y, num_neurons_x), device=device, dtype=DTYPE
        ),
        "cnt" : X.shape[0]
    }
    pos_dict = {
        "e_xy": torch.zeros(
            (num_neurons_y, num_neurons_x), device=device, dtype=DTYPE
        ),
        "e_x": torch.zeros(
            (num_neurons_y, num_neurons_x), device=device, dtype=DTYPE
        ),
        "e_y": torch.zeros((1, num_neurons_y), device=device, dtype=DTYPE),
        "cnt" : torch.zeros(num_neurons_y, device=device, dtype=DTYPE)
    }

    stat_dict = {"linear": lin_dict, "positive": pos_dict}

    e_x_linear = torch.mean(X, dim=0)
    e_x_linear = e_x_linear.type(DTYPE)
    stat_dict["linear"]["e_x"] = e_x_linear.expand(
        num_neurons_y, num_neurons_x
    )

    for i in range(num_neurons_y):
        # for each neuron and each estimator (component) compute the expected
        # value of x and y and the expected values of x and y, respectively
        # start with the linear ones:
        _, e_y_linear, e_xy_linear = _mean_x_y_xy(X, Y[:, i])

        yy = torch.mul(Y[:, i], Y[:, i])
        e_yy = torch.mean(yy)
        stat_dict["linear"]["e_yy"][0, i] = e_yy
        stat_dict["linear"]["e_y"][0, i] = e_y_linear
        stat_dict["linear"]["e_xy"][i, :] = e_xy_linear

        # now the positive ones:
        # first find the samples with positive y-value
        ind_pos = torch.squeeze(torch.nonzero(Y_wb[:, i] > 0))
        ind_pos_wob = torch.squeeze(torch.nonzero(Y[:, i] > 0))

        # paper seems to use e_y over all y-values, not just the positive ones
        stat_dict["positive"]["e_y"][0, i] = e_y_linear
        if ind_pos.size() == torch.Size([0]):
            # device needed here?
            stat_dict["positive"]["e_xy"][i, :] = torch.zeros(
                (1, num_neurons_x), device=device, dtype=DTYPE
            )
            stat_dict["positive"]["e_x"][i, :] = torch.zeros(
                (1, num_neurons_x), device=device, dtype=DTYPE
            )
            # stat_dict["positive"]["e_y"][0, i] = 0

        else:
            # select only samples with positive y-value in x
            x_plus = torch.index_select(X, 0, ind_pos)
            # select only samples with positive y-value in y
            y_plus = torch.index_select(Y[:, i], 0, ind_pos)

            cnt = (
                1
                if ind_pos.shape == torch.Size([])
                else ind_pos.shape[0]
            )
            stat_dict["positive"]["cnt"][i] = cnt
            e_x_pos, e_y_pos, e_xy_pos = _mean_x_y_xy(x_plus, y_plus)

            stat_dict["positive"]["e_xy"][i, :] = e_xy_pos
            stat_dict["positive"]["e_x"][i, :] = e_x_pos
            # stat_dict["positive"]["e_y"][0, i] = e_y_pos

    return stat_dict


def update_statistics(
    X, Y, Y_wb, stats_dict, device=torch.device("cpu")
):
    """ Perform a linear update of the statistics in `stats_dict` with the new
        data given in `X`, `Y` and `Y_wb`.

        Parameters
        ----------
        X: torch.FloatTensor
            input to layer for which statistics should be computed
        Y: torch.FloatTensor
            output of layer for which statistics should be comptuted
        Y_wb: torch.FloatTensor
            output of layer for which statistics should be computed minus bias
            of the layer
        stats_dict: dict
            Statistics dictionary as returned from `compute_statistics`. Needs
            to have been computed on a layer with the same input and output 
            shapes as given by `X` and `Y`.
        device: torch.device
            device to use for the statistics update computation

        Returns
        -------
        dict
            dictionary with the updated statistics

    """
    # first compute new statistics
    new_stats = compute_statistics(X, Y, Y_wb, device)

    # compute old and new sample factors for linear and positive statistics
    factor_old_linear = stats_dict['linear']['cnt'] / (stats_dict['linear']['cnt'] \
                                                    + new_stats['linear']['cnt'])
    factor_new_linear = 1 - factor_old_linear
    # for the positive statistics start with zeros to avoid having to divide by zero
    factor_old_positive = torch.zeros(stats_dict['positive']['cnt'].shape)
    inds_nonzero = torch.squeeze(stats_dict['positive']['cnt'].nonzero())
    factor_old_positive[inds_nonzero] = stats_dict['positive']['cnt'][inds_nonzero] \
                                        / (stats_dict['positive']['cnt'][inds_nonzero] + \
                                           new_stats['positive']['cnt'][inds_nonzero])
    factor_new_positive = torch.ones(stats_dict['positive']['cnt'].shape) - \
                          factor_old_positive

    for param_type in stats_dict:
        for param in stats_dict[param_type]:
            # cnt only needed for factors
            if param == 'cnt':
                continue
            # use linear factors
            if param_type == 'linear' or param == 'e_y':
                new_stats[param_type][param] = factor_old_linear * stats_dict[param_type][param] \
                                             + factor_new_linear * new_stats[param_type][param]
            # use factors for plus/minus-patterns
            else:
                old_stats_weighted = _rowwise_mul(stats_dict[param_type][param], 
                                                factor_old_positive)
                new_stats_weighted = _rowwise_mul(new_stats[param_type][param], 
                                                factor_new_positive)
                new_stats[param_type][param] = old_stats_weighted + new_stats_weighted
  
    # before returning new_stats the cnt has to be updated as well
    new_stats['linear']['cnt'] += stats_dict['linear']['cnt']
    new_stats['positive']['cnt'] += stats_dict['positive']['cnt']

    return new_stats


# pattern computation
def _compute_a(weights, stats):
    """ Computes the linear and +/- patterns.

        Parameters
        ----------
        weights: torch.FloatTensor
            layer weights. For conv layers must be transformed to 
            two-dimensional weight matrix
        stats: dict
            dictionary with the computed statistics

        Returns
        -------
        dict
            dictionary with the linear and +/- patterns
    """

    # for the linear estimator
    # the variance of y
    weights = weights.type(DTYPE)
    var_y = (
        stats["linear"]["e_yy"]
        - stats["linear"]["e_y"] * stats["linear"]["e_y"]
    )
    # the covariance between x and y
    cov_xy = stats["linear"]["e_xy"] - _rowwise_mul(
        stats["linear"]["e_x"], torch.squeeze(stats["linear"]["e_y"])
    )
    # the linear estimator is cov/var_y
    a_linear = _rowwise_div(cov_xy, torch.squeeze(var_y))

    # for the plus-minus estimator
    # get Ex*Ey
    ex_ey = _rowwise_mul(
        stats["positive"]["e_x"], torch.squeeze(stats["positive"]["e_y"])
    )
    # get the nominator of the whole formula: Exy - Ex*Ey
    nom_a_plus = stats["positive"]["e_xy"] - ex_ey
    # now the denominator
    w_exy = _rowwise_mul(weights, stats["positive"]["e_xy"])
    w_ex_ey = _rowwise_mul(weights, ex_ey)
    denom_a_plus = w_exy - w_ex_ey
    # now divide the nominator by the denominator
    a_plus = _rowwise_div(nom_a_plus, denom_a_plus)

    a_dict = {"A_linear": a_linear, "A_plus": a_plus}

    return a_dict


def compute_patterns_linear(stats, weights):
    """ Computes patterns for dense layers.

        Parameters
        ----------
        stats: dict
            dictionary with the computed statistics
        weights: torch.FloatTensor
            weight matrix of the dense layer

        Returns
        -------
        dict
            dictionary with the linear and +/- patterns of the layer
    """
    a_dict = _compute_a(weights, stats)

    return a_dict


def compute_patterns_conv(stats, kernel):
    """ Computes patterns for conv2d layers.

        Parameters
        ----------
        stats: dict
            dictionary with the computed statistics
        kernel: torch.FloatTensor
            kernel weights of the conv2d layer

        Returns
        -------
        dict
            dictionary with the linear and +/- patterns of the layer
    """

    # convert kernel to weight matrix
    k_as_w = _conv_kernel_to_dense(kernel)
    # print("Conv kernel as dense weight matrix:")
    # print(k_as_w.shape)
    # print(k_as_w)
    # compute patterns from weight matrix and statistics
    a_dict = _compute_a(k_as_w, stats)

    # get size of pattern kernel in right order st view works along the right
    # dimensions
    k_s = (kernel.shape[0], kernel.shape[2], kernel.shape[3], kernel.shape[1])
    # k_s = (kernel.shape[1], kernel.shape[0], kernel.shape[2], kernel.shape[3])
    # convert pattern matrix to pattern kernel
    a_dict["A_linear"] = a_dict["A_linear"].contiguous().view(k_s)
    a_dict["A_plus"] = a_dict["A_plus"].contiguous().view(k_s)
    a_dict["A_linear"] = a_dict["A_linear"].permute(3, 0, 1, 2)
    a_dict["A_plus"] = a_dict["A_plus"].permute(3, 0, 1, 2)

    # the kernels have to be reversed along the height and width dimension
    def revert_tensor(tensor, axis=0):
        idx = [i for i in range(tensor.size(axis) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        return tensor.index_select(axis, idx)

    a_dict["A_linear"] = revert_tensor(a_dict["A_linear"], 2)
    a_dict["A_linear"] = revert_tensor(a_dict["A_linear"], 3)
    a_dict["A_plus"] = revert_tensor(a_dict["A_plus"], 2)
    a_dict["A_plus"] = revert_tensor(a_dict["A_plus"], 3)

    return a_dict


###############################################################################
############  helper functions  ###############################################
###############################################################################


def _mean_x_y_xy(x, y):
    """ Compute the columnwise mean of x, the mean of y and the mean of x*y."""
    # x = x.type(torch.float32)
    # y = y.type(torch.float32)

    # m_x = torch.sum(x, dim=0, dtype=torch.float64) / x.shape[0]
    m_x = torch.mean(x, dim=0)
    # m_y = torch.sum(y, dtype=torch.float64) / y.shape[0]
    m_y = torch.mean(y)

    xy = x * torch.t(y.expand_as(torch.t(x)))
    m_xy = torch.mean(xy, dim=0)
    # m_xy = torch.sum(xy, dim=0) / xy.shape[0]

    return m_x, m_y, m_xy


def _rowwise_mul(matrix, other):
    """ Compute the rowwise multiplication of `matrix` and `other`
    
        `other` can be a scalar, a vector or a matrix
        - if `other` is a scalar the matrix is multiplied by the scalar
        - if `other` is a vector each row of `matrix` is multiplied with the 
          corresponding scalar value in the vector
        - if `other` is a matrix the dot product of the corresponding rows of
          the two matrices is computed

        Parameters
        ----------
        matrix: torch.FloatTensor
            matrix which should be multiplied with sth else rowwise
        other: torch.FloatTensor or float
            scalar, vector or matrix with which `matrix` should be multiplied.

        Returns
        -------
        torch.FloatTensor
            result of the rowwise multiplication
    """

    # compute the rowwise multiplication by seperately
    # going throuth the rows

    if other.dim() == 0:
        result = matrix * other
    elif other.dim() == 1:
        result = torch.zeros(matrix.size(), dtype=DTYPE)
        for i in range(matrix.size()[0]):
            result[i, :] = matrix[i, :] * other[i]
    else:
        result = torch.zeros(matrix.size()[0], dtype=DTYPE)
        for i in range(matrix.size()[0]):
            result[i] = torch.dot(matrix[i, :], other[i, :])

    return result


def _rowwise_div(matrix, vector):
    # need to implement a check for zeros in vector

    result = torch.zeros(matrix.size(), dtype=DTYPE)
    for i in range(matrix.size()[0]):
        # if there's a 0 in vector we just set the result
        # to 0 or leave the values that are in matrix?
        if vector[i] == 0:
            result[i, :] = matrix[i, :]
        else:
            result[i, :] = matrix[i, :] * (1 / vector[i])

    return result



    """ reshapes the input map, kernels and output map of a convolutional
        layer to a dense layer
        converts a convolutional layer to a fully connected layer,
        i.e. a layer where the output is a simple matrix
        multiplication of the weight matrix and the input
        Note that the input and output of the resulting fc layer
        are two dimensional for one sample already which is one
        dimension more than for a typical fully connected layer
        PADDING NOT IMPLEMENTED YET!!!

        input:
        inp: input to the convolutional layer
             shape: batch_size x inp_channels x height x width
        kernels: kernels of the convolutional layer
                 shape: out_channels x inp_channels x kernel x kernel
        stride: stride size
        padding: how many padding values are added to each side


        output:
        inp_fc: input as fc layer
        weights: kernels as weight matrix
        out_fc: output of the layer as fc layer

    """
    i_s = inp.size()
    k_s = kernels.size()
    # patch size
    num_rows = k_s[1] * k_s[2] * k_s[3]
    num_cols = int((i_s[2] - k_s[2]) / stride + 1) * int(
        (i_s[3] - k_s[3]) / stride + 1
    )
    # print(num_rows,num_cols)
    weight_matrix_height = k_s[0]  # number of kernels
    weight_matrix_width = k_s[1] * k_s[2] * k_s[3]  # one kernel reshaped

    # batch_size at first dimension
    inp_fc = torch.zeros(i_s[0], num_rows, num_cols)
    weights = torch.zeros(weight_matrix_height, weight_matrix_width)

    # now insert all the patches into inp_fc
    col_count = 0
    for i in range(0, i_s[2] - k_s[2] + 1, stride):
        for j in range(0, i_s[3] - k_s[3] + 1, stride):
            # print(
            #     (
            #         (inp[:, :, i : i + k_s[2], j : j + k_s[3]])
            #         .contiguous()
            #         .view(i_s[0], num_rows, -1)
            #         .size()
            #     )
            # )
            # print(inp_fc[:, :, col_count].size())
            inp_fc[:, :, [col_count]] = (
                (inp[:, :, i : i + k_s[2], j : j + k_s[3]]).contiguous()
            ).view(i_s[0], num_rows, -1)
            col_count += 1

    # now flatten the kernels and save them into the weight matrix
    for k in range(k_s[0]):
        weights[k, :] = kernels[k, :, :, :].contiguous().view(1, -1)

    # now the output can be computed as a simple matrix multiplication
    out_fc = torch.matmul(weights, inp_fc)

    return inp_fc, weights, out_fc


def _conv_maps_to_dense(
    inp, out, kernel_size, stride=1, padding=None, device=torch.device("cpu")
):
    """
    Transform convolutional input and output map to dense maps

    Parameters
    ----------
    inp: convolutional input map of a layer
    out: convolutioanl output map of a layer
    kernel_size: size of kernels used in this layer
    padding: number of zeros used for padding, if None, no padding

    Returns
    -------
    inp_dense: input map as dense map
    out_dense: output map as dense map

    """
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if theres padding there are more patches -> first convert input map to
    # input map with zeros padded, then do the rest as before?
    if padding is not None:
        i_s = inp.size()
        # zeros on the right and left of each map
        zeros_rl = torch.zeros(
            (i_s[0], i_s[1], i_s[2], padding), device=device
        )
        # zeros at the top and bottom of each map
        zeros_tb = torch.zeros(
            (i_s[0], i_s[1], padding, i_s[3] + 2 * padding), device=device
        )

        inp = torch.cat((zeros_rl, inp, zeros_rl), dim=3)
        inp = torch.cat((zeros_tb, inp, zeros_tb), dim=2)
    # conv map shape: samples x channels x height x width
    i_s = inp.size()
    o_s = out.size()

    num_rows = i_s[1] * kernel_size[0] * kernel_size[1]
    num_cols = int((i_s[2] - kernel_size[0]) / stride + 1) * int(
        (i_s[3] - kernel_size[1]) / stride + 1
    )

    inp_dense = torch.zeros((i_s[0], num_rows, num_cols), device=device)
    col_count = 0
    for i in range(0, i_s[2] - kernel_size[0] + 1, stride):
        for j in range(0, i_s[3] - kernel_size[1] + 1, stride):
            patch = (
                inp[:, :, i : i + kernel_size[0], j : j + kernel_size[1]]
            ).contiguous()
            # INNVESTIGATE: next line added for innvestigate similarity
            patch = patch.permute(0, 2, 3, 1).contiguous()
            #############################################
            patch_reshaped = patch.view(i_s[0], num_rows)
            # patch_reshaped = patch.view(i_s[0], num_rows, -1).squeeze()
            inp_dense[:, :, col_count] = patch_reshaped
            col_count += 1

    # out_dense = torch.zeros(o_s[0],o_s[1],o_s[2]*o_s[3])
    out_dense = out.view(o_s[0], o_s[1], o_s[2] * o_s[3])

    # reshaping so that the maps can directly be used for the statistics
    # computation
    inp_dense_t = torch.transpose(inp_dense, 1, 2).contiguous()
    inp_stat = inp_dense_t.view(i_s[0] * num_cols, -1)

    out_dense_t = torch.transpose(out_dense, 1, 2).contiguous()
    out_stat = out_dense_t.view(o_s[0] * o_s[2] * o_s[3], -1)

    return inp_stat, out_stat


def _conv_kernel_to_dense(kernels):
    k_s = kernels.size()

    kernels = kernels.contiguous()
    # first permute the order, then reshape the kernel
    kernels_permuted = kernels.permute(0, 2, 3, 1).contiguous()
    weights = kernels_permuted.view(k_s[0], k_s[1] * k_s[2] * k_s[3])

    return weights
