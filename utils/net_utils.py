import torch
import torch.nn as nn
from config.load_parser import load_parser
import copy
import numpy as np
from models.generators import GetSubnet

opt = load_parser()
c = 1 - np.exp(
    (opt.freezing_period * opt.project_freq / opt.niter) * np.log(opt.sparsity)
)
imp_c = 1 - np.exp((opt.rewind / opt.prune_Id) * np.log(opt.sparsity))


def freeze_model_weights(model):
    print("=> Freezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            # print(f"==> No gradient to {n}.weight")
            m.weight.requires_grad = False
            if m.weight.grad is not None:
                # print(f"==> Setting gradient of {n}.weight to None")
                m.weight.grad = None

            if hasattr(m, "bias") and m.bias is not None:
                # print(f"==> No gradient to {n}.bias")
                m.bias.requires_grad = False

                if m.bias.grad is not None:
                    # print(f"==> Setting gradient of {n}.bias to None")
                    m.bias.grad = None


def freeze_model_subnet(model):
    print("=> Freezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            m.scores.requires_grad = False
            # print(f"==> No gradient to {n}.scores")
            if m.scores.grad is not None:
                # print(f"==> Setting gradient of {n}.scores to None")
                m.scores.grad = None


def unfreeze_model_weights(model):
    print("=> Unfreezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            # print(f"==> Gradient to {n}.weight")
            m.weight.requires_grad = True
            if hasattr(m, "bias") and m.bias is not None:
                # print(f"==> Gradient to {n}.bias")
                m.bias.requires_grad = True


def unfreeze_model_subnet(model):
    print("=> Unfreezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            # print(f"==> Gradient to {n}.scores")
            m.scores.requires_grad = True


def round_model(model, round_scheme, noise=False, ratio=0.0, rank=None):
    print("Rounding model with scheme: {}".format(round_scheme))
    if isinstance(model, nn.parallel.DistributedDataParallel):
        cp_model = copy.deepcopy(model.module)
    else:
        cp_model = copy.deepcopy(model)
    for name, params in cp_model.named_parameters():
        if ".score" in name:
            if noise:
                delta = torch.randn_like(params.data) * ratio
                params.data += delta

            params.data = (
                torch.gt(params.data, torch.ones_like(params.data) * c).int().float()
            )

            """    
            if noise:
                delta = torch.bernoulli(torch.ones_like(params.data)*ratio)
                params.data = (params.data + delta) % 2
            """

    if isinstance(model, nn.parallel.DistributedDataParallel):
        cp_model = nn.parallel.DistributedDataParallel(
            cp_model, device_ids=[rank], find_unused_parameters=True
        )

    return cp_model


def prune(
    model,
    update_thresholds_only=False,
    update_scores=False,
    drop_bottom_half_weights=False,
):
    if update_thresholds_only:
        pass
        # print("Updating prune thresholds")
    else:
        print("Pruning Model:")

    scores_threshold = bias_scores_threshold = -np.inf

    if opt.FLfreeze == True:
        conv_layers, linear_layers = get_sublayers(opt.netGType, model)
    else:
        conv_layers, linear_layers = get_layers(opt.netGType, model)

    ##########
    if opt.algorithm == "imp":
        for layer in conv_layers + linear_layers:
            layer.scores.data = layer.weight.data * layer.flag.data
            if opt.bias:
                layer.bias_scores.data = layer.bias.data * layer.bias_flag.data
    ###########

    num_active_weights = 0
    num_active_biases = 0
    active_scores_list = []
    active_bias_scores_list = []
    for layer in conv_layers + linear_layers:
        num_active_weights += layer.flag.data.sum().item()
        active_scores = (layer.scores.data[layer.flag.data == 1]).clone()
        active_scores_list.append(active_scores)
        if opt.bias:
            num_active_biases += layer.bias_flag.data.sum().item()
            active_biases = (layer.bias_scores.data[layer.bias_flag.data == 1]).clone()
            active_bias_scores_list.append(active_biases)
    if opt.algorithm == "gm":
        number_of_weights_to_prune = np.ceil(c * num_active_weights).astype(int)
        number_of_biases_to_prune = np.ceil(c * num_active_biases).astype(int)
    elif opt.algorithm == "imp":
        number_of_weights_to_prune = np.ceil(imp_c * num_active_weights).astype(int)
        number_of_biases_to_prune = np.ceil(imp_c * num_active_biases).astype(int)
    elif opt.algorithm == "global_ep":
        number_of_weights_to_prune = np.ceil(
            (1 - opt.sparsity) * num_active_weights
        ).astype(int)
        number_of_biases_to_prune = np.ceil(
            (1 - opt.sparsity) * num_active_biases
        ).astype(int)
    elif opt.algorithm == "ep":
        number_of_weights_to_prune = np.ceil(
            (1 - opt.sparsity) * num_active_weights
        ).astype(int)
        number_of_biases_to_prune = np.ceil(
            (1 - opt.sparsity) * num_active_biases
        ).astype(int)

    else:
        print("algorithm error")

    agg_scores = torch.cat(active_scores_list)
    agg_bias_scores = (
        torch.cat(active_bias_scores_list) if opt.bias else torch.tensor([])
    )

    # if invert_sanity_check, then threshold is based on sorted scores in descending order, and we prune all scores ABOVE it
    scores_threshold = (
        torch.sort(torch.abs(agg_scores), descending=False)
        .values[number_of_weights_to_prune - 1]
        .item()
    )
    # print(scores_threshold)
    if opt.bias:
        bias_scores_threshold = (
            torch.sort(torch.abs(agg_bias_scores), descending=False)
            .values[number_of_biases_to_prune - 1]
            .item()
        )
    else:
        bias_scores_threshold = -1

    if update_thresholds_only:
        for layer in conv_layers + linear_layers:
            layer.scores_prune_threshold = scores_threshold
            if opt.bias:
                layer.bias_scores_prune_threshold = bias_scores_threshold
            layer.flag.data, layer.bias_flag.data = GetSubnet.apply(
                layer.scores.abs(),
                layer.bias_scores.abs(),
                opt.sparsity,
                scores_threshold,
                bias_scores_threshold,
            )

    else:
        for layer in conv_layers + linear_layers:
            if opt.algorithm == "ep":
                layer.flag.data, layer.bias_flag.data = GetSubnet.apply(
                    layer.scores.abs(), layer.bias_scores.abs(), opt.sparsity
                )
            else:
                layer.flag.data = (
                    (
                        layer.flag.data
                        + torch.gt(
                            layer.scores.abs(),  # TODO
                            torch.ones_like(layer.scores) * scores_threshold,
                        ).int()
                        == 2
                    ).int()
                ).float()
                if update_scores:
                    layer.scores.data = layer.scores.data * layer.flag.data
                if opt.bias:
                    layer.bias_flag.data = (
                        (
                            layer.bias_flag.data
                            + torch.gt(
                                layer.bias_scores,
                                torch.ones_like(layer.bias_scores)
                                * bias_scores_threshold,
                            ).int()
                            == 2
                        ).int()
                    ).float()
                    if update_scores:
                        layer.bias_scores.data = (
                            layer.bias_scores.data * layer.bias_flag.data
                        )
        if opt.algorithm == "imp":
            layer.flag.data = (
                (
                    layer.flag.data
                    + torch.gt(
                        layer.scores.abs(),  # TODO
                        torch.ones_like(layer.scores) * scores_threshold,
                    ).int()
                    == 2
                ).int()
            ).float()
            for layer in conv_layers + linear_layers:
                layer.weight.data = layer.weight.data * layer.flag.data
                if opt.bias:
                    layer.bias.data = layer.bias.data * layer.bias_flag.data
    return scores_threshold, bias_scores_threshold


def get_sub_param(model, threshold=0):
    total_numer = 0
    total_denom = 0
    conv_layers, linear_layers = get_sublayers(opt.netGType, model)
    for layer in conv_layers + linear_layers:
        numer = torch.sum(layer.flag)
        denom = torch.sum(torch.ones_like(layer.flag))
        total_numer += numer
        total_denom += denom
        if opt.bias:
            numer += torch.sum(layer.bias_flag)
            denom += torch.sum(torch.ones_like(layer.bias_flag))
    return total_denom


def get_total_param(model, threshold=0):
    total_numer = 0
    total_denom = 0
    conv_layers, linear_layers = get_layers(opt.netGType, model)
    for layer in conv_layers + linear_layers:
        numer = torch.sum(layer.flag)
        denom = torch.sum(torch.ones_like(layer.flag))
        total_numer += numer
        total_denom += denom
        if opt.bias:
            numer += torch.sum(layer.bias_flag)
            denom += torch.sum(torch.ones_like(layer.bias_flag))
        print(layer, "'s params: ", denom)
    return total_denom


def reset_flag(model):
    if opt.FLfreeze == True:
        conv_layers, linear_layers = get_sublayers(opt.netGType, model)
    else:
        conv_layers, linear_layers = get_layers(opt.netGType, model)
    for layer in conv_layers + linear_layers:
        layer.flag.data = torch.ones_like(layer.scores).float()
        if opt.bias:
            layer.bias_flag.data = torch.ones_like(layer.bias_scores).float()


def get_model_sparsity(model, threshold=0):
    conv_layers, linear_layers = get_layers(opt.netGType, model)
    numer = 0
    denom = 0

    # TODO: find a nicer way to do this (skip dropout)
    # TODO: Update: can't use .children() or .named_modules() because of the way things are wrapped in builder
    for conv_layer in conv_layers:
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(conv_layer, threshold)
        numer += w_numer
        denom += w_denom
        if opt.bias:
            numer += b_numer
            denom += b_denom

    for lin_layer in linear_layers:
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(lin_layer, threshold)
        numer += w_numer
        denom += w_denom
        if opt.bias:
            numer += b_numer
            denom += b_denom
    # print('Overall sparsity: {}/{} ({:.2f} %)'.format((int)(numer), denom, 100*numer/denom))
    return 100 * numer / denom


# returns num_nonzero elements, total_num_elements so that it is easier to compute
# average sparsity in the end
def layer_sparsity(model, threshold=0):
    total_numer = 0
    total_denom = 0
    conv_layers, linear_layers = get_layers(opt.netGType, model)
    for layer in conv_layers + linear_layers:
        numer = torch.sum(layer.flag)
        denom = torch.sum(torch.ones_like(layer.flag))
        total_numer += numer
        total_denom += denom
        if opt.bias:
            numer += torch.sum(layer.bias_flag)
            denom += torch.sum(torch.ones_like(layer.bias_flag))
        print(layer, "'s sparsity", 100 * numer / denom)
    print("total sparsity: ", 100 * total_numer / total_denom)


def get_layer_sparsity(layer, threshold=0):

    eff_scores = layer.flag * layer.scores  # layer.scores * layer.flag
    if opt.bias:
        eff_bias_scores = (
            layer.bias_scores * layer.bias_flag
        )  # layer.bias_scores * layer.bias_flag
    num_middle = torch.sum(
        torch.gt(eff_scores, torch.ones_like(eff_scores) * threshold)
        * torch.lt(
            eff_scores, torch.ones_like(eff_scores.detach() * (1 - threshold)).int()
        )
    )

    if num_middle > 0:
        print("WARNING: Model scores are not binary. Sparsity number is unreliable.")
        raise ValueError
    w_numer, w_denom = (
        eff_scores.detach().sum().item(),
        eff_scores.detach().flatten().numel(),
    )

    if opt.bias:
        b_numer, b_denom = (
            eff_bias_scores.detach().sum().item(),
            eff_bias_scores.detach().flatten().numel(),
        )
    else:
        b_numer, b_denom = 0, 0

    return w_numer, w_denom, b_numer, b_denom


def get_num_param(model):
    if opt.distributed:
        model = model.module
    total_params = sum(param.numel() for param in model.parameters())
    effect_param = total_params / 3
    return effect_param


def get_sublayers(arch, model=None):
    if opt.distributed:
        model = model.module
    if arch == "resnet":
        conv_layers = [model.net.firstConv]
        conv_layers.append(model.net.resblock_0.conv1)
        conv_layers.append(model.net.resblock_0.conv2)
        conv_layers.append(model.net.resblock_0.conv_bypass)
        conv_layers.append(model.net.resblock_0.model[3])
        conv_layers.append(model.net.resblock_0.model[6])
        conv_layers.append(model.net.resblock_0.bypass[0])
        conv_layers.append(model.net.resblock_1.conv1)
        conv_layers.append(model.net.resblock_1.conv2)
        conv_layers.append(model.net.resblock_1.conv_bypass)
        conv_layers.append(model.net.resblock_1.model[3])
        conv_layers.append(model.net.resblock_1.model[6])
        conv_layers.append(model.net.resblock_1.bypass[0])
        conv_layers.append(model.net.resblock_2.conv1)
        conv_layers.append(model.net.resblock_2.conv2)
        conv_layers.append(model.net.resblock_2.conv_bypass)
        conv_layers.append(model.net.resblock_2.model[3])
        conv_layers.append(model.net.resblock_2.model[6])
        conv_layers.append(model.net.resblock_2.bypass[0])
        conv_layers.append(model.net.resblock_3.conv1)
        conv_layers.append(model.net.resblock_3.conv2)
        conv_layers.append(model.net.resblock_3.conv_bypass)
        conv_layers.append(model.net.resblock_3.model[3])
        conv_layers.append(model.net.resblock_3.model[6])
        conv_layers.append(model.net.resblock_3.bypass[0])
        conv_layers.append(model.net.lastConv)
        linear_layers = []

    elif arch == "sngan":
        conv_layers = []
        linear_layers = []

        conv_layers.append(model.block2.c1)
        conv_layers.append(model.block2.c2)
        conv_layers.append(model.block2.c_sc)
        conv_layers.append(model.block3.c1)
        conv_layers.append(model.block3.c2)
        conv_layers.append(model.block3.c_sc)
        conv_layers.append(model.block4.c1)
        conv_layers.append(model.block4.c2)
        conv_layers.append(model.block4.c_sc)
        conv_layers.append(model.block5.c1)
        conv_layers.append(model.block5.c2)
        conv_layers.append(model.block5.c_sc)
        if opt.First_freeze != True:
            linear_layers.append(model.l1)
        if opt.Last_freeze != True:
            conv_layers.append(model.c5)

    return (conv_layers, linear_layers)


def get_fixlayers(arch, model=None):
    if opt.distributed:
        model = model.module
    if arch == "resnet":
        conv_layers = [model.net.firstConv]
        conv_layers.append(model.net.resblock_0.conv1)
        conv_layers.append(model.net.resblock_0.conv2)
        conv_layers.append(model.net.resblock_0.conv_bypass)
        conv_layers.append(model.net.resblock_0.model[3])
        conv_layers.append(model.net.resblock_0.model[6])
        conv_layers.append(model.net.resblock_0.bypass[0])
        conv_layers.append(model.net.resblock_1.conv1)
        conv_layers.append(model.net.resblock_1.conv2)
        conv_layers.append(model.net.resblock_1.conv_bypass)
        conv_layers.append(model.net.resblock_1.model[3])
        conv_layers.append(model.net.resblock_1.model[6])
        conv_layers.append(model.net.resblock_1.bypass[0])
        conv_layers.append(model.net.resblock_2.conv1)
        conv_layers.append(model.net.resblock_2.conv2)
        conv_layers.append(model.net.resblock_2.conv_bypass)
        conv_layers.append(model.net.resblock_2.model[3])
        conv_layers.append(model.net.resblock_2.model[6])
        conv_layers.append(model.net.resblock_2.bypass[0])
        conv_layers.append(model.net.resblock_3.conv1)
        conv_layers.append(model.net.resblock_3.conv2)
        conv_layers.append(model.net.resblock_3.conv_bypass)
        conv_layers.append(model.net.resblock_3.model[3])
        conv_layers.append(model.net.resblock_3.model[6])
        conv_layers.append(model.net.resblock_3.bypass[0])
        conv_layers.append(model.net.lastConv)
        linear_layers = []

    elif arch == "sngan":
        conv_layers = []
        linear_layers = []
        if opt.First_freeze == True:
            linear_layers.append(model.l1)
        if opt.Last_freeze == True:
            conv_layers.append(model.c5)

    return (conv_layers, linear_layers)


def get_layers(arch, model=None):
    if opt.distributed:
        model = model.module
    if arch == "resnet":
        conv_layers = [model.net.firstConv]
        conv_layers.append(model.net.resblock_0.conv1)
        conv_layers.append(model.net.resblock_0.conv2)
        conv_layers.append(model.net.resblock_0.conv_bypass)
        conv_layers.append(model.net.resblock_0.model[3])
        conv_layers.append(model.net.resblock_0.model[6])
        conv_layers.append(model.net.resblock_0.bypass[0])
        conv_layers.append(model.net.resblock_1.conv1)
        conv_layers.append(model.net.resblock_1.conv2)
        conv_layers.append(model.net.resblock_1.conv_bypass)
        conv_layers.append(model.net.resblock_1.model[3])
        conv_layers.append(model.net.resblock_1.model[6])
        conv_layers.append(model.net.resblock_1.bypass[0])
        conv_layers.append(model.net.resblock_2.conv1)
        conv_layers.append(model.net.resblock_2.conv2)
        conv_layers.append(model.net.resblock_2.conv_bypass)
        conv_layers.append(model.net.resblock_2.model[3])
        conv_layers.append(model.net.resblock_2.model[6])
        conv_layers.append(model.net.resblock_2.bypass[0])
        conv_layers.append(model.net.resblock_3.conv1)
        conv_layers.append(model.net.resblock_3.conv2)
        conv_layers.append(model.net.resblock_3.conv_bypass)
        conv_layers.append(model.net.resblock_3.model[3])
        conv_layers.append(model.net.resblock_3.model[6])
        conv_layers.append(model.net.resblock_3.bypass[0])
        conv_layers.append(model.net.lastConv)
        linear_layers = []

    elif arch == "sngan":
        conv_layers = []
        linear_layers = []

        conv_layers.append(model.block2.c1)
        conv_layers.append(model.block2.c2)
        conv_layers.append(model.block2.c_sc)
        conv_layers.append(model.block3.c1)
        conv_layers.append(model.block3.c2)
        conv_layers.append(model.block3.c_sc)
        conv_layers.append(model.block4.c1)
        conv_layers.append(model.block4.c2)
        conv_layers.append(model.block4.c_sc)
        conv_layers.append(model.block5.c1)
        conv_layers.append(model.block5.c2)
        conv_layers.append(model.block5.c_sc)
        conv_layers.append(model.c5)

        linear_layers.append(model.l1)

    return (conv_layers, linear_layers)


def get_regularization_loss(
    model, regularizer="L2", lmbda=1, alpha=1, alpha_prime=1, device=None
):

    conv_layers, linear_layers = get_layers(opt.netGType, model)

    if opt.distributed:
        model = model.module

    def get_special_reg_sum(layer):
        # reg_loss =  \sum_{i} w_i^2 * p_i(1-p_i)
        # NOTE: alpha = alpha' = 1 here. Change if needed.
        reg_sum = torch.tensor(0.0).to(device)
        w_i = layer.weight
        p_i = layer.scores
        reg_sum += torch.sum(
            torch.pow(w_i, 2) * torch.pow(p_i, 1) * torch.pow(1 - p_i, 1)
        )
        if layer.bias != None:
            b_i = layer.bias
            p_i = layer.bias_scores
            reg_sum += torch.sum(
                torch.pow(b_i, 2) * torch.pow(p_i, 1) * torch.pow(1 - p_i, 1)
            )
        return reg_sum

    # pdb.set_trace()
    regularization_loss = torch.tensor(0.0).to(device)
    if regularizer == "L2":
        # reg_loss =  ||p||_2^2
        for name, params in model.named_parameters():
            if ".bias_score" in name:
                if params.shape != torch.Tensor(1).shape:
                    regularization_loss += torch.norm(params, p=2) ** 2

            elif ".score" in name:
                regularization_loss += torch.norm(params, p=2) ** 2
        regularization_loss = lmbda * regularization_loss

    elif regularizer == "L1":
        # reg_loss =  ||p||_1
        for name, params in model.named_parameters():
            if ".bias_score" in name:
                if params.shape != torch.Tensor(1).shape:
                    regularization_loss += torch.norm(params, p=1)

            elif ".score" in name:
                regularization_loss += torch.norm(params, p=1)
        regularization_loss = lmbda * regularization_loss

    elif regularizer == "L1_L2":
        # reg_loss =  ||p||_1 + ||p||_2^2
        for name, params in model.named_parameters():
            if ".bias_score" in name:
                if params.shape != torch.Tensor(1).shape:
                    regularization_loss += torch.norm(params, p=1)
                    regularization_loss += torch.norm(params, p=2) ** 2

            elif ".score" in name:
                regularization_loss += torch.norm(params, p=1)
                regularization_loss += torch.norm(params, p=2) ** 2
        regularization_loss = lmbda * regularization_loss

    elif regularizer == "var_red_1":
        # reg_loss = lambda * p^{alpha} (1-p)^{alpha'}
        for name, params in model.named_parameters():
            if ".bias_score" in name:
                if params.shape != torch.Tensor(1).shape:
                    regularization_loss += torch.sum(
                        torch.pow(params, alpha) * torch.pow(1 - params, alpha_prime)
                    )

            elif ".score" in name:
                # import pdb; pdb.set_trace()
                regularization_loss += torch.sum(
                    torch.pow(params, alpha) * torch.pow(1 - params, alpha_prime)
                )

        regularization_loss = lmbda * regularization_loss

    elif regularizer == "var_red_2":
        # reg_loss =  \sum_{i} w_i^2 * p_i(1-p_i)
        # NOTE: alpha = alpha' = 1 here. Change if needed.
        for conv_layer in conv_layers:
            regularization_loss += get_special_reg_sum(conv_layer)

        for lin_layer in linear_layers:
            regularization_loss += get_special_reg_sum(lin_layer)
        regularization_loss = lmbda * regularization_loss

    elif regularizer == "bin_entropy":
        # reg_loss = -p \log(p) - (1-p) \log(1-p)
        # NOTE: This will be nan because log(0) = inf. therefore, ignoring the end points
        for name, params in model.named_parameters():
            if ".bias_score" in name:
                if params.shape != torch.Tensor(1).shape:
                    params_filt = params[(params > 0) & (params < 1)]
                    regularization_loss += torch.sum(
                        -1.0 * params_filt * torch.log(params_filt)
                        - (1 - params_filt) * torch.log(1 - params_filt)
                    )

            elif ".score" in name:
                params_filt = params[(params > 0) & (params < 1)]
                regularization_loss += torch.sum(
                    -1.0 * params_filt * torch.log(params_filt)
                    - (1 - params_filt) * torch.log(1 - params_filt)
                )

        regularization_loss = lmbda * regularization_loss

    # print('red loss: ', regularization_loss)

    return regularization_loss
