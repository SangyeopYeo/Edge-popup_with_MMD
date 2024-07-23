from collections import OrderedDict
from math import log
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils import spectral_norm
from config.load_parser import load_parser

logger = logging.getLogger(__name__)
opt = load_parser()

channel_multiplier = opt.channel_multiplier

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, bias_scores, k, scores_prune_threshold=-np.inf, bias_scores_prune_threshold=-np.inf):
        # Get the subnetwork by sorting the scores and using the top k%
        if opt.algorithm == "ep":
            out = scores.clone()
            _, idx = scores.flatten().sort()
            j = int((1 - k) * scores.numel())

            # flat_out and out access the same memory.
            flat_out = out.flatten()
            flat_out[idx[:j]] = False
            flat_out[idx[j:]] = True

            bias_out = bias_scores.clone()
            _, idx = bias_scores.flatten().sort()
            j = int((1 - k) * bias_scores.numel())

            bias_flat_out = bias_out.flatten()
            bias_flat_out[idx[:j]] = 0
            bias_flat_out[idx[j:]] = 1

        elif opt.algorithm in ["global_ep", "global_ep_iter"]:
            out = torch.gt(scores, torch.ones_like(scores)*scores_prune_threshold).float()
            bias_out = torch.gt(bias_scores, torch.ones_like(bias_scores)*bias_scores_prune_threshold).float()
        
        elif opt.algorithm == "gm":
            out = torch.gt(scores, torch.ones_like(scores)*scores_prune_threshold).float()
            bias_out = torch.gt(bias_scores, torch.ones_like(bias_scores)*bias_scores_prune_threshold).float()            


        return out, bias_out

    @staticmethod
    def backward(ctx, g_1, g_2):
        # send the gradient g straight-through on the backward pass.
        return g_1, g_2, None, None, None



##############################
##############################
##############################
class FixedSubnetLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.flag = nn.Parameter(torch.ones(self.weight.size()))
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))

        if opt.bias:
            if self.bias != None:
                self.bias_flag = nn.Parameter(torch.ones(self.bias.size()))
                self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
            else:
                self.bias_flag = nn.Parameter(torch.tensor(0.9))
                self.bias_scores = nn.Parameter(torch.tensor(0.9))    
        else:
            self.bias_flag = nn.Parameter(torch.tensor(0.9))
            self.bias_scores = nn.Parameter(torch.tensor(0.9))

        self.scores_prune_threshold = -np.inf
        self.bias_scores_prune_threshold = -np.inf

        if opt.algorithm == "gm":
            nn.init.uniform_(self.scores, a=0.0, b=1.0)
            nn.init.uniform_(self.bias_scores, a=0.0, b=1.0)
        #elif opt.algorithm in ["ep", "global_ep"]:
        else:
            if opt.score_init == "dense":
                nn.init.uniform_(self.scores, a=51, b=51.1)
            else:
                nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0)

        self.flag.requires_grad_(False)
        self.bias_flag.requires_grad_(False)
        if opt.algorithm not in ["tr", "imp"]:
            self.weight.requires_grad_(False)
            self.scores.requires_grad_(False)
            if opt.bias:
                if self.bias != None:
                    self.bias.requires_grad_(False)
                    self.bias_scores.requires_grad_(False)
                else:
                    pass
        else:
            self.scores.requires_grad_(False)
            if opt.bias:
                self.bias_scores.requires_grad_(False)

            #weight initialization
            #signed kaiming constant
        if opt.algorithm not in ["tr", "imp"]:
            if opt.init == "Constant":
                fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
                gain = nn.init.calculate_gain("relu")
                std = gain/math.sqrt(fan)
                self.weight.data = self.weight.data.sign()*std
            elif opt.init == "G_Normal":
            ##gaussian normal
                nn.init.normal_(self.weight, std = 10.0)
            ##kaiming normal
            elif opt.init =="K_Normal":
                if opt.scale_fan == True:
                    fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
                    fan = fan * opt.sparsity
                    gain = nn.init.calculate_gain("relu")
                    std = gain / math.sqrt(fan)
                    with torch.no_grad():
                        self.weight.data.normal_(0, std)
                else:
                    nn.init.kaiming_normal_(self.weight, mode = "fan_in", nonlinearity = "relu")

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        if opt.algorithm == "gm":         
            subnet, bias_subnet = 1, 1
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()

        if opt.algorithm == "ep":
            subnet, bias_subnet = 1, 1
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()

        if opt.algorithm in ["global_ep", "global_ep_iter"]:
            subnet, bias_subnet = 1, 1
        
        if opt.algorithm in ["tr", "imp"]:
            subnet, bias_subnet = 1, 1
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()
            
        w = self.weight * subnet
        if opt.bias:
            if self.bias != None:
                b = self.bias * bias_subnet
            else:
                b = self.bias
        else:
            b = self.bias

        w = self.weight * subnet

        return F.linear(x, w, b)
        return x


class FixedSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.flag = nn.Parameter(torch.ones(self.weight.size()))
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))

        if opt.bias:
            if self.bias != None:
                self.bias_flag = nn.Parameter(torch.ones(self.bias.size()))
                self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
            else:
                self.bias_flag = nn.Parameter(torch.tensor(0.9))
                self.bias_scores = nn.Parameter(torch.tensor(0.9))
        else:
            self.bias_flag = nn.Parameter(torch.tensor(0.9))
            self.bias_scores = nn.Parameter(torch.tensor(0.9))

        self.scores_prune_threshold = -np.inf
        self.bias_scores_prune_threshold = -np.inf

        if opt.algorithm == "gm":
            nn.init.uniform_(self.scores, a=0.0, b=1.0)
            nn.init.uniform_(self.bias_scores, a=0.0, b=1.0)
        #elif opt.algorithm in ["ep", "global_ep"]:
        else:
            if opt.score_init == "dense":
                nn.init.uniform_(self.scores, a=51.1, b=52.1)
            else:
                nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0)
        
        self.flag.requires_grad_(False)
        self.bias_flag.requires_grad_(False)
        if opt.algorithm not in ["tr", "imp"]:
            self.weight.requires_grad_(False)
            self.scores.requires_grad_(False)
            if opt.bias:
                if self.bias != None:
                    self.bias.requires_grad_(False)
                    self.bias_scores.requires_grad_(False)
                else:
                    pass
        else:
            self.scores.requires_grad_(False)
            if opt.bias:
                self.bias_scores.requires_grad_(False)
            #weight initialization
            #signed kaiming constant
        if opt.algorithm not in ["tr", "imp"]:
            if opt.init == "Constant":
                fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
                gain = nn.init.calculate_gain("relu")
                std = gain/math.sqrt(fan)
                self.weight.data = self.weight.data.sign()*std
            elif opt.init == "G_Normal":
            ##gaussian normal
                nn.init.normal_(self.weight, std = 10.0)
            ##kaiming normal
            elif opt.init =="K_Normal":
                if opt.scale_fan == True:
                    fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
                    fan = fan * opt.sparsity
                    gain = nn.init.calculate_gain("relu")
                    std = gain / math.sqrt(fan)
                    with torch.no_grad():
                        self.weight.data.normal_(0, std)
                else:
                    nn.init.kaiming_normal_(self.weight, mode = "fan_in", nonlinearity = "relu")



    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()


    def forward(self, x):
        if opt.algorithm == "gm":
            subnet, bias_subnet = 1, 1
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()

        if opt.algorithm == "ep":
            subnet, bias_subnet = 1, 1
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()

        if opt.algorithm in ["global_ep", "global_ep_iter"]:
            subnet, bias_subnet = 1, 1
        
        if opt.algorithm in ["tr", "imp"]:
            subnet, bias_subnet = 1, 1
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()
        w = self.weight * subnet
        if opt.bias:
            if self.bias != None:
                b = self.bias * bias_subnet
            else:
                b = self.bias
        else:
            b = self.bias

        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class FixedSubnetConvT(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.flag = nn.Parameter(torch.ones(self.weight.size()))
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))

        if opt.bias:
            if self.bias != None:
                self.bias_flag = nn.Parameter(torch.ones(self.bias.size()))
                self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
            else:
                self.bias_flag = nn.Parameter(torch.tensor(0.9))
                self.bias_scores = nn.Parameter(torch.tensor(0.9))    
        else:
            self.bias_flag = nn.Parameter(torch.tensor(0.9))
            self.bias_scores = nn.Parameter(torch.tensor(0.9))

        self.scores_prune_threshold = -np.inf
        self.bias_scores_prune_threshold = -np.inf

        if opt.algorithm == "gm":
            nn.init.uniform_(self.scores, a=0.0, b=1.0)
            nn.init.uniform_(self.bias_scores, a=0.0, b=1.0)
        #elif opt.algorithm in ["ep", "global_ep"]:
        else:
            if opt.score_init == "dense":
                nn.init.uniform_(self.scores, a=51.1, b=52.1)
            else:
                nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0)

        self.flag.requires_grad_(False)
        self.bias_flag.requires_grad_(False)
        if opt.algorithm not in ["tr", "imp"]:
            self.weight.requires_grad_(False)
            self.scores.requires_grad_(False)
            if opt.bias:
                if self.bias != None:
                    self.bias.requires_grad_(False)
                    self.bias_scores.requires_grad_(False)
                else:
                    pass
        else:
            self.scores.requires_grad_(False)
            if opt.bias:
                self.bias_scores.requires_grad_(False)

            #weight initialization
            #signed kaiming constant
        if opt.algorithm not in ["tr", "imp"]:
            if opt.init == "Constant":
                fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
                gain = nn.init.calculate_gain("relu")
                std = gain/math.sqrt(fan)
                self.weight.data = self.weight.data.sign()*std
            elif opt.init == "G_Normal":
            ##gaussian normal
                nn.init.normal_(self.weight, std = 10.0)
            ##kaiming normal
            elif opt.init =="K_Normal":
                if opt.scale_fan == True:
                    fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
                    fan = fan * opt.sparsity
                    gain = nn.init.calculate_gain("relu")
                    std = gain / math.sqrt(fan)
                    with torch.no_grad():
                        self.weight.data.normal_(0, std)
                else:
                    nn.init.kaiming_normal_(self.weight, mode = "fan_in", nonlinearity = "relu")

                

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        if opt.algorithm == "gm":         
            subnet, bias_subnet = 1, 1
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()

        if opt.algorithm == "ep":
            subnet, bias_subnet = 1, 1
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()
        
        if opt.algorithm in ["global_ep", "global_ep_iter"]:
            subnet, bias_subnet = 1, 1

        if opt.algorithm in ["tr", "imp"]:
            subnet, bias_subnet = 1, 1
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()
            
        w = self.weight * subnet
        if opt.bias:
            if self.bias != None:
                b = self.bias * bias_subnet
            else:
                b = self.bias
        else:
            b = self.bias

        x = F.conv_transpose2d(
            x, w, b, self.stride, self.padding, self.output_padding, self.groups, self.dilation
        )
        return x

##############################
##############################
##############################




class SubnetLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.flag = nn.Parameter(torch.ones(self.weight.size()))
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))

        if opt.bias:
            if self.bias != None:
                self.bias_flag = nn.Parameter(torch.ones(self.bias.size()))
                self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
            else:
                self.bias_flag = nn.Parameter(torch.tensor(0.9))
                self.bias_scores = nn.Parameter(torch.tensor(0.9))    
        else:
            self.bias_flag = nn.Parameter(torch.tensor(0.9))
            self.bias_scores = nn.Parameter(torch.tensor(0.9))

        self.scores_prune_threshold = -np.inf
        self.bias_scores_prune_threshold = -np.inf

        if opt.algorithm == "gm":
            nn.init.uniform_(self.scores, a=0.0, b=1.0)
            nn.init.uniform_(self.bias_scores, a=0.0, b=1.0)
        #elif opt.algorithm in ["ep", "global_ep"]:
        else:
            if opt.score_init == "dense":
                nn.init.uniform_(self.scores, a=-0.1, b=0.1)
            else:
                nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0)

        self.flag.requires_grad_(False)
        self.bias_flag.requires_grad_(False)
        if opt.algorithm not in ["tr", "imp"]:
            self.weight.requires_grad_(False)
            if opt.bias:
                if self.bias != None:
                    self.bias.requires_grad_(False)
                else:
                    pass
        else:
            self.scores.requires_grad_(False)
            if opt.bias:
                self.bias_scores.requires_grad_(False)

            #weight initialization
            #signed kaiming constant
        if opt.algorithm not in ["tr", "imp"]:
            if opt.init == "Constant":
                fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
                gain = nn.init.calculate_gain("relu")
                std = gain/math.sqrt(fan)
                self.weight.data = self.weight.data.sign()*std
            elif opt.init == "G_Normal":
            ##gaussian normal
                nn.init.normal_(self.weight, std = 10.0)
            ##kaiming normal
            elif opt.init =="K_Normal":
                if opt.scale_fan == True:
                    fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
                    fan = fan * opt.sparsity
                    gain = nn.init.calculate_gain("relu")
                    std = gain / math.sqrt(fan)
                    with torch.no_grad():
                        self.weight.data.normal_(0, std)
                else:
                    nn.init.kaiming_normal_(self.weight, mode = "fan_in", nonlinearity = "relu")
        else:
            if opt.training_loss == "ADV":
                nn.init.xavier_uniform(self.weight.data, 1.)
            else:
                pass

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        if opt.algorithm == "gm":         
            subnet, bias_subnet = GetSubnet.apply(self.scores, self.bias_scores, opt.sparsity, opt.threshold, opt.threshold)
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()

        if opt.algorithm == "ep":
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), opt.sparsity)
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()

        if opt.algorithm in ["global_ep", "global_ep_iter"]:
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), 0, self.scores_prune_threshold, self.bias_scores_prune_threshold)
        
        if opt.algorithm in ["tr", "imp"]:
            subnet, bias_subnet = 1, 1
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()
            
        w = self.weight * subnet
        if opt.bias:
            if self.bias != None:
                b = self.bias * bias_subnet
            else:
                b = self.bias
        else:
            b = self.bias

        w = self.weight * subnet

        return F.linear(x, w, b)
        return x


class SubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.flag = nn.Parameter(torch.ones(self.weight.size()))
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))

        if opt.bias:
            if self.bias != None:
                self.bias_flag = nn.Parameter(torch.ones(self.bias.size()))
                self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
            else:
                self.bias_flag = nn.Parameter(torch.tensor(0.9))
                self.bias_scores = nn.Parameter(torch.tensor(0.9))
        else:
            self.bias_flag = nn.Parameter(torch.tensor(0.9))
            self.bias_scores = nn.Parameter(torch.tensor(0.9))

        self.scores_prune_threshold = -np.inf
        self.bias_scores_prune_threshold = -np.inf

        if opt.algorithm == "gm":
            nn.init.uniform_(self.scores, a=0.0, b=1.0)
            nn.init.uniform_(self.bias_scores, a=0.0, b=1.0)
        #elif opt.algorithm in ["ep", "global_ep"]:
        else:
            if opt.score_init == "dense":
                nn.init.uniform_(self.scores, a=-0.1, b=0.1)
            else:
                nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0)
        
        self.flag.requires_grad_(False)
        self.bias_flag.requires_grad_(False)
        if opt.algorithm not in ["tr", "imp"]:
            self.weight.requires_grad_(False)
            if opt.bias:
                if self.bias != None:
                    self.bias.requires_grad_(False)
                else:
                    pass
        else:
            self.scores.requires_grad_(False)
            if opt.bias:
                self.bias_scores.requires_grad_(False)
            #weight initialization
            #signed kaiming constant
        if opt.algorithm not in ["tr", "imp"]:
            if opt.init == "Constant":
                fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
                gain = nn.init.calculate_gain("relu")
                std = gain/math.sqrt(fan)
                self.weight.data = self.weight.data.sign()*std
            elif opt.init == "G_Normal":
            ##gaussian normal
                nn.init.normal_(self.weight, std = 10.0)
            ##kaiming normal
            elif opt.init =="K_Normal":
                if opt.scale_fan == True:
                    fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
                    fan = fan * opt.sparsity
                    gain = nn.init.calculate_gain("relu")
                    std = gain / math.sqrt(fan)
                    with torch.no_grad():
                        self.weight.data.normal_(0, std)
                else:
                    nn.init.kaiming_normal_(self.weight, mode = "fan_in", nonlinearity = "relu")
        else:
            if opt.training_loss == "ADV":
                nn.init.xavier_uniform(self.weight.data, 1.)
            else:
                pass


    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()


    def forward(self, x):
        if opt.algorithm == "gm":
            subnet, bias_subnet = GetSubnet.apply(self.scores, self.bias_scores, opt.sparsity, opt.threshold, opt.threshold)
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()

        if opt.algorithm == "ep":
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), opt.sparsity)
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()

        if opt.algorithm in ["global_ep", "global_ep_iter"]:
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), 0, self.scores_prune_threshold, self.bias_scores_prune_threshold)
        
        if opt.algorithm in ["tr", "imp"]:
            subnet, bias_subnet = 1, 1
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()
        w = self.weight * subnet
        if opt.bias:
            if self.bias != None:
                b = self.bias * bias_subnet
            else:
                b = self.bias
        else:
            b = self.bias

        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SubnetConvT(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.flag = nn.Parameter(torch.ones(self.weight.size()))
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))

        if opt.bias:
            if self.bias != None:
                self.bias_flag = nn.Parameter(torch.ones(self.bias.size()))
                self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
            else:
                self.bias_flag = nn.Parameter(torch.tensor(0.9))
                self.bias_scores = nn.Parameter(torch.tensor(0.9))    
        else:
            self.bias_flag = nn.Parameter(torch.tensor(0.9))
            self.bias_scores = nn.Parameter(torch.tensor(0.9))

        self.scores_prune_threshold = -np.inf
        self.bias_scores_prune_threshold = -np.inf

        if opt.algorithm == "gm":
            nn.init.uniform_(self.scores, a=0.0, b=1.0)
            nn.init.uniform_(self.bias_scores, a=0.0, b=1.0)
        #elif opt.algorithm in ["ep", "global_ep"]:
        else:
            if opt.score_init == "dense":
                nn.init.uniform_(self.scores, a=-0.1, b=0.1)
            else:
                nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0)

        self.flag.requires_grad_(False)
        self.bias_flag.requires_grad_(False)
        if opt.algorithm not in ["tr", "imp"]:
            self.weight.requires_grad_(False)
            if opt.bias:
                if self.bias != None:
                    self.bias.requires_grad_(False)
                else:
                    pass
        else:
            self.scores.requires_grad_(False)
            if opt.bias:
                self.bias_scores.requires_grad_(False)

            #weight initialization
            #signed kaiming constant
        if opt.algorithm not in ["tr", "imp"]:
            if opt.init == "Constant":
                fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
                gain = nn.init.calculate_gain("relu")
                std = gain/math.sqrt(fan)
                self.weight.data = self.weight.data.sign()*std
            elif opt.init == "G_Normal":
            ##gaussian normal
                nn.init.normal_(self.weight, std = 10.0)
            ##kaiming normal
            elif opt.init =="K_Normal":
                if opt.scale_fan == True:
                    fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
                    fan = fan * opt.sparsity
                    gain = nn.init.calculate_gain("relu")
                    std = gain / math.sqrt(fan)
                    with torch.no_grad():
                        self.weight.data.normal_(0, std)
                else:
                    nn.init.kaiming_normal_(self.weight, mode = "fan_in", nonlinearity = "relu")
        else:
            if opt.training_loss == "ADV":
                nn.init.xavier_uniform(self.weight.data, 1.)
            else:
                pass
                

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        if opt.algorithm == "gm":         
            subnet, bias_subnet = GetSubnet.apply(self.scores, self.bias_scores, opt.sparsity, opt.threshold, opt.threshold)
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()

        if opt.algorithm == "ep":
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), opt.sparsity)
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()
        
        if opt.algorithm in ["global_ep", "global_ep_iter"]:
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), 0, self.scores_prune_threshold, self.bias_scores_prune_threshold)

        if opt.algorithm in ["tr", "imp"]:
            subnet, bias_subnet = 1, 1
            subnet = subnet * self.flag.data.float()
            bias_subnet = bias_subnet * self.bias_flag.data.float()
            
        w = self.weight * subnet
        if opt.bias:
            if self.bias != None:
                b = self.bias * bias_subnet
            else:
                b = self.bias
        else:
            b = self.bias

        x = F.conv_transpose2d(
            x, w, b, self.stride, self.padding, self.output_padding, self.groups, self.dilation
        )
        return x


class ResnetG(nn.Module):
    def __init__(self, nz, nc, ndf, imageSize = 32, adaptFilterSize = False, useConvAtSkipConn = False):
        super(ResnetG, self).__init__()
        self.nz = nz
        self.ndf = ndf

        if adaptFilterSize == True and useConvAtSkipConn == False:
            useConvAtSkipConn = True
            logger.warn("WARNING: In ResnetG, setting useConvAtSkipConn to True because adaptFilterSize is True.")

        numUpsampleBlocks = int(log(imageSize, 2)) - 2 
        
        numLayers = numUpsampleBlocks + 1
        filterSizePerLayer = [ndf] * numLayers
        if adaptFilterSize:
            for i in range(numLayers - 1, -1, -1):
                if i == numLayers - 1:
                    filterSizePerLayer[i] = ndf
                else:
                    filterSizePerLayer[i] = filterSizePerLayer[i+1]*2
            
        firstL = SubnetConvT(nz, int(filterSizePerLayer[0] * channel_multiplier), 4, 1, 0, bias=False)
#        nn.init.xavier_uniform(firstL.weight.data, 1.)
        lastL  = SubnetConv(int(filterSizePerLayer[-1] * channel_multiplier), nc, 3, stride=1, padding=1, bias=opt.bias)
#        nn.init.xavier_uniform(lastL.weight.data, 1.)

        nnLayers = OrderedDict()
        # first deconv goes from the z size
        nnLayers["firstConv"]   = firstL
        
        layerNumber = 1
        for i in range(numUpsampleBlocks):
            nnLayers["resblock_%d"%i] = ResidualBlockG(filterSizePerLayer[layerNumber-1], filterSizePerLayer[layerNumber], stride=2, useConvAtSkipConn = useConvAtSkipConn)
            layerNumber += 1
        nnLayers["batchNorm"] = nn.BatchNorm2d(int(filterSizePerLayer[-1] * channel_multiplier), affine = opt.affine)
        nnLayers["relu"]      = nn.ReLU()
        nnLayers["lastConv"]  = lastL
        nnLayers["tanh"]      = nn.Tanh()

        self.net = nn.Sequential(nnLayers)

    def forward(self, input):
        return self.net(input)

class Upsample(nn.Module):
    def __init__(self, scale_factor=2, size=None):
        super(Upsample, self).__init__()
        self.upsample = F.upsample_nearest
        self.size = size
        self.scale_factor = scale_factor
        
    def forward(self, x):
        x = self.upsample(x, size=self.size, scale_factor = self.scale_factor)
        return x

class ResidualBlockG(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, useConvAtSkipConn = False):
        super(ResidualBlockG, self).__init__()

        self.conv1 = SubnetConv(int(in_channels * channel_multiplier), int(out_channels * channel_multiplier), 3, 1, padding=1, bias=opt.bias)
        self.conv2 = SubnetConv(int(out_channels * channel_multiplier), int(out_channels * channel_multiplier), 3, 1, padding=1, bias=opt.bias)
        
        if useConvAtSkipConn:
            self.conv_bypass = SubnetConv(int(in_channels * channel_multiplier), int(out_channels * channel_multiplier), 1, 1, padding=0, bias=opt.bias)
#            nn.init.xavier_uniform(self.conv_bypass.weight.data, 1.)
        
#        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
#        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(int(in_channels * channel_multiplier), affine = opt.affine),
            nn.ReLU(),
            Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(int(out_channels * channel_multiplier), affine = opt.affine),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            if useConvAtSkipConn:
                self.bypass = nn.Sequential(self.conv_bypass, Upsample(scale_factor=2))
            else:
                self.bypass = Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class SNGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU(), upsample=False, n_classes=0):
        super(SNGenBlock, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        self.c1 = SubnetConv(in_channels, hidden_channels, kernel_size=ksize, padding=pad, bias = opt.bias)
        self.c2 = SubnetConv(hidden_channels, out_channels, kernel_size=ksize, padding=pad, bias = opt.bias)

        self.b1 = nn.BatchNorm2d(in_channels, affine = opt.affine)
        self.b2 = nn.BatchNorm2d(hidden_channels, affine = opt.affine)
        if self.learnable_sc:
            self.c_sc = SubnetConv(in_channels, out_channels, kernel_size=1, padding=0, bias = opt.bias)

    def upsample_conv(self, x, conv):
        return conv(nn.UpsamplingNearest2d(scale_factor=2)(x))

    def residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self.upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class SNGenerator(nn.Module):
    def __init__(self, activation=nn.ReLU(), n_classes=0):
        super(SNGenerator, self).__init__()
        self.bottom_width = 4 #args.bottom_width
        self.activation = activation
        self.n_classes = n_classes
        self.ch = 256 #args.gf_dim
        if opt.First_freeze == True:
            self.l1 = FixedSubnetLinear(128, (self.bottom_width ** 2) * self.ch, bias = opt.bias)
        else:
            self.l1 = SubnetLinear(128, (self.bottom_width ** 2) * self.ch, bias = opt.bias) #(args.latent_dim, (self.bottom_width ** 2) * self.ch, bias = opt.bias)
        self.block2 = SNGenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.block3 = SNGenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.block4 = SNGenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.block5 = SNGenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.b5 = nn.BatchNorm2d(self.ch, affine = opt.affine)
        if opt.Last_freeze == True:
            self.c5 = FixedSubnetConv(self.ch, 3, kernel_size=3, stride=1, padding=1, bias = opt.bias)
        else:
            self.c5 = SubnetConv(self.ch, 3, kernel_size=3, stride=1, padding=1, bias = opt.bias)

    def forward(self, z):
        h = z
        h = self.l1(h).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.b5(h)
        h = self.activation(h)
        h = nn.Tanh()(self.c5(h))
        return h


"""Discriminator"""


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class SNOptimizedDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=nn.ReLU()):
        super(SNOptimizedDisBlock, self).__init__()
        self.activation = activation

        self.c1 = SubnetConv(in_channels, out_channels, kernel_size=ksize, padding=pad, bias = opt.bias)
        self.c2 = SubnetConv(out_channels, out_channels, kernel_size=ksize, padding=pad, bias = opt.bias)
        self.c_sc = SubnetConv(in_channels, out_channels, kernel_size=1, padding=0, bias = opt.bias)
        if True: #args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class SNDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU(), downsample=False):
        super(SNDisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = SubnetConv(in_channels, hidden_channels, kernel_size=ksize, padding=pad, bias = opt.bias)
        self.c2 = SubnetConv(hidden_channels, out_channels, kernel_size=ksize, padding=pad, bias = opt.bias)
        if True: #args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = SubnetConv(in_channels, out_channels, kernel_size=1, padding=0, bias = opt.bias)
            if True: #args.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class SNDiscriminator(nn.Module):
    def __init__(self, activation=nn.ReLU()):
        super(SNDiscriminator, self).__init__()
        self.ch = 128 #args.df_dim
        self.activation = activation
        self.block1 = SNOptimizedDisBlock(3, self.ch)
        self.block2 = SNDisBlock(self.ch, self.ch, activation=activation, downsample=True)
        self.block3 = SNDisBlock(self.ch, self.ch, activation=activation, downsample=False)
        self.block4 = SNDisBlock(self.ch, self.ch, activation=activation, downsample=False)
        #self.block5 = SNDisBlock(self.ch, self.ch, activation=activation, downsample=False)
        self.l5 = SubnetLinear(self.ch, 1, bias=False)
        if True: #args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        #h = self.block5(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output