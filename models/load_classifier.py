import torch
import torch.nn as nn


def getClassifier(opt, log=None, device="cpu"):
    assert opt is not None, "please input the running classifier configuration"
    assert len(opt.netEncType) == 1, "."
    for idx, netEncType in enumerate(opt.netEncType):

        net = setClassifier(netEncType, opt, log)

        for param in net.parameters():
            param.requires_grad = False
        log.info("check training parameters")
        for name, param in net.named_parameters():
            if param.requires_grad == True:
                log.info("\t", name)
        log.info(net)
        net.to(device)
    return net


def setClassifier(netEncType, opt, log):
    net = None
    netEncType == "vgg19-pytorch"
    from models.vgg_pytorch import VGG19 as VGG19Pytorch

    ###################################################
    #   VGG 19 of Pytorch official Loading
    ###################################################
    net = VGG19Pytorch(
        get_perceptual_feats=True,
        num_classes=opt.numClassesInFtrExt,
        image_size=opt.imageSize,
    )
    return net
