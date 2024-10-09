from __future__ import print_function
from tqdm import tqdm
import os

import random
import logging.handlers
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable

from train import mmdtrain


from dataset import load_dataset_and_dataloader
from models.generators import ResnetG, SNGenerator, SNDiscriminator
from models.load_classifier import getClassifier
from config.load_parser import load_parser

import datetime

from metrics.evaluate import EvalModel
from metrics import fid_score

from utils.util import save_embedding, load_embedding
from utils.net_utils import (
    freeze_model_weights,
    freeze_model_subnet,
    unfreeze_model_weights,
    unfreeze_model_subnet,
    get_num_param,
    prune,
    get_layers,
    get_sub_param,
    get_total_param,
)
from utils.logging import LoggerSetting


# custom weights initialization called on netG
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname.find("ConvEncoder") == -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def createG(opt, log, device):
    log.info("# Generator:")
    if opt.netGType == "sngan":
        netG = SNGenerator()

    elif opt.netGType == "resnet":
        netG = ResnetG(
            opt.nz,
            opt.nc,
            opt.ngf,
            opt.imageSize,
            adaptFilterSize=not opt.notAdaptFilterSize,
            useConvAtSkipConn=opt.useConvAtGSkipConn,
        )
    log.info(netG)
    netG.to(device)
    if opt.distributed:
        netG = nn.DataParallel(netG, device_ids=opt.gpu_device_ids)
    return netG


def createD(opt, log, device):
    log.info("# Discriminator:")
    if opt.netGType == "sngan":
        netD = SNDiscriminator()
    log.info(netD)
    netD.to(device)
    if opt.distributed:
        netD = nn.DataParallel(netD, device_ids=opt.gpu_device_ids)
    return netD


def createMovingNet(opt, log, curNetEnc, device):
    numFeaturesInEnc = 0
    numFeaturesForEachEncLayer = curNetEnc.numberOfFeaturesPerLayer
    numLayersToFtrMatching = min(
        opt.numLayersToFtrMatching, len(numFeaturesForEachEncLayer)
    )
    numFeaturesInEnc += sum(numFeaturesForEachEncLayer[-numLayersToFtrMatching:])
    netMean = nn.Linear(numFeaturesInEnc, 1, bias=False)
    netVar = nn.Linear(numFeaturesInEnc, 1, bias=False)
    netMean.to(device)
    netVar.to(device)
    log.info(
        "# numFeaturesForEachEncLayer (from top to bottom): {}".format(
            numFeaturesForEachEncLayer
        )
    )
    log.info("@ opt.ftrMatchingWithTopLayers: {}".format(opt.ftrMatchingWithTopLayers))
    log.info("@ actual numLayersToFtrMatching: {}".format(numLayersToFtrMatching))
    log.info("# of features to be used: {}".format(numFeaturesInEnc))
    return netMean, netVar


def make_dir(opt):
    # Making Directories for saving grid-images and Generator models.
    try:
        os.makedirs(os.path.join(opt.saveroot, opt.outf))
    except OSError:
        pass

    try:
        log.info(
            "Make : {}/{} directory\n".format(opt.outf, "images")
            + "Make : {}/{} directory\n".format(opt.outf, "images_va")
            + "Make : {}/{} directory\n".format(opt.outf, "models")
        )

        # images : generate images from same noise per iteration
        os.makedirs(os.path.join(opt.saveroot, opt.outf, "images"))
        # images_va : generate images from different noise per iteration (for sampling)
        os.makedirs(os.path.join(opt.saveroot, opt.outf, "images_va"))
        # models : generator
        os.makedirs(os.path.join(opt.saveroot, opt.outf, "models"))
    except OSError:
        pass


def make_directories():
    try:
        os.makedirs(os.path.join("embeddings", "npz"), exist_ok=True)
        os.makedirs(os.path.join("features"), exist_ok=True)
    except OSError:
        pass


def make_date(outf):
    m_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print(m_time)
    return m_time + "_" + opt.outf


if __name__ == "__main__":
    # parser loading
    opt = load_parser()
    # make directories
    make_directories()
    # save results folder
    opt.outf = make_date(opt.outf)

    # set embeddings for evaluation
    path_dict = os.path.join(
        "embeddings",
        "npz",
        opt.dataset + str(opt.imageSize) + "_" + str(opt.test_num) + "_test.npz",
    )

    # CUDA setting
    n_gpu = len(opt.gpu_device_ids)
    opt.distributed = n_gpu > 1
    device = torch.device(f"cuda:{str(opt.gpunum)}" if opt.cuda else "cpu")
    print("CUDA available : ", torch.cuda.is_available())
    print("Device count : ", torch.cuda.device_count())
    print("Device name : ", torch.cuda.get_device_name(torch.cuda.current_device()))

    log = LoggerSetting(opt)
    log.info("Opt: {}".format(opt))

    # Manual seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    log.info("Random Seed: {}".format(opt.manualSeed))
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    # Dataloader
    dataloader = load_dataset_and_dataloader(opt)

    # Classifier creatinig
    if opt.training_loss == "MMD":
        curNetEnc = getClassifier(opt, log, device)
    else:
        OSError

    if opt.training == True:
        make_dir(opt)
        # Network creating
        netG = createG(opt, log, device)
        netG.apply(weights_init)
        num_param = get_num_param(netG)
        print("number of generator's parameters: ", num_param)

        netMean, netVar = createMovingNet(opt, log, curNetEnc, device)

        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizerMean = optim.Adam(
            netMean.parameters(), lr=opt.lrMovAvrg, betas=(opt.beta1, 0.999)
        )
        optimizerVar = optim.Adam(
            netVar.parameters(), lr=opt.lrMovAvrg, betas=(opt.beta1, 0.999)
        )

        if opt.ckpt != None:
            loadmodel = torch.load(opt.ckpt, map_location=device)
            netG.load_state_dict(loadmodel["netG_state_dict"])
            netMean.load_state_dict(loadmodel["netMean_state_dict"])
            netVar.load_state_dict(loadmodel["netVar_state_dict"])
            curNetEnc.load_state_dict(loadmodel["classifier"])
            optimizerG.load_state_dict(loadmodel["optimizerG_state_dict"])
            optimizerMean.load_state_dict(loadmodel["optimizerMean_state_dict"])
            optimizerVar.load_state_dict(loadmodel["optimizerVar_state_dict"])
            reset_flag(netG)
        if opt.training_loss == "MMD":
            freeze_model_weights(curNetEnc)
        if opt.algorithm in ["tr", "imp"]:
            unfreeze_model_weights(netG)
            freeze_model_subnet(netG)
        else:
            freeze_model_weights(netG)
            unfreeze_model_subnet(netG)

        mmdtrain(
            opt,
            log,
            netG,
            curNetEnc,
            netMean,
            netVar,
            optimizerG,
            optimizerMean,
            optimizerVar,
            dataloader,
            path_dict,
            device,
        )
