from __future__ import print_function
from tqdm import tqdm
import os

import numpy as np
import torch
import torch.nn as nn

import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import time

from dataset import load_dataset_and_dataloader

from metrics.evaluate import EvalModel
from metrics import fid_score
from metrics.prdc import iprdc

from utils.util import save_embedding, load_embedding
from utils.net_utils import (
    get_regularization_loss,
    prune,
    get_model_sparsity,
    round_model,
    reset_flag,
    layer_sparsity,
)

import copy


def tanh(x):
    p_exp_x = torch.exp(x)
    m_exp_x = torch.exp(-x)
    y = (p_exp_x - m_exp_x) / (p_exp_x + m_exp_x)
    return y


# custom weights initialization called on netG
def extractFeatures(opt, batchOfData, curNetEnc, detachOutput=False):
    """
    Applies feature extractor. Concatenate feature vectors from all selected layers.
    """

    # gets features from each layer of netEnc
    ftrs = []
    ftrsPerLayer = curNetEnc(batchOfData)[1]
    numFeaturesForEachEncLayer = curNetEnc.numberOfFeaturesPerLayer
    numLayersToFtrMatching = min(
        opt.numLayersToFtrMatching, len(numFeaturesForEachEncLayer)
    )
    for lId in range(1, numLayersToFtrMatching + 1):
        cLid = lId - 1  # gets features in forward order

        ftrsOfLayer = ftrsPerLayer[cLid].view(ftrsPerLayer[cLid].size()[0], -1)

        if detachOutput:
            ftrs.append(ftrsOfLayer.detach())
        else:
            ftrs.append(ftrsOfLayer)
    ftrs = torch.cat(ftrs, dim=1)
    return ftrs


def compute_real_features(
    opt, log, dataloader, curNetEnc, device, numExamplesProcessed
):
    input_t = torch.FloatTensor(opt.batchSize, opt.nc, opt.imageSize, opt.imageSize).to(
        device
    )
    if numExamplesProcessed is None:
        numExamplesProcessed = 0.0
    globalFtrMeanValues = []
    log.info("Computing mean features from TRUE data")
    for i, data in enumerate(tqdm(dataloader), 1):
        real_cpu = data[0]  # img, target

        if opt.cuda:
            real_cpu = real_cpu.to(device)

        if real_cpu.shape[1] == 1:
            real_cpu = real_cpu.expand(
                real_cpu.shape[0], 3, real_cpu.shape[-1], real_cpu.shape[-1]
            )

        input_t.resize_as_(real_cpu).copy_(real_cpu)
        realData = Variable(input_t)
        numExamplesProcessed += realData.size()[0]

        # extracts features for TRUE data
        allFtrsTrue = extractFeatures(opt, realData, curNetEnc, detachOutput=True)

        if len(globalFtrMeanValues) < 1:
            globalFtrMeanValues = torch.sum(allFtrsTrue, dim=0).detach()
            featureSqrdValues = torch.sum(allFtrsTrue**2, dim=0).detach()
        else:
            globalFtrMeanValues += torch.sum(allFtrsTrue, dim=0).detach()
            featureSqrdValues += torch.sum(allFtrsTrue**2, dim=0).detach()

    return numExamplesProcessed, globalFtrMeanValues, featureSqrdValues


def get_embedding(opt, log, path_dict, real=True, netG=None, device=None):
    if real:
        if os.path.isfile(path_dict):
            return load_embedding(path_dict)
        else:
            save_embedding(
                calculate_embedding(opt, log, real=True, device=device), path_dict
            )
            return 0
    else:
        return calculate_embedding(opt, log, real=False, netG=netG, device=device)


def calculate_embedding(opt, log, real=True, netG=None, device=None):

    if real:
        eval_dataloader = load_dataset_and_dataloader(opt, eval=True)
    else:
        assert netG is not None, "Generator is None"
    embed_list = ["inceptionV3"]
    embed_dict = {}

    log.info("Computing features from data for evaluating")

    for embedder in embed_list:
        embed_dict[embedder] = {}
        embed_model = EvalModel(
            embedder, batch_size=opt.batchSize, device=device, test_num=opt.test_num
        )
        if real:
            # real_pred
            embed_dict[embedder]["pred"] = embed_model.get_embeddings_from_loaders(
                eval_dataloader
            )
        else:
            # fake_pred
            embed_dict[embedder]["pred"] = embed_model.get_embeddings_from_generator(
                netG, opt, device
            )

        # mu, sigma
        embed_dict[embedder]["mu"], embed_dict[embedder]["sigma"] = (
            fid_score.getMean_and_Sigma(embed_dict[embedder]["pred"])
        )
    return embed_dict


def Preprocessing_ImgNet(device):
    imageNetNormMean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    imageNetNormStd = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    imageNetNormMin = -imageNetNormMean / imageNetNormStd
    imageNetNormMax = (1.0 - imageNetNormMean) / imageNetNormStd
    imageNetNormRange = imageNetNormMax - imageNetNormMin
    imageNetNormMinV = torch.FloatTensor(imageNetNormMin).to(device)
    imageNetNormRangeV = torch.FloatTensor(imageNetNormRange).to(device)
    imageNetNormMinV.resize_(1, 3, 1, 1)
    imageNetNormRangeV.resize_(1, 3, 1, 1)
    imageNetNormMinV = Variable(imageNetNormMinV)
    imageNetNormRangeV = Variable(imageNetNormRangeV)

    return imageNetNormMinV, imageNetNormRangeV


def saveModel(
    opt,
    netG,
    netMean,
    netVar,
    curNetEnc,
    optimizerG,
    optimizerMean,
    optimizerVar,
    globalFtrMeanValues,
    globalFtrVarValues,
    schedulerG=None,
    schedulerMean=None,
    schedulerVar=None,
    suffix="",
):
    # saving current best model
    torch.save(
        {
            "netG_state_dict": netG.state_dict(),
            "netMean_state_dict": netMean.state_dict(),
            "netVar_state_dict": netVar.state_dict(),
            "classifier": curNetEnc.state_dict(),
            "optimizerG_state_dict": optimizerG.state_dict(),
            "optimizerMean_state_dict": optimizerMean.state_dict(),
            "optimizerVar_state_dict": optimizerVar.state_dict(),
            "Mean": globalFtrMeanValues,
            "Var": globalFtrVarValues,
            "schedulerG_state_dict": (
                schedulerG.state_dict() if opt.scheduler != False else None
            ),
            "schedulerMean_state_dict": (
                schedulerMean.state_dict() if opt.scheduler != False else None
            ),
            "schedulerVar_state_dict": (
                schedulerVar.state_dict() if opt.scheduler != False else None
            ),
        },
        "%s/%s/%s/netG_%s.tar" % (opt.saveroot, opt.outf, "models", suffix),
    )


def evaluate(opt, log, path_dict, netG, iterId, device):
    real_pred = get_embedding(opt, log, path_dict, real=True, device=device)
    fake_pred = get_embedding(opt, log, path_dict, real=False, netG=netG, device=device)
    for embedder in real_pred.keys():
        # fid
        real_mu = real_pred[embedder][()]["mu"]
        real_sigma = real_pred[embedder][()]["sigma"]
        fake_mu = fake_pred[embedder]["mu"]
        fake_sigma = fake_pred[embedder]["sigma"]
        fid_val = fid_score.get_fid((real_mu, real_sigma), (fake_mu, fake_sigma))
        # prec
        r_pred = real_pred[embedder][()]["pred"]
        f_pred = fake_pred[embedder]["pred"]
        prdc_value = iprdc(r_pred, f_pred)
        # logging
        log.info(
            "[{%d}/{%d}] FID_Value: {%.6f} for Embedder %s"
            % (iterId + 1, opt.niter, fid_val, embedder)
        )
        log.info(
            "[{%d}/{%d}] Precision: %.6f / Recall: %.6f / Density: %.6f / Coverage: %.6f for Embedder %s"
            % (
                iterId + 1,
                opt.niter,
                prdc_value["precision"],
                prdc_value["recall"],
                prdc_value["density"],
                prdc_value["coverage"],
                embedder,
            )
        )
    return fid_val


def computeReal(opt, log, dataloader, curNetEnc, device):
    numExamplesProcessed = 0.0
    numExamplesProcessed, globalFtrMeanValues, featureSqrdValues = (
        compute_real_features(
            opt,
            log,
            dataloader=dataloader,
            curNetEnc=curNetEnc,
            device=device,
            numExamplesProcessed=numExamplesProcessed,
        )
    )
    # variance = (SumSq - (Sum x Sum) / n) / (n - 1)
    globalFtrVarValues = (
        featureSqrdValues - (globalFtrMeanValues**2) / numExamplesProcessed
    ) / (numExamplesProcessed - 1)
    log.info(
        "Normalizing sum of features with denominator: {}".format(numExamplesProcessed)
    )
    globalFtrMeanValues = globalFtrMeanValues / numExamplesProcessed
    return globalFtrMeanValues, globalFtrVarValues


def mmdtrain(
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
):

    # Logging
    log.info("Computed features from {} data for evaluating FID".format("real"))
    if not get_embedding(opt, log, path_dict, real=True):
        log.info("Save the Real Embedding in {}".format(path_dict))
    else:
        log.info("Exist the Real Embedding in {}".format(path_dict))

    netG.train()
    # Initial settings
    if opt.scheduler == "cos":
        schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizerG, T_max=10, eta_min=0
        )
        schedulerMean = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizerMean, T_max=10, eta_min=0
        )
        schedulerVar = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizerVar, T_max=10, eta_min=0
        )
    elif opt.scheduler == False:
        schedulerG = None
        schedulerMean = None
        schedulerVar = None
    if (opt.scheduler != False) and opt.ckpt != None:
        loadmodel = torch.load(opt.ckpt, map_location=device)
        schedulerG.load_state_dict(loadmodel["schedulerG_state_dict"])
        schedulerMean.load_state_dict(loadmodel["schedulerMean_state_dict"])
        schedulerVar.load_state_dict(loadmodel["schedulerVar_state_dict"])
        opt.firstBatchId = loadmodel["optimizerVar_state_dict"]["state"][0]["step"]

    if opt.netGType == "sngan":
        noise = torch.FloatTensor(opt.batchSize, opt.nz).to(device)
        fixed_noise = Variable(
            torch.FloatTensor(min(64, opt.batchSize), opt.nz).normal_(0, 1).to(device)
        )
    else:
        noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1).to(device)
        fixed_noise = Variable(
            torch.FloatTensor(min(64, opt.batchSize), opt.nz, 1, 1)
            .normal_(0, 1)
            .to(device)
        )
    avrgLossNetGMean = 0.0
    avrgLossNetGVar = 0.0
    avrgLossNetMean = 0.0
    avrgLossNetVar = 0.0
    criterionL1Loss = nn.L1Loss().to(device)
    criterionL2Loss = nn.MSELoss().to(device)
    FID_list = []

    # Preporcessing for ImageNet
    imageNetNormMinV, imageNetNormRangeV = Preprocessing_ImgNet(device)
    # Computing Real Dataset
    features_dict = (
        opt.netEncType[0] + "_" + opt.dataset + "_" + str(opt.imageSize) + ".tar"
    )
    if opt.ckpt != None:
        loadmodel = torch.load(opt.ckpt, map_location=device)
        globalFtrMeanValues, globalFtrVarValues = loadmodel["Mean"].to(
            device
        ), loadmodel["Var"].to(device)
    else:
        if os.path.isfile("./features/" + features_dict) == True:
            ftr_check = torch.load("./features/" + features_dict, map_location=device)
            globalFtrMeanValues, globalFtrVarValues = ftr_check["Mean"].to(
                device
            ), ftr_check["Var"].to(device)
        else:
            globalFtrMeanValues, globalFtrVarValues = computeReal(
                opt, log, dataloader, curNetEnc, device
            )  # 얘네 저장하기
            torch.save(
                {"Mean": globalFtrMeanValues, "Var": globalFtrVarValues},
                "./features/" + features_dict,
            )

    # Training start
    start_time = time.time()
    log.info("Start Checking Time . . .")
    for iterId in range(opt.firstBatchId, int(opt.niter)):
        curNetEnc.zero_grad()
        netG.zero_grad()
        netMean.zero_grad()
        netVar.zero_grad()
        if (
            (opt.algorithm == "gm")
            and (iterId % opt.project_freq == 0)
            and not opt.differentiate_clamp
        ):
            for name, params in netG.named_parameters():
                if "score" in name:
                    scores = params
                    with torch.no_grad():
                        scores.data = torch.clamp(scores.data, 0.0, 1.0)
        # creates noise
        if opt.netGType == "sngan":
            noise.resize_(opt.batchSize, int(opt.nz)).normal_(0, 1.0)
            noisev = Variable(noise)
        else:
            noise.resize_(opt.batchSize, int(opt.nz), 1, 1).normal_(0, 1.0)
            noisev = Variable(noise)
        fakeData = netG(noisev)

        # normalize part
        if fakeData.shape[1] == 1:  # gray img
            fakeData = fakeData.expand(
                fakeData.shape[0], 3, fakeData.shape[-1], fakeData.shape[-1]
            )
        elif fakeData.shape[1] == 3:  # color img
            fakeData = (((fakeData + 1) * imageNetNormRangeV) / 2) + imageNetNormMinV

        ftrsFake = [
            extractFeatures(opt, fakeData, curNetEnc, detachOutput=False)
        ]  # featureextract

        # updates Adam moving average of mean differences

        ftrsMeanFakeData = [
            torch.mean(ftrsFakeData, 0) for ftrsFakeData in ftrsFake
        ]  # evaluate mean
        diffFtrMeanTrueFake = (
            globalFtrMeanValues.detach() - ftrsMeanFakeData[0].detach()
        )

        lossNetMean = criterionL2Loss(
            netMean.weight, diffFtrMeanTrueFake.detach().view(1, -1)
        )

        lossNetMean.backward()
        avrgLossNetMean += lossNetMean.item()
        optimizerMean.step()
        if opt.scheduler != False:
            schedulerMean.step()

        # updates moving average of variance differences
        ftrsVarFakeData = [torch.var(ftrsFakeData, 0) for ftrsFakeData in ftrsFake]
        diffFtrVarTrueFake = globalFtrVarValues.detach() - ftrsVarFakeData[0].detach()

        lossNetVar = criterionL2Loss(
            netVar.weight, diffFtrVarTrueFake.detach().view(1, -1)
        )

        lossNetVar.backward()
        avrgLossNetVar += lossNetVar.item()
        optimizerVar.step()
        if opt.scheduler != False:
            schedulerVar.step()

        # updates generator
        meanDiffXTrueMean = netMean(globalFtrMeanValues.view(1, -1)).detach()
        meanDiffXFakeMean = netMean(ftrsMeanFakeData[0].view(1, -1))

        varDiffXTrueVar = netVar(globalFtrVarValues.view(1, -1)).detach()
        varDiffXFakeVar = netVar(ftrsVarFakeData[0].view(1, -1))

        lossNetGMean = meanDiffXTrueMean - meanDiffXFakeMean
        avrgLossNetGMean += lossNetGMean.item()

        lossNetGVar = varDiffXTrueVar - varDiffXFakeVar
        avrgLossNetGVar += lossNetGVar.item()

        regularization_loss = torch.tensor(0).to(device)
        if opt.algorithm == "gm":
            regularization_loss = get_regularization_loss(
                netG,
                regularizer=opt.regularization,
                lmbda=opt.lmbda,
                alpha=opt.alpha,
                alpha_prime=opt.alpha_prime,
                device=device,
            )
            regularization_loss.to(device)

        lossNetG = lossNetGMean + lossNetGVar + regularization_loss

        lossNetG.backward()
        optimizerG.step()
        if opt.scheduler != False:
            schedulerG.step()
        if (
            (opt.algorithm == "gm")
            and ((iterId) % (opt.project_freq * opt.freezing_period) == 0)
            and (iterId != 0)
        ):
            prune(netG, update_scores=True)
        if opt.algorithm == "global_ep":
            prune(netG, update_thresholds_only=True)
            reset_flag(netG)

        if (iterId) % opt.numBatchsToValid == 0:
            log.info(
                "[{%d}/{%d}] Loss_Gz: %.6f Loss_GzVar: %.6f Loss_vMean: %.6f Loss_vVar: %.6f"
                % (
                    iterId,
                    opt.niter,
                    avrgLossNetGMean / opt.numBatchsToValid,
                    avrgLossNetGVar / opt.numBatchsToValid,
                    avrgLossNetMean / opt.numBatchsToValid,
                    avrgLossNetVar / opt.numBatchsToValid,
                )
            )
            if opt.algorithm == "gm":
                cp_model = round_model(netG, 0.5, True, 0, None)
                print("sparsity: ", get_model_sparsity(cp_model))

            os.sys.stdout.flush()

            avrgLossNetGMean = 0.0
            avrgLossNetMean = 0.0
            avrgLossNetGVar = 0.0
            avrgLossNetVar = 0.0

        if (iterId + 1) % opt.eval_freq == 0:
            netG.eval()
            iterID_FID = evaluate(opt, log, path_dict, netG, iterId, device)
            FID_list.append(iterID_FID)
            print(
                "Best performance(FID, idx): ",
                min(FID_list),
                opt.eval_freq * (FID_list.index(min(FID_list)) + 1),
            )
            if min(FID_list) == iterID_FID:
                if opt.algorithm in ["ep", "global_ep"]:
                    prune(
                        netG,
                        update_thresholds_only=(
                            True if opt.algorithm == "global_ep" else False
                        ),
                    )
                    saveModel(
                        opt,
                        netG,
                        netMean,
                        netVar,
                        curNetEnc,
                        optimizerG,
                        optimizerMean,
                        optimizerVar,
                        globalFtrMeanValues,
                        globalFtrVarValues,
                        schedulerG,
                        schedulerMean,
                        schedulerVar,
                        suffix="best",
                    )
                    cp_model = round_model(netG, 0.5, True, 0, None)
                    layer_sparsity(cp_model)
                    reset_flag(netG)
                else:
                    saveModel(
                        opt,
                        netG,
                        netMean,
                        netVar,
                        curNetEnc,
                        optimizerG,
                        optimizerMean,
                        optimizerVar,
                        globalFtrMeanValues,
                        globalFtrVarValues,
                        schedulerG,
                        schedulerMean,
                        schedulerVar,
                        suffix="best",
                    )

            fileSuffix = iterId + 1 / opt.eval_freq

            fake = netG(fixed_noise).detach()
            vutils.save_image(
                fake.data[: min(64, opt.batchSize)],
                "%s/%s/%s/fake_samples_iterId_%04d.png"
                % (opt.saveroot, opt.outf, "images", fileSuffix),
                nrow=int(8),
                normalize=True,
                value_range=None,
            )
            del fake

            fake_va = netG(noisev).detach()
            vutils.save_image(
                fake_va.data[: min(64, opt.batchSize)],
                "%s/%s/%s/fake_samples_iterId_%04d.png"
                % (opt.saveroot, opt.outf, "images_va", fileSuffix),
                nrow=int(8),
                normalize=True,
                value_range=None,
            )
            del fake_va

            netG.train()

        # saving models
        if (iterId + 1) % opt.numBatchsToSaveModel == 0:
            if opt.algorithm in ["ep", "global_ep"]:
                prune(netG)
                saveModel(
                    opt,
                    netG,
                    netMean,
                    netVar,
                    curNetEnc,
                    optimizerG,
                    optimizerMean,
                    optimizerVar,
                    globalFtrMeanValues,
                    globalFtrVarValues,
                    schedulerG,
                    schedulerMean,
                    schedulerVar,
                    suffix="newest",
                )
                layer_sparsity(netG)
                reset_flag(netG)
            else:
                saveModel(
                    opt,
                    netG,
                    netMean,
                    netVar,
                    curNetEnc,
                    optimizerG,
                    optimizerMean,
                    optimizerVar,
                    globalFtrMeanValues,
                    globalFtrVarValues,
                    schedulerG,
                    schedulerMean,
                    schedulerVar,
                    suffix="newest",
                )

        # saving model with a different suffix
        if (iterId + 1) % opt.numBatchsToSaveModelToNewFile == 0:
            if opt.algorithm in ["ep", "global_ep"]:
                prune(netG)
                saveModel(
                    opt,
                    netG,
                    netMean,
                    netVar,
                    curNetEnc,
                    optimizerG,
                    optimizerMean,
                    optimizerVar,
                    globalFtrMeanValues,
                    globalFtrVarValues,
                    schedulerG,
                    schedulerMean,
                    schedulerVar,
                    suffix=".%02d" % (iterId / opt.numBatchsToSaveModelToNewFile),
                )
                reset_flag(netG)
            else:
                saveModel(
                    opt,
                    netG,
                    netMean,
                    netVar,
                    curNetEnc,
                    optimizerG,
                    optimizerMean,
                    optimizerVar,
                    globalFtrMeanValues,
                    globalFtrVarValues,
                    schedulerG,
                    schedulerMean,
                    schedulerVar,
                    suffix=".%02d" % (iterId / opt.numBatchsToSaveModelToNewFile),
                )

    if opt.algorithm == "gm" and not opt.differentiate_clamp:
        for name, params in netG.named_parameters():
            if "score" in name:
                scores = params
                with torch.no_grad():
                    scores.data = torch.clamp(scores.data, 0.0, 1.0)

    print("#" * 40)
    print("Finish Training")
    print("Running Time: ", time.time() - start_time)
    print("#" * 40)
    print(
        "Best performance(FID, idx): ",
        min(FID_list),
        opt.eval_freq * (FID_list.index(min(FID_list)) + 1),
    )
