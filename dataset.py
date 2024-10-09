from logging import root
from typing import Callable
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset
from torch.utils.data import Subset


from typing import Callable
from io import BytesIO


def load_dataset_and_dataloader(opt, eval=False):
    """
    Loading Dataset for training and test(evaluation) datasets
    opt: parser
    eval: bool  If you set eval to True, You get evaluation dataset.
    Returns:
        dataloader
    """
    # Check some error raises
    assert opt.test_num > 0, "The number of image for evaluation should be more than 0"

    if opt.centerCropSize == 0:
        opt.centerCropSize = opt.imageSize

    Dataset_func = dataset_function[opt.dataset]

    dataset = Dataset_func(opt, eval=eval)
    dataloader = get_dataloader(dataset, eval, opt.workers, opt.batchSize)
    return dataloader


#########################################
#       Dataset
#########################################
def ImageNet(opt, eval):
    opt.nc = 3

    if eval:
        transformations = Normalize_for_Eval(
            centerCropSize=opt.centerCropSize, imageSize=opt.imageSize
        )
        eval_state = "test"
    else:
        transformations = Normalize_for_Real(
            nc=opt.nc, centerCropSize=opt.centerCropSize, imageSize=opt.imageSize
        )

        eval_state = "train"
    dataset = dset.ImageNet(
        root=opt.dataroot,
        transform=transforms.Compose(transformations),
        split=eval_state,
        download=True,
    )
    if eval:
        dataset = split_dataset(dataset, opt.test_num, False)
    return dataset


def CelebA(opt, eval):
    opt.nc = 3
    if eval:
        transformations = Normalize_for_Eval(
            centerCropSize=opt.centerCropSize, imageSize=opt.imageSize
        )
        eval_state = "test"
    else:
        transformations = Normalize_for_Real(
            nc=opt.nc, centerCropSize=opt.centerCropSize, imageSize=opt.imageSize
        )
        eval_state = "train"

    dataset = CustomCelebADataset(
        root_dir=opt.dataroot,
        split=eval_state,
        transforms=transforms.Compose(transformations),
    )

    if eval:
        dataset = split_dataset(dataset, opt.test_num, False)
    return dataset


def CIFAR10(opt, eval):
    opt.nc = 3
    if eval:
        transformations = Normalize_for_Eval(
            centerCropSize=None, imageSize=opt.imageSize
        )
        eval_state = False
    else:
        transformations = Normalize_for_Real(
            nc=opt.nc, centerCropSize=None, imageSize=opt.imageSize
        )
        eval_state = True
    dataset = dset.CIFAR10(
        root=opt.dataroot,
        download=True,
        transform=transforms.Compose(transformations),
        train=eval_state,
    )
    if eval:
        dataset = split_dataset(dataset, opt.test_num, False)
    return dataset


def MNIST(opt, eval):
    opt.nc = 1

    if eval:
        transformations = Normalize_for_Eval(
            centerCropSize=None, imageSize=opt.imageSize
        )
        eval_state = False
    else:
        transformations = Normalize_for_Real(
            nc=opt.nc, centerCropSize=None, imageSize=opt.imageSize
        )
        eval_state = True
    dataset = dset.MNIST(
        root=opt.dataroot,
        download=True,
        transform=transforms.Compose(transformations),
        train=eval_state,
    )
    if eval:
        dataset = split_dataset(dataset, opt.test_num, False)
    return dataset


def MultiMNIST(opt, eval):
    opt.nc = 1
    if eval:
        transformations = Normalize_for_Eval(
            centerCropSize=None, imageSize=opt.imageSize
        )
        path = os.path.join(opt.dataroot, "test")
    else:
        transformations = Normalize_for_Real(
            nc=opt.nc, centerCropSize=None, imageSize=opt.imageSize
        )
        path = os.path.join(opt.dataroot, "train")

    dataset = dset.ImageFolder(root=path, transform=transforms.Compose(transformations))
    if eval:
        dataset = split_dataset(dataset, opt.test_num, False)
    return dataset


def LSUN_Bedroom(opt, eval):
    opt.nc = 3
    if eval:
        transformations = Normalize_for_Eval(
            centerCropSize=opt.centerCropSize, imageSize=opt.imageSize
        )
    else:
        transformations = Normalize_for_Real(
            nc=opt.nc, centerCropSize=opt.centerCropSize, imageSize=opt.imageSize
        )
    dataset = dset.LSUN(
        opt.dataroot,
        classes=["bedroom_train"],
        transform=transforms.Compose(transformations),
    )
    if eval:
        dataset = split_dataset(dataset, opt.test_num, False)
    else:
        # Particularly, LSUN Dataset doesn't contain the testset. So We split the trainset into the eval and trainset.
        indices = torch.arange(opt.test_num, len(dataset))
        dataset = Subset(dataset, indices)
    return dataset


def STL10(opt, eval):
    opt.nc = 3
    pass


def FFHQ(opt, eval):
    opt.nc = 3
    if eval:
        transformations = Normalize_for_Eval(
            centerCropSize=opt.centerCropSize, imageSize=opt.imageSize
        )
    else:
        transformations = Normalize_for_Real(
            nc=opt.nc, centerCropSize=opt.centerCropSize, imageSize=opt.imageSize
        )

    dataset = FFHQ_Dataset(opt.dataroot, transform=transforms.Compose(transformations))
    if eval:
        dataset = split_dataset(dataset, opt.test_num, False)
    else:
        # Particularly, FFHQ Dataset doesn't contain the testset. So We split the trainset into the eval and trainset.
        indices = torch.arange(opt.test_num, len(dataset))
        dataset = Subset(dataset, indices)
    return dataset


#########################################
#       Utilizing
#########################################
def split_dataset(dataset, test_num, random_sample=False):
    """
    Params:
        dataset: Dataset
        test_num: int, the number of images for evaluation metrics
        random_sample: bool, If this param is true, you sample randomly.
    Returns:
        dataset: splitted Dataset
    """
    if random_sample:
        # Random sampling from the testset 5000
        indices = torch.randperm(len(dataset))[:test_num]
    else:
        indices = torch.arange(0, test_num)
    dataset = Subset(dataset, indices)
    return dataset


def Normalize_for_Real(nc: int, centerCropSize, imageSize: int) -> list:
    """
    Args:
        nc: int
        centerCropSize: int
        imageSize: int
    """
    transformations = []
    if centerCropSize is not None:
        if centerCropSize > imageSize:
            transformations.extend(
                [
                    transforms.CenterCrop(centerCropSize),
                    transforms.Resize(imageSize),
                    transforms.ToTensor(),
                ]
            )
        else:
            transformations.extend(
                [
                    transforms.Resize(imageSize),
                    transforms.CenterCrop(centerCropSize),
                    transforms.ToTensor(),
                ]
            )

    else:
        transformations = [
            transforms.Resize((imageSize, imageSize)),
            transforms.ToTensor(),
        ]

    if nc == 3:
        # ImageNet stat Normalize
        transformations.extend(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )

    elif nc == 1:
        transformations.extend([transforms.Normalize((0.5,), (0.5,))])
    else:
        raise SystemError
    return transformations


def Normalize_for_Eval(centerCropSize, imageSize: int) -> list:
    """
    Params:
        centerCropSize int:
        imageSize int:
    Returns:
        transformations list:
    """
    # evaluation for dataset
    transformations = []
    if centerCropSize is not None:
        if centerCropSize > imageSize:
            transformations.extend(
                [
                    transforms.CenterCrop(centerCropSize),
                    transforms.Resize(imageSize),
                    ToOnlyTensor(),
                ]
            )
        else:
            transformations.extend(
                [
                    transforms.Resize(imageSize),
                    transforms.CenterCrop(centerCropSize),
                    ToOnlyTensor(),
                ]
            )
    else:
        transformations.extend(
            [transforms.Resize((imageSize, imageSize)), ToOnlyTensor()]
        )
    return transformations


def get_dataloader(dataset, eval, workers, batchSize):
    if eval:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batchSize, shuffle=True, num_workers=int(workers)
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batchSize, shuffle=True, num_workers=int(workers)
        )
    return dataloader


dataset_function = {
    "imagenet": ImageNet,
    "cifar10": CIFAR10,
    "celeba": CelebA,
    "multimnist": MultiMNIST,
    "mnist": MNIST,
    "lsun": LSUN_Bedroom,
    "stl10": STL10,
    "ffhq": FFHQ,
}


#########################################
#   For Loading Custom CelebA Dataset, You should upload your list_eval_partition.csv is under the root folder.
#########################################
class CustomCelebADataset(Dataset):
    def __init__(self, root_dir, split, transforms=None):
        self.image_folder = "img_align_celeba"
        self.root_dir = root_dir

        self.annotation_file = "list_eval_partition.csv"
        self.transform = transforms
        self.split = split
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[self.split]

        df = pd.read_csv(self.root_dir + self.annotation_file)

        self.filename = df.loc[df["partition"] == split_, :].reset_index(drop=True)
        self.length = len(self.filename)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = int(idx)
        img = Image.open(
            os.path.join(
                self.root_dir, self.image_folder, self.filename.iloc[idx,].values[0]
            )
        ).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        target = False
        return img, target


import numpy as np

try:
    import accimage
except ImportError:
    accimage = None


#########################################
#   For Normalizing and Quantization Dataset, ToOnlyTensor is standarization.
#########################################
class ToOnlyTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pic):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.
        Returns:
            Tensor
        """
        default_float_dtype = torch.get_default_dtype()

        if isinstance(pic, np.ndarray):
            # handle numpy array
            if pic.ndim == 2:
                pic = pic[:, :, None]

            img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
            # backward compatibility
            if isinstance(img, torch.ByteTensor):
                return img.to(dtype=default_float_dtype)
            else:
                return img

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic).to(dtype=default_float_dtype)

        # handle PIL Image
        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        img = torch.from_numpy(
            np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
        )

        if pic.mode == "1":
            img = 255 * img
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=default_float_dtype)
        else:
            return img


class FFHQ_Dataset(Dataset):
    """
    Usage:
        Self-coded class for loading the FFHQ data
    """

    def __init__(self, image_folder, transform=None):
        """
        image_folder: image folder
        transform: transform
        """
        images_walk = os.walk(image_folder)
        images_list = []
        for images_file in images_walk:
            images_list = images_list + images_file[2]
        self.images_list = sorted(
            [os.path.join(image_folder, image) for image in images_list]
        )
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        img_id = self.images_list[index]
        code = img_id[-9:]
        dir = img_id[:-7]
        img_dir = dir + "000/" + code
        img = Image.open(img_dir).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        target = False
        return img, target
