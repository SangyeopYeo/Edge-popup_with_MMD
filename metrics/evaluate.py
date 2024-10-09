import math
import torch
import torch.nn.functional as F
import metrics.inception as inception

from metrics.util import prepare_generated_img, get_noise


try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


class EvalModel:
    def __init__(self, eval_embedder, batch_size, device, test_num):
        super(EvalModel, self).__init__()
        self.eval_embedder = eval_embedder
        self.device = device
        self.test_num = test_num

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.mean = torch.Tensor(mean).view(1, 3, 1, 1).to(device)
        self.std = torch.Tensor(std).view(1, 3, 1, 1).to(device)

        self.load_Model(eval_embedder)
        if batch_size is None:
            self.batch_size = 32
        else:
            self.batch_size = batch_size
        self.eval()

    def eval(self):
        self.eval_embedder.eval()

    def load_Model(self, eval_embedder):
        if eval_embedder == "inceptionV3":
            self.eval_embedder = inception.InceptionV3((3,))
        self.eval_embedder = self.eval_embedder.to(self.device)

    def get_embeddings_from_loaders(self, dataloader):
        features_list = []
        total_instance = len(dataloader.dataset)
        num_batches = math.ceil(float(total_instance) / float(self.batch_size))
        data_iter = iter(dataloader)

        start_idx = 0
        with torch.no_grad():
            for _ in tqdm(range(0, num_batches)):
                try:
                    images = torch.tensor(next(data_iter)[0]).to(self.device)
                    images = self.resize_and_normalize(images).to(self.device)
                except StopIteration:
                    break

                features = self.eval_embedder(images).to(self.device)
                features = features.detach().cpu().numpy()
                features_list[start_idx : start_idx + features.shape[0]] = features
                start_idx = start_idx + features.shape[0]
        return features_list

    def get_embeddings_from_generator(self, generator, opt, device):
        features_list = []
        total_instance = self.test_num
        num_batches = math.ceil(float(total_instance) / float(self.batch_size))

        latent = get_noise(opt, self.test_num, device)

        start_idx = 0
        with torch.no_grad():
            for _ in tqdm(range(0, num_batches)):
                images = prepare_generated_img(
                    opt,
                    generator,
                    latent[start_idx : start_idx + self.batch_size],
                    self.device,
                )

                images = torch.FloatTensor(images).to(self.device)
                images = self.resize_and_normalize(images)

                features = self.eval_embedder(images).to(self.device)
                features = features.detach().cpu().numpy()
                features_list[start_idx : start_idx + features.shape[0]] = features
                start_idx = start_idx + features.shape[0]

                if start_idx == self.test_num:
                    break
            return features_list

    def resize_and_normalize(self, x):
        # Resize imageSize -> 299
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        # Convert pixel range 0 ~ 255 to 0 ~ 1 using x /  255.
        x = x / 255.0
        # Convert pixel range 0 ~ 1 to -1 ~ 1 using z-score
        x = (x - self.mean) / self.std
        return x
