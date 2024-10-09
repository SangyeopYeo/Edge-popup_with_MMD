from prdc import compute_prdc
import numpy as np


def iprdc(real_pred, fake_pred):
    nearest_k = 5
    real_pred = np.array(real_pred)
    fake_pred = np.array(fake_pred)
    prdc = compute_prdc(
        real_features=real_pred, fake_features=fake_pred, nearest_k=nearest_k
    )
    return prdc
