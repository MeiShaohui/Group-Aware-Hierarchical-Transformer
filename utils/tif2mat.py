import scipy.io as scipyio
import skimage
from collections import Counter

if __name__ == "__main__":
    print("convert HyRANK-Loukia (HR-L) from .tif to .mat...")

    hrl_image = skimage.io.imread("datasets/hrl/Loukia.tif")
    hrl_gt = skimage.io.imread("datasets/hrl/Loukia_GT.tif")
    print(Counter(hrl_gt.flatten()))

    hrl_image.dtype = 'uint16'
    hrl_gt.dtype = 'uint8'

    print("shape of the image =", hrl_image.shape)
    # shape of the image = (249, 945, 176)

    scipyio.savemat("datasets/hrl/Loukia.mat", mdict=dict(loukia=hrl_image))
    scipyio.savemat("datasets/hrl/Loukia_GT.mat", mdict=dict(loukia_gt=hrl_gt))

    print("finish!")


