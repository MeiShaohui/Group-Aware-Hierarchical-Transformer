from scipy import io
import os
import numpy as np
import sklearn.model_selection
import torch
import torch.utils.data


def load_mat_hsi(dataset_name, dataset_dir):
    """ load HSI.mat dataset """
    # available sets
    available_sets = [
        'sa',
        'pu',
        'whulk',
        'hrl',
    ]
    assert dataset_name in available_sets, "dataset should be one of" + ' ' + str(available_sets)

    image = None
    gt = None
    labels = None

    if (dataset_name == 'sa'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "Salinas_corrected.mat"))
        image = image['salinas_corrected']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "Salinas_gt.mat"))
        gt = gt['salinas_gt']
        labels = [
            "Undefined",
            "Brocoli_green_weeds_1",
            "Brocoli_green_weeds_2",
            "Fallow",
            "Fallow_rough_plow",
            "Fallow_smooth",
            "Stubble",
            "Celery",
            "Grapes_untrained",
            "Soil_vinyard_develop",
            "Corn_senesced_green_weeds",
            "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk",
            "Lettuce_romaine_6wk",
            "Lettuce_romaine_7wk",
            "Vinyard_untrained",
            "Vinyard_vertical_trellis",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif (dataset_name == 'pu'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "PaviaU.mat"))
        image = image['paviaU']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "PaviaU_gt.mat"))
        gt = gt['paviaU_gt']
        labels = [
            "Undefined",
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Painted metal sheets",
            "Bare Soil",
            "Bitumen",
            "Self-Blocking Bricks",
            "Shadows",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif (dataset_name == 'whulk'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_LongKou.mat"))
        image = image['WHU_Hi_LongKou']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_LongKou_gt.mat"))
        gt = gt['WHU_Hi_LongKou_gt']
        labels = [
            'Undefined',
            'Corn',
            'Cotton',
            'Sesame',
            'Broad-leaf soybean',
            'Narrow-leaf soybean',
            'Rice',
            'Water',
            'Roads and houses',
            'Mixed weed',
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif (dataset_name == 'hrl'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "Loukia.mat"))
        image = image['loukia']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "Loukia_GT.mat"))
        gt = gt['loukia_gt']
        labels = [
            'Undefined',
            'Dense Urban Fabric',
            'Mineral Extraction Sites',
            'Non Irrigated Arable Land',
            'Fruit Trees',
            'Olive Groves',
            'Broad-leaved Forest',
            'Coniferous Forest',
            'Mixed Forest',
            'Dense Sclerophyllous Vegetation',
            'Sparce Sclerophyllous Vegetation',
            'Sparcely Vegetated Areas',
            'Rocks and Sand',
            'Water',
            'Coastal Water'
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    # after getting image and ground truth (gt), let us do data preprocessing!
    # step1 filter nan values out
    nan_mask = np.isnan(image.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print("warning: nan values found in dataset {}, using 0 replace them".format(dataset_name))
        image[nan_mask] = 0
        gt[nan_mask] = 0

    # step2 normalise the HSI data (method from SSAN, TGRS 2020)
    image = np.asarray(image, dtype=np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    mean_by_c = np.mean(image, axis=(0, 1))
    for c in range(image.shape[-1]):
        image[:, :, c] = image[:, :, c] - mean_by_c[c]

    # step3 set undefined index 0 to -1, so class index starts from 0
    gt = gt.astype('int') - 1

    # step4 remove undefined label
    labels = labels[1:]

    return image, gt, labels


def sample_gt(gt, percentage, seed):
    """
    :param gt: 2d int array, -1 for undefined or not selected, index starts at 0
    :param percentage: for example, 0.1 for 10%, 0.02 for 2%, 0.5 for 50%
    :param seed: random seed
    :return:
    """
    indices = np.where(gt >= 0)
    X = list(zip(*indices))
    y = gt[indices].ravel()

    train_gt = np.full_like(gt, fill_value=-1)
    test_gt = np.full_like(gt, fill_value=-1)

    train_indices, test_indices = sklearn.model_selection.train_test_split(
        X,
        train_size=percentage,
        random_state=seed,
        stratify=y
    )

    train_indices = [list(t) for t in zip(*train_indices)]
    test_indices = [list(t) for t in zip(*test_indices)]

    train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
    test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    return train_gt, test_gt


class HSIDataset(torch.utils.data.Dataset):
    def __init__(self, image, gt, patch_size, data_aug=True):
        """
        :param image: 3d float np array of HSI, image
        :param gt: train_gt or val_gt or test_gt
        :param patch_size: 7 or 9 or 11 ...
        :param data_aug: whether to use data augment, default is True
        """
        super().__init__()
        self.data_aug = data_aug
        self.patch_size = patch_size
        self.ps = self.patch_size // 2  # padding size
        self.data = np.pad(image, ((self.ps, self.ps), (self.ps, self.ps), (0, 0)), mode='reflect')
        self.label = np.pad(gt, ((self.ps, self.ps), (self.ps, self.ps)), mode='reflect')

        mask = np.ones_like(self.label)
        mask[self.label < 0] = 0
        x_pos, y_pos = np.nonzero(mask)

        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)
                                 if self.ps <= x < image.shape[0] + self.ps
                                 and self.ps <= y < image.shape[1] + self.ps])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    def hsi_augment(self, data):
        # e.g. (7 7 200) data = numpy array float32
        do_augment = np.random.random()
        if do_augment > 0.5:
            prob = np.random.random()
            if 0 <= prob <= 0.2:
                data = np.fliplr(data)
            elif 0.2 < prob <= 0.4:
                data = np.flipud(data)
            elif 0.4 < prob <= 0.6:
                data = np.rot90(data, k=1)
            elif 0.6 < prob <= 0.8:
                data = np.rot90(data, k=2)
            elif 0.8 < prob <= 1.0:
                data = np.rot90(data, k=3)
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x, y]

        if self.data_aug:
            # Perform data augmentation (only on 2D patches)
            data = self.hsi_augment(data)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        # Add a fourth dimension for 3D CNN

        data = data.unsqueeze(0)

        return data, label

