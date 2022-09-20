import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from torchsummaryX import summary

from utils.dataset import load_mat_hsi, sample_gt, HSIDataset
from utils.utils import split_info_print, metrics, show_results
from utils.scheduler import load_scheduler
from models.get_model import get_model
from train import train, test

if __name__ == "__main__":
    # fixed means for all models
    parser = argparse.ArgumentParser(description="run patch-based HSI classification")
    parser.add_argument("--model", type=str, default='cnn3d')
    parser.add_argument("--dataset_name", type=str, default="sa")
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--num_run", type=int, default=5) 
    parser.add_argument("--epoch", type=int, default=200)    
    parser.add_argument("--bs", type=int, default=128)  # bs = batch size  
    parser.add_argument("--ratio", type=float, default=0.1)

    opts = parser.parse_args()

    device = torch.device("cuda:{}".format(opts.device))

    # print parameters
    print("experiments will run on GPU device {}".format(opts.device))
    print("model = {}".format(opts.model))    
    print("dataset = {}".format(opts.dataset_name))
    print("dataset folder = {}".format(opts.dataset_dir))
    print("patch size = {}".format(opts.patch_size))
    print("batch size = {}".format(opts.bs))
    print("total epoch = {}".format(opts.epoch))
    print("{} for training, {} for validation and {} testing".format(opts.ratio / 2, opts.ratio / 2, 1 - opts.ratio))

    # load data
    image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir)

    num_classes = len(labels)
    num_bands = image.shape[-1]

    # random seeds
    seeds = [202201, 202202, 202203, 202204, 202205]

    # empty list to storing results
    results = []

    for run in range(opts.num_run):
        np.random.seed(seeds[run])
        print("running an experiment with the {} model".format(opts.model))
        print("run {} / {}".format(run+1, opts.num_run))

        # get train_gt, val_gt and test_gt
        trainval_gt, test_gt = sample_gt(gt, opts.ratio, seeds[run])
        train_gt, val_gt = sample_gt(trainval_gt, 0.5, seeds[run])
        del trainval_gt

        train_set = HSIDataset(image, train_gt, patch_size=opts.patch_size, data_aug=True)
        val_set = HSIDataset(image, val_gt, patch_size=opts.patch_size, data_aug=False)

        train_loader = torch.utils.data.DataLoader(train_set, opts.bs, drop_last=False, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, opts.bs, drop_last=False, shuffle=False)

        # load model and loss
        model = get_model(opts.model, opts.dataset_name, opts.patch_size)

        if run == 0:
            split_info_print(train_gt, val_gt, test_gt, labels)
            print("network information:")
            with torch.no_grad():
                summary(model, torch.zeros((1, 1, num_bands, opts.patch_size, opts.patch_size)))
        
        model = model.to(device)
        
        optimizer, scheduler = load_scheduler(opts.model, model)

        criterion = nn.CrossEntropyLoss()

        # where to save checkpoint model
        model_dir = "./checkpoints/" + opts.model + '/' + opts.dataset_name + '/' + str(run)

        try:
            train(model, optimizer, criterion, train_loader, val_loader, opts.epoch, model_dir, device, scheduler)
        except KeyboardInterrupt:
            print('"ctrl+c" is pused, the training is over')

        # test the model
        probabilities = test(model, model_dir, image, opts.patch_size, num_classes, device)
        
        prediction = np.argmax(probabilities, axis=-1)

        # computing metrics
        run_results = metrics(prediction, test_gt, n_classes=num_classes)  # only for test set
        results.append(run_results)
        show_results(run_results, label_values=labels)

        del model, train_set, train_loader, val_set, val_loader

    if opts.num_run > 1:
        show_results(results, label_values=labels, agregated=True)


