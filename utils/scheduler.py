import torch.optim as optim


def load_scheduler(model_name, model):
    optimizer, scheduler = None, None
    
    if model_name == 'm3ddcnn':
        optimizer = optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.01)
        scheduler = None

    elif model_name == 'cnn3d':
        # MaxEpoch in the paper is unknown, so 300 is set as MaxEpoch
        # and paper said: for each (Max Epoch / 3) iteration, the learning rate is divided by 10
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], gamma=0.1)

    elif model_name == 'rssan':
        optimizer = optim.RMSprop(model.parameters(), lr=0.0003, weight_decay=0.0, momentum=0.0)
        scheduler = None

    elif model_name == 'ablstm':
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [150], gamma=0.1)

    elif model_name == 'dffn':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], gamma=0.1)

    elif model_name == 'speformer':
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120, 150, 180, 210, 240, 270], gamma=0.9)

    elif model_name == 'ssftt':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = None

    elif model_name == 'proposed':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
        scheduler = None

    return optimizer, scheduler


