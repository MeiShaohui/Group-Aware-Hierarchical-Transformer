from .m3ddcnn import m3ddcnn
from .cnn3d import cnn3d
from .rssan import rssan
from .ablstm import ablstm
from .dffn import dffn
from .speformer import speformer
from .ssftt import ssftt
from .proposed import proposed


def get_model(model_name, dataset_name, patch_size):
    # example: model_name='cnn3d', dataset_name='pu'
    if model_name == 'm3ddcnn':
        model = m3ddcnn(dataset_name, patch_size)

    elif model_name == 'cnn3d':
        model = cnn3d(dataset_name, patch_size)
    
    elif model_name == 'rssan':
        model = rssan(dataset_name, patch_size)
    
    elif model_name == 'ablstm':
        model = ablstm(dataset_name, patch_size)

    elif model_name == 'dffn':
        model = dffn(dataset_name, patch_size)    
    
    elif model_name == 'speformer':
        model = speformer(dataset_name, patch_size) 

    elif model_name == 'proposed':
        model = proposed(dataset_name, patch_size)

    elif model_name == 'ssftt':
        model = ssftt(dataset_name, patch_size)

    else:
        raise KeyError("{} model is not supported yet".format(model_name))

    return model


