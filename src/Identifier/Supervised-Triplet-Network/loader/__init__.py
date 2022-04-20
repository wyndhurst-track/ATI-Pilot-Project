import json

from loader.cnn_core50 import cnn_core50
from loader.triplet_resnet_core50 import triplet_resnet_core50
from loader.triplet_resnet_core50_softmax import triplet_resnet_core50_softmax
from loader.triplet_resnet_cow_id import triplet_resnet_cow_id
from loader.triplet_resnet_open_cows import triplet_resnet_open_cows

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'cnn_core50':cnn_core50,
        'triplet_resnet_core50': triplet_resnet_core50,
        'triplet_resnet_core50_softmax': triplet_resnet_core50_softmax,
        'triplet_resnet_cow_id': triplet_resnet_cow_id,
        'triplet_resnet_open_cows': triplet_resnet_open_cows
    }[name]

def get_data_path(name, config_file='dataset-config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']
