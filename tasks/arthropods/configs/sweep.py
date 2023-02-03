import bocas
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.batch_size = 8
    config.backbone = "ResNet50"
    config.od_model = "RetinaNet"
    config.weights = "imagenet"

    config.augmenter = "basic"
    config.backbone_trainable = True
    return config
