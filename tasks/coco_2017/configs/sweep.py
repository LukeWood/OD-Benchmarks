import ml_collections
import bocas


def get_config():
    config = ml_collections.ConfigDict()

    config.batch_size = 8
    config.augmenter = "basic"
    config.backbone = "ResNet50"
    config.od_model = "RetinaNet"

    config.backbone_weights = "imagenet"
    config.backbone_trainable = True

    return config
