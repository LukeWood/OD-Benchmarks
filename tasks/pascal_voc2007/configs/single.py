import bocas
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.batch_size = 8
    config.augmenter = "kpl"
    config.backbone = "keras_cv.models.ResNet50-imagenet"
    config.backbone_trainable = True
    return config
