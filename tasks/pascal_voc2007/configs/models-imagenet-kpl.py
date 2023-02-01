import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.batch_size = 8
    config.augmenter = "kpl"
    config.backbone = "keras_cv.models.ResNet50-imagenet"
    config.batch_augment = True
    config.backbone_trainable = False

    config.name = "models-imagenet-kpl"
    return config
