import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.batch_size = 32
    config.augmenter = "kpl"
    config.backbone = "keras_cv.models.CSPDarkNetTiny"
    config.batch_augment = True
    config.backbone_trainable = True
    config.epochs = 300

    config.name = "yolox"
    return config
