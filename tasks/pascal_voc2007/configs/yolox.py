import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.batch_size = 16
    config.augmenter = "kpl_yolox"
    config.backbone = "keras_cv.models.CSPDarkNetTiny"
    config.batch_augment = True
    config.backbone_trainable = True
    config.epochs = 300

    config.name = "yolox"
    return config
