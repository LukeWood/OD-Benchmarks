import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.batch_size = 64
    config.augmenter = 'function'
    config.backbone = "keras_cv.models.ResNet50/imagenet"
    config.batch_augment = False

    config.name = 'models-imagenet-fn'
    return config
