import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.batch_size = 2
    config.augmenter = 'function'
    config.backbone = "keras.applications.ResNet50/imagenet"
    config.batch_augment = False

    config.name = 'applications-imagenet-fn'
    return config
