import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.batch_size = 64
    config.augmenter = 'function'
    config.backbone = "keras_cv.models.ResNet50/imagenet":
    config.name = 'models-imagenet-fn'
