import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.batch_size = 64
    config.augmenter = 'kpl'
    config.backbone = "keras.applications.ResNet50/imagenet":

    config.name = 'applications-imagenet-jpl'
