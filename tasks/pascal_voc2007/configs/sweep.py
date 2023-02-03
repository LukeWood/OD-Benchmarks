import bocas
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.batch_size = 8
    config.augmenter = bocas.Sweep(["kpl", "function"])
    config.backbone = bocas.Sweep(
        ["keras.applications.ResNet50-imagenet", "keras_cv.models.ResNet50-imagenet"]
    )
    config.backbone_trainable = bocas.Sweep([True, False])
    return config
