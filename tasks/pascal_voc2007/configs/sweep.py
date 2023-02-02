import ml_collections
import bocas


def get_config():
    config = ml_collections.ConfigDict()

    config.batch_size = 8
    config.augmenter = "kpl"
    config.backbone = "keras_cv.models.ResNet50-imagenet"
    #    config.backbone = bocas.Sweep(
    #        ["keras.applications.ResNet50-imagenet", "keras_cv.models.ResNet50-imagenet"]
    #    )
    config.backbone_trainable = bocas.Sweep([True, False])

    config.name = "applications-imagenet-kpl"
    return config
