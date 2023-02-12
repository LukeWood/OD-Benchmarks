import bocas
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.batch_size = 8
    config.augmenter = "kpl"
    config.backbone = bocas.Sweep(
        [
            "keras_cv.models.ResNet51-simsiam.openimages-prototype",
        ]
    )
    config.backbone_trainable = bocas.Sweep([False, True])
    return config
