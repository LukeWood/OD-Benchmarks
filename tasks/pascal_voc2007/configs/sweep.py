import ml_collections
import ml_experiments


def get_config():
    config = ml_collections.ConfigDict()

    config.batch_size = 8
    config.augmenter = "kpl"

    config.backbone = ml_experiments.Sweep(
        ["keras.applications.ResNet50-imagenet", "keras_cv.models.ResNet50-imagenet"]
    )
    config.backbone_trainable = ml_experiments.Sweep([True, False])

    config.batch_augment = True
    config.name = "applications-imagenet-kpl"
    return config
