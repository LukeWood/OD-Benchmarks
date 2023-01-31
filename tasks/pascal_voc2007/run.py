import ml_experiments

import keras_cv
import loader
import augmenters
import tensorflow as tf

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


def load_datasets(config):
    train_ds = loader.load(
        "train", batch_size=config.batch_size, bounding_box_format="xywh"
    )
    eval_ds = loader.load(
        "test", batch_size=config.batch_size, bounding_box_format="xywh"
    )

    augmenter = augmenters.get(config.augmenter)
    inference_resizing = keras_cv.layers.Resizing(
        640, 640, bounding_box_format="xywh", pad_to_aspect_ratio=True
    )

    if batch_augment:
        train_ds = train_ds.apply(
            tf.data.experimental.dense_to_ragged_batch(BATCH_SIZE)
        )
        train_ds = train_ds.map(
            lambda x: augmenter(x, training=True), num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        train_ds = train_ds.map(
            lambda x: augment(x), num_parallel_calls=tf.data.AUTOTUNE
        )
        train_ds = train_ds.apply(
            tf.data.experimental.dense_to_ragged_batch(BATCH_SIZE)
        )

    eval_ds = eval_ds.apply(tf.data.experimental.dense_to_ragged_batch(BATCH_SIZE))
    eval_ds = eval_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)
    return train_ds, eval_ds


def get_backbone(config):
    if config.backbone == "keras.applications.ResNet50/imagenet":
        inputs = keras.layers.Input(shape=image_size)
        x = inputs
        x = keras.applications.resnet.preprocess_input(x)

        backbone = keras.applications.ResNet50(
            include_top=False, input_tensor=x, weights="imagenet"
        )

        c3_output, c4_output, c5_output = [
            backbone.get_layer(layer_name).output
            for layer_name in [
                "conv3_block4_out",
                "conv4_block6_out",
                "conv5_block3_out",
            ]
        ]
        return keras.Model(inputs=inputs, outputs=[c3_output, c4_output, c5_output])
    if config.backbone == "keras_cv.models.ResNet50/imagenet":
        return (
            keras_cv.models.ResNet50(
                include_top=False, weights="imagenet", include_rescaling=True
            ).as_backbone(),
        )
    raise ValueError(f"Invalid backbone, received backbone={config.backbone}")


def get_model(config):
    model = keras_cv.models.RetinaNet(
        # number of classes to be used in box classification
        classes=20,
        # For more info on supported bounding box formats, visit
        # https://keras.io/api/keras_cv/bounding_box/
        bounding_box_format="xywh",
        # KerasCV offers a set of pre-configured backbones
        backbone=keras_cv.models.ResNet50(
            include_top=False, weights="imagenet", include_rescaling=True
        ).as_backbone(),
    )
    pass


def run(config):
    train_ds, eval_ds = load_datasets(config)
    model = get_model(config)
