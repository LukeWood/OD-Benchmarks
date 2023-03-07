import os
import resource
import sys
import math

import augmenters
import bocas
import keras_cv
import loader
import tensorflow as tf
import termcolor
from absl import app, flags
from keras_cv.callbacks import PyCOCOCallback
from luketils import visualization
from tensorflow import keras


image_size = (640, 640, 3)

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


def unpackage_dict_format(inputs):
    return inputs["images"], keras_cv.bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32
    )


def load_datasets(config, bounding_box_format):
    train_ds = loader.load("train", bounding_box_format=bounding_box_format)
    eval_ds = loader.load("test", bounding_box_format=bounding_box_format)

    augmenter = augmenters.get(
        config.augmenter, bounding_box_format=bounding_box_format
    )
    inference_resizing = keras_cv.layers.Resizing(
        640, 640, bounding_box_format=bounding_box_format, pad_to_aspect_ratio=True
    )

    if config.augmenter == "kpl_yolox" or config.augmenter == "kpl":
        train_ds = train_ds.apply(
            tf.data.experimental.dense_to_ragged_batch(config.batch_size)
        )
        train_ds = train_ds.map(
            lambda x: augmenter(x, training=True), num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        train_ds = train_ds.map(
            lambda x: augmenter(x), num_parallel_calls=tf.data.AUTOTUNE
        )
        train_ds = train_ds.apply(
            tf.data.experimental.dense_to_ragged_batch(config.batch_size)
        )
    eval_ds = eval_ds.apply(
        tf.data.experimental.dense_to_ragged_batch(config.batch_size)
    )
    eval_ds = eval_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.map(unpackage_dict_format, num_parallel_calls=tf.data.AUTOTUNE)
    eval_ds = eval_ds.map(unpackage_dict_format, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    eval_ds = eval_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, eval_ds


def get_backbone(config):
    if config.backbone == "keras.applications.ResNet50-imagenet":
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
    if config.backbone == "keras_cv.models.ResNet50-imagenet":
        return keras_cv.models.ResNet50(
            include_top=False, weights="imagenet", include_rescaling=True
        ).as_backbone()
    if config.backbone == "keras_cv.models.ResNet50-simsiam.openimages-prototype":
        return keras_cv.models.ResNet50(
            include_top=False,
            weights="gs://keras-cv/models/resnet50/openimages-simsiam-v3.h5",
            include_rescaling=True,
        ).as_backbone()
    if config.backbone == "keras_cv.models.CSPDarkNetTiny":
        return keras_cv.models.CSPDarkNetTiny(
            include_rescaling=False, include_top=False,
        ).as_backbone(min_level=3) 
    raise ValueError(f"Invalid backbone, received backbone={config.backbone}")


def get_model(config):
    model = keras_cv.models.YoloX_tiny(
        classes=20,
        bounding_box_format="xywh",
        backbone=get_backbone(config),
    )
    model.backbone.trainable = config.backbone_trainable
    return model


def get_name(config):
    return f"{config.backbone}-trainable={config.backbone_trainable}-{config.augmenter}"


class_ids = [
    "Aeroplane",
    "Bicycle",
    "Bird",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cow",
    "Dining Table",
    "Dog",
    "Horse",
    "Motorbike",
    "Person",
    "Potted Plant",
    "Sheep",
    "Sofa",
    "Train",
    "Tvmonitor",
    "Total",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))


def visualize_dataset(dataset, bounding_box_format, path):
    sample = next(iter(dataset))
    images, boxes = sample
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=boxes,
        scale=4,
        rows=2,
        cols=2,
        thickness=4,
        font_scale=1,
        class_mapping=class_mapping,
        path=path,
    )


def run(config):
    name = get_name(config)

    termcolor.cprint(termcolor.colored("#" * 10, "cyan"))
    termcolor.cprint(
        termcolor.colored(f"Training model: {name}", "green", attrs=["bold"])
    )
    termcolor.cprint(termcolor.colored("#" * 10, "cyan"))
    train_ds, eval_ds = load_datasets(config, bounding_box_format="xywh")
    model = get_model(config)

    result_dir = f"artifacts/{name}"
    os.makedirs(result_dir, exist_ok=True)

    visualize_dataset(
        train_ds, bounding_box_format="xywh", path=f"{result_dir}/train.png"
    )
    visualize_dataset(
        eval_ds, bounding_box_format="xywh", path=f"{result_dir}/eval.png"
    )

    base_lr = 0.01 * config.batch_size / 64
    optimizer = tf.optimizers.SGD(
        learning_rate=base_lr,
        weight_decay=0.0005,
        global_clipnorm=10.0,
        momentum=0.01,
    )

    model.compile(
        classification_loss="binary_crossentropy",
        objectness_loss="binary_crossentropy",
        box_loss="iou",
        optimizer=optimizer,
    )

    def get_lr_callback(batch_size=8):
        lr_start = 1e-6
        lr_max = 0.01 * (batch_size / 64)
        lr_min = lr_max * 0.05
        warmup_epochs = 5
        no_aug_iter = 15

        def lrfn(epoch):
            if epoch <= warmup_epochs:
                lr = (lr_max - lr_start) * pow((epoch / warmup_epochs), 2) + lr_start

            elif epoch >= config.epochs - no_aug_iter:
                lr = lr_min

            else:
                lr = lr_min + 0.5 * (lr_max - lr_min) * (
                    1.0
                    + math.cos(
                        math.pi
                        * (epoch - warmup_epochs)
                        / (config.epochs - warmup_epochs - no_aug_iter)
                    )
                )
            return lr

        lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
        return lr_callback

    history = model.fit(
        train_ds,
        validation_data=eval_ds,
        epochs=config.epochs,
        callbacks=[
            PyCOCOCallback(eval_ds, "xywh"),
            keras.callbacks.TerminateOnNaN(),
            get_lr_callback(config.batch_size),
        ],
    )
    # metrics = model.evaluate(eval_ds, return_dict=True)
    return bocas.Result(
        # Must be generated for sweeps
        name=name,
        config=config,
        artifacts=[
            bocas.artifacts.KerasHistory(history, name="history"),
        ],
    )
