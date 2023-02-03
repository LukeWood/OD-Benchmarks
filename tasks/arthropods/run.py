import os
import resource
import sys

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
    train_ds, num_train = loader.load("train", bounding_box_format=bounding_box_format)
    eval_ds, num_test = loader.load("test", bounding_box_format=bounding_box_format)

    train_ds = train_ds.shuffle(1024)
    eval_ds = eval_ds.shuffle(1024)

    augmenter = augmenters.get(
        config.augmenter, bounding_box_format=bounding_box_format
    )
    inference_resizing = keras_cv.layers.Resizing(
        640, 640, bounding_box_format=bounding_box_format, pad_to_aspect_ratio=True
    )
    train_ds = train_ds.apply(
        tf.data.experimental.dense_to_ragged_batch(config.batch_size)
    )
    train_ds = train_ds.map(
        lambda x: augmenter(x, training=True), num_parallel_calls=tf.data.AUTOTUNE
    )

    eval_ds = eval_ds.apply(
        tf.data.experimental.dense_to_ragged_batch(config.batch_size)
    )
    eval_ds = eval_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.map(unpackage_dict_format, num_parallel_calls=tf.data.AUTOTUNE)
    eval_ds = eval_ds.map(unpackage_dict_format, num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds, eval_ds, num_train, num_test


def get_backbone(config):
    if config.backbone == "ResNet50":
        return keras_cv.models.ResNet50(
            include_top=False, weights=config.weights, include_rescaling=True
        ).as_backbone()
    raise ValueError(f"Invalid backbone, received backbone={config.backbone}")


def get_model(config):
    if config.od_model == "RetinaNet":
        model = keras_cv.models.RetinaNet(
            classes=len(class_ids),
            bounding_box_format="xywh",
            backbone=get_backbone(config),
        )
        model.backbone.trainable = config.backbone_trainable
        return model
    raise ValueError(f"Invalid OD Model: {config.od_model}")


def get_name(config):
    return f"{config.backbone}-trainable={config.backbone_trainable}-{config.augmenter}"


class_ids = [
    "Lepidoptera",
    "Hymenoptera",
    "Hemiptera",
    "Odonata",
    "Diptera",
    "Araneae",
    "Coleoptera",
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
    train_ds, eval_ds, num_train, num_test = load_datasets(
        config, bounding_box_format="xywh"
    )
    model = get_model(config)

    result_dir = f"artifacts/{name}"
    os.makedirs(result_dir, exist_ok=True)

    visualize_dataset(
        train_ds, bounding_box_format="xywh", path=f"{result_dir}/train.png"
    )
    visualize_dataset(
        eval_ds, bounding_box_format="xywh", path=f"{result_dir}/eval.png"
    )

    train_steps_per_epoch = num_train // config.batch_size
    test_steps_per_epoch = num_test // config.batch_size

    base_lr = 0.05
    lr_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[train_steps_per_epoch * 16, train_steps_per_epoch * 32],
        values=[base_lr, 0.1 * base_lr, 0.01 * base_lr],
    )
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_decay, momentum=0.9, global_clipnorm=10.0
    )

    model.compile(
        classification_loss="focal",
        box_loss="smoothl1",
        optimizer=optimizer,
    )

    history = model.fit(
        train_ds,
        validation_data=eval_ds,
        epochs=50,
        callbacks=[
            PyCOCOCallback(eval_ds.take(200), "xywh"),
            keras.callbacks.TerminateOnNaN(),
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
