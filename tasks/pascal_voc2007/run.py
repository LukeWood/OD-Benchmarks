import keras_cv
import bocas
import tensorflow as tf
import sys
import augmenters
import loader
import resource
from absl import flags
from absl import app
from tensorflow import keras
from keras_cv.callbacks import PyCOCOCallback
import os
from luketils import visualization

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

    if config.augmenter == "kpl":
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
    raise ValueError(f"Invalid backbone, received backbone={config.backbone}")


def get_model(config):
    model = keras_cv.models.RetinaNet(
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
    train_ds, eval_ds = load_datasets(config, bounding_box_format="xywh")
    model = get_model(config)

    name = get_name(config)
    result_dir = f"artifacts/{name}"
    os.makedirs(result_dir, exist_ok=True)

    visualize_dataset(
        train_ds, bounding_box_format="xywh", path=f"{result_dir}/train.png"
    )
    visualize_dataset(
        eval_ds, bounding_box_format="xywh", path=f"{result_dir}/eval.png"
    )
    return
    base_lr = 0.01
    lr_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[12000 * 16, 16000 * 16],
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
        callbacks=[PyCOCOCallback(eval_ds, "xywh"), keras.callbacks.TerminateOnNaN()],
    )
    # metrics = model.evaluate(eval_ds, return_dict=True)
    return bocas.Result(
        # Must be generated for sweeps
        name=name,
        config=config,
        artifacts=[
            bocas.artifacts.KerasHistory(history, name="history"),
            bocas.artifacts.Metrics(metrics, name="metrics"),
        ],
    )
