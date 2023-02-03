import tensorflow_datasets as tfds
import tensorflow as tf
import keras_cv

def unpackage_raw_tfds(inputs, bounding_box_format):
    image = tf.cast(inputs["image"], tf.float32)
    boxes = inputs["objects"]["bbox"]
    labels = inputs["objects"]["label"]
    bounding_boxes = {"boxes": boxes, "classes": labels}
    bounding_boxes = keras_cv.bounding_box.convert_format(bounding_boxes, source='xywh', target=bounding_box_format)
    return {"images": image, "bounding_boxes": bounding_boxes}


def load(split, bounding_box_format):
    train_ds = tfds.load("coco/2017", split=split)
    train_ds = train_ds.map(
        lambda x: unpackage_raw_tfds(x, bounding_box_format=bounding_box_format),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return train_ds
