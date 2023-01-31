def unpackage_tfds_inputs(inputs, bounding_box_format):
    image = inputs["image"]
    boxes = keras_cv.bounding_box.convert_format(
        inputs["objects"]["bbox"],
        images=image,
        source="rel_yxyx",
        target=bounding_box_format,
    )
    bounding_boxes = {
        "classes": tf.cast(inputs["objects"]["label"], dtype=tf.float32),
        "boxes": tf.cast(boxes, dtype=tf.float32),
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}


def load(split, bounding_box_format):
    if split == "train":
        ds = tfds.load(
            "voc/2007", split="train+validation", with_info=False, shuffle_files=True
        )
        # add pascal 2012 dataset to augment the training set
        ds = ds.concatenate(
            tfds.load(
                "voc/2012",
                split="train+validation",
                with_info=False,
                shuffle_files=True,
            )
        )
    if split == "test":
        ds = tfds.load("voc/2007", split="test", with_info=False)

    ds = ds.map(
        lambda x: unpackage_tfds_inputs(x, bounding_box_format),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds
