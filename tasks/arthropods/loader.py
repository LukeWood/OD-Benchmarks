import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds

TRAIN_DATA = (
    "gs://practical-ml-vision-book/arthropod_detection_tfr/size_w1024px/*.train.tfrec"
)
VALID_DATA = (
    "gs://practical-ml-vision-book/arthropod_detection_tfr/size_w1024px/*.test.tfrec"
)

CLASSES = [
    "Lepidoptera",
    "Hymenoptera",
    "Hemiptera",
    "Odonata",
    "Diptera",
    "Araneae",
    "Coleoptera",
    "_truncated",
    "_blurred",
    "_occluded",
]

CLASSES = [
    klass
    for klass in RAW_CLASSES
    if klass
    not in [
        "_truncated",
        "_blurred",
        "_occluded",
    ]
]
AUTO = tf.data.AUTOTUNE

features = tfds.features.FeaturesDict(
    {
        "image/encoded": tfds.features.Image(
            shape=(None, None, 3), encoding_format="jpeg"
        ),  # automatic JPEG compression/decompresion
        "image/width": tfds.features.Tensor(shape=(), dtype=tf.int64),
        "image/height": tfds.features.Tensor(shape=(), dtype=tf.int64),
        "image/source_id": tfds.features.Tensor(shape=(), dtype=tf.string),
        "image/object/class/label": tfds.features.Sequence(
            tfds.features.Tensor(shape=(), dtype=tf.int64)
        ),
        "image/object/bbox/xmin": tfds.features.Sequence(
            tfds.features.Tensor(shape=(), dtype=tf.float32)
        ),
        "image/object/bbox/ymin": tfds.features.Sequence(
            tfds.features.Tensor(shape=(), dtype=tf.float32)
        ),
        "image/object/bbox/xmax": tfds.features.Sequence(
            tfds.features.Tensor(shape=(), dtype=tf.float32)
        ),
        "image/object/bbox/ymax": tfds.features.Sequence(
            tfds.features.Tensor(shape=(), dtype=tf.float32)
        ),
    }
)


def unpackage_raw_inputs(inputs, bounding_box_format):
    # get the image, which is no longer encoded here thanks to tfds.features.Image
    image = data["image/encoded"]
    # boxes extracted in rel_xyxy format
    boxes = tf.stack(
        [
            data["image/object/bbox/xmin"],
            data["image/object/bbox/ymin"],
            data["image/object/bbox/xmax"],
            data["image/object/bbox/ymax"],
        ],
        axis=-1,
    )
    classes = data["image/object/class/label"]
    classes = classes - 1  # classes are 1-based in this dataset, converting to 0-based
    bounding_boxes = {"boxes": boxes, "classes": classes}

    bounding_boxes = keras_cv.bounding_box.convert_format(
        bounding_boxes, source="rel_xyxy", target=bounding_box_format, images=image
    )
    # format in Keras-CV standard
    return {"images": image, "bounding_boxes": bounding_boxes}


def format_inputs(data, bounding_box_format):
    data["images"] = tf.cast(data["images"], dtype=tf.float32)
    data["bounding_boxes"] = keras_cv.bounding_box.clip_to_image(
        bounding_boxes=data["bounding_boxes"],
        images=data["images"],
        bounding_box_format=bounding_box_format,
    )
    return data


def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return int(np.sum(n))


def load(split, bounding_box_format):
    if split not in ["train", "test"]:
        raise ValueError(f"Invalid data split, {split}")
    filename_pattern = TRAIN_DATA if split == "train" else VALID_DATA
    # list files and decode TFRecords
    filenames = tf.io.gfile.glob(filename_pattern)
    num_samples = count_data_items(filenames)
    ignore_order = (
        tf.data.Options()
    )  # Order does not matter since we will be shuffling the data anyway.
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(
        filenames, num_parallel_reads=AUTO
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(features.deserialize_example, num_parallel_calls=AUTO)

    # process the data
    dataset = dataset.map(
        lambda x: unpackage_raw_inputs(x, bounding_box_format), num_parallel_calls=AUTO
    )
    dataset = dataset.map(
        lambda x: format_inputs(x, bounding_box_format), num_parallel_calls=AUTO
    )

    return dataset, num_samples
