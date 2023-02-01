import tensorflow as tf
import keras_cv


def resize_and_crop_image(
    image,
    desired_size,
    padded_size,
    aug_scale_min=1.0,
    aug_scale_max=1.0,
    seed=1,
    method=tf.image.ResizeMethod.BILINEAR,
):
    with tf.name_scope("resize_and_crop_image"):
        image_size = tf.cast(tf.shape(image)[0:2], tf.float32)

        random_jittering = aug_scale_min != 1.0 or aug_scale_max != 1.0

        if random_jittering:
            random_scale = tf.random.uniform(
                [], aug_scale_min, aug_scale_max, seed=seed
            )
            scaled_size = tf.round(random_scale * desired_size)
        else:
            scaled_size = desired_size

        scale = tf.minimum(
            scaled_size[0] / image_size[0], scaled_size[1] / image_size[1]
        )
        scaled_size = tf.round(image_size * scale)

        # Computes 2D image_scale.
        image_scale = scaled_size / image_size

        # Selects non-zero random offset (x, y) if scaled image is larger than
        # desired_size.
        if random_jittering:
            max_offset = scaled_size - desired_size
            max_offset = tf.where(
                tf.less(max_offset, 0), tf.zeros_like(max_offset), max_offset
            )
            offset = max_offset * tf.random.uniform(
                [
                    2,
                ],
                0,
                1,
                seed=seed,
            )
            offset = tf.cast(offset, tf.int32)
        else:
            offset = tf.zeros((2,), tf.int32)

        scaled_image = tf.image.resize(
            image, tf.cast(scaled_size, tf.int32), method=method
        )

        if random_jittering:
            scaled_image = scaled_image[
                offset[0] : offset[0] + desired_size[0],
                offset[1] : offset[1] + desired_size[1],
                :,
            ]

        output_image = tf.image.pad_to_bounding_box(
            scaled_image, 0, 0, padded_size[0], padded_size[1]
        )

        image_info = tf.stack(
            [
                image_size,
                tf.constant(desired_size, dtype=tf.float32),
                image_scale,
                tf.cast(offset, tf.float32),
            ]
        )
        return output_image, image_info


def resize_and_crop_boxes(boxes, image_scale, output_size, offset):
    with tf.name_scope("resize_and_crop_boxes"):
        # Adjusts box coordinates based on image_scale and offset.
        boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
        boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
        # Clips the boxes.
        boxes = clip_boxes(boxes, output_size)
        return boxes


def clip_boxes(boxes, image_shape):
    if boxes.shape[-1] != 4:
        raise ValueError(
            "boxes.shape[-1] is {:d}, but must be 4.".format(boxes.shape[-1])
        )

    with tf.name_scope("clip_boxes"):
        if isinstance(image_shape, list) or isinstance(image_shape, tuple):
            height, width = image_shape
            max_length = [height, width, height, width]
        else:
            image_shape = tf.cast(image_shape, dtype=boxes.dtype)
            height, width = tf.unstack(image_shape, axis=-1)
            max_length = tf.stack([height, width, height, width], axis=-1)

        clipped_boxes = tf.math.maximum(tf.math.minimum(boxes, max_length), 0.0)
        return clipped_boxes


def get_non_empty_box_indices(boxes):
    # Selects indices if box height or width is 0.
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    indices = tf.where(tf.logical_and(tf.greater(height, 0), tf.greater(width, 0)))
    return indices[:, 0]


def resize_fn(image, boxes, classes):
    image, image_info = resize_and_crop_image(image, (640, 640), (640, 640), 0.8, 1.25)
    boxes = resize_and_crop_boxes(
        boxes, image_info[2, :], image_info[1, :], image_info[3, :]
    )
    indices = get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)
    return image, boxes, classes


def flip_fn(image, boxes):
    if tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32) > 0.5:
        image = tf.image.flip_left_right(image)
        y1, x1, y2, x2 = tf.split(boxes, num_or_size_splits=4, axis=-1)
        boxes = tf.concat([y1, 1.0 - x2, y2, 1.0 - x1], axis=-1)
    return image, boxes


def make_train_function(bounding_box_format, img_size):
    def apply(inputs):
        image = inputs["images"]
        boxes = inputs["bounding_boxes"]["boxes"]
        classes = inputs["bounding_boxes"]["classes"]
        bounding_boxes = keras_cv.bounding_box.convert_format(
            boxes, images=image, source=bounding_box_format, target="yxyx"
        )
        image, boxes = flip_fn(image, boxes)
        image, boxes, classes = resize_fn(image, boxes, classes)
        bounding_boxes = keras_cv.bounding_box.convert_format(
            boxes, images=image, source="yxyx", target=bounding_box_format
        )
        bounding_boxes = {"boxes": bounding_boxes, "classes": classes}
        return {"images": image, "bounding_boxes": bounding_boxes}

    return apply


def make_eval_function(bounding_box_format, target_size):
    def apply(inputs):
        raw_image = inputs["image"]
        raw_image = tf.cast(raw_image, tf.float32)

        img_size = tf.shape(raw_image)
        height = img_size[0]
        width = img_size[1]

        target_height = tf.cond(
            height > width,
            lambda: float(IMG_SIZE),
            lambda: tf.cast(height / width * IMG_SIZE, tf.float32),
        )
        target_width = tf.cond(
            width > height,
            lambda: float(IMG_SIZE),
            lambda: tf.cast(width / height * IMG_SIZE, tf.float32),
        )
        image = tf.image.resize(
            raw_image, (target_height, target_width), antialias=False
        )

        boxes = keras_cv.bounding_box.convert_format(
            inputs["objects"]["bbox"],
            images=image,
            source="rel_yxyx",
            target="xyxy",
        )
        image = tf.image.pad_to_bounding_box(
            image, 0, 0, target_size[0], target_size[1]
        )
        boxes = keras_cv.bounding_box.convert_format(
            boxes,
            images=image,
            source="xyxy",
            target=bounding_box_format,
        )
        classes = tf.cast(inputs["objects"]["label"], tf.float32)
        bounding_boxes = {"boxes": boxes, "classes": classes}
        return {"images": image, "bounding_boxes": bounding_boxes}

    return apply
