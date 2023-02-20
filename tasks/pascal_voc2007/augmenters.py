import keras_cv
from fn_augmenters import make_train_function


def get(augmenter, bounding_box_format):
    if augmenter == "function":
        return make_train_function(bounding_box_format, (640, 640, 3))
    if augmenter == "kpl":
        return keras_cv.layers.Augmenter(
            layers=[
                keras_cv.layers.RandomFlip(
                    mode="horizontal", bounding_box_format=bounding_box_format
                ),
                keras_cv.layers.JitteredResize(
                    target_size=(640, 640),
                    scale_factor=(0.8, 1.25),
                    bounding_box_format=bounding_box_format,
                ),
            ]
        )
    if augmenter == "kpl_yolox":
        return keras_cv.layers.Augmenter(
            layers=[
                keras_cv.layers.RandomTranslation(
                    height_factor=(-0.2, 0.2),
                    width_factor=(-0.2, 0.2),
                    bounding_box_format="xywh",
                    fill_mode="constant",
                    fill_value=114,
                ),
                keras_cv.layers.RandomRotation(
                    factor=0.03,
                    bounding_box_format="xywh",
                    fill_mode="constant",
                    fill_value=114,
                ),
                keras_cv.layers.RandomShear(
                    x_factor=0.2,
                    y_factor=0.2,
                    bounding_box_format="xywh",
                    fill_mode="constant",
                    fill_value=114,
                ),
                keras_cv.layers.RandomFlip(
                    bounding_box_format="xywh",
                ),
                # keras_cv.layers.RandomColorJitter(
                #     value_range=(0, 255),
                #     brightness_factor=(0.4,0.4),
                #     contrast_factor=0.0,
                #     saturation_factor=(0.7,0.7),
                #     hue_factor=(0.1,0.1),
                # ),
                keras_cv.layers.MixUp(alpha=0.5),
                keras_cv.layers.Mosaic(bounding_box_format="xywh"),
            ]
        )
    raise ValueError(f"Unimplemented augmenter, {augmenter}")
