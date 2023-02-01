from fn_augmenters import make_train_function
import keras_cv

def get(augmenter):
    if augmenter == "function":
        return make_train_function("xywh", (640 ,640, 3))
    if augmenter == "kpl":
        return keras_cv.layers.Augmenter(
            layers=[
                keras_cv.layers.RandomFlip(
                    mode="horizontal", bounding_box_format="xywh"
                ),
                keras_cv.layers.JitteredResize(
                    target_size=(640, 640),
                    scale_factor=(0.8, 1.25),
                    bounding_box_format="xywh",
                ),
            ]
        )
    raise ValueError(f"Unimplemented augmenter, {augmenter}")
