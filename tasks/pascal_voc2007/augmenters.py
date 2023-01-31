from fn_augmenters import proc_train_function


def get(augmenter):
    if augmenter == "function":
        return proc_train_function("xywh")
    if augmenter == "kpl":
        return keras_cv.layers.Augmenter(
            layers=[
                keras_cv.layers.RandomFlip(
                    mode="horizontal", bounding_box_format="xywh"
                ),
                keras_cv.layers.JitteredResize(
                    target_size=(640, 640),
                    scale_factor=(0.75, 1.3),
                    bounding_box_format="xywh",
                ),
            ]
        )
    raise ValueError(f"Unimplemented augmenter, {augmenter}")
