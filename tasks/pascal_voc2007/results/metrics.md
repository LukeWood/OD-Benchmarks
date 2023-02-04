# PascalVOC Results
|    | backbone                    | backbone trainable   | weights   | augmenter   |     loss |      mAP |   Recall |
|---:|:----------------------------|:---------------------|:----------|:------------|---------:|---------:|---------:|
|  0 | keras.applications.ResNet50 | False                | imagenet  | function    | 0.900034 | 0.326701 | 0.396513 |
|  0 | keras.applications.ResNet50 | True                 | imagenet  | function    | 0.844981 | 0.336244 | 0.390878 |
|  0 | keras_cv.models.ResNet50    | True                 | imagenet  | kpl         | 0.855949 | 0.33908  | 0.395572 |
|  0 | keras.applications.ResNet50 | False                | imagenet  | kpl         | 0.913027 | 0.321774 | 0.385703 |
|  0 | keras.applications.ResNet50 | True                 | imagenet  | kpl         | 0.845851 | 0.317074 | 0.372956 |
|  0 | keras_cv.models.ResNet50    | False                | imagenet  | kpl         | 0.88654  | 0.286879 | 0.352644 |