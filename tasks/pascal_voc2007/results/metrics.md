# PascalVOC Results
|    | backbone                    | backbone trainable   | weights            | augmenter   |     loss |   val_loss |   val_classification_loss |   val_box_loss |      mAP |   Recall |
|---:|:----------------------------|:---------------------|:-------------------|:------------|---------:|-----------:|--------------------------:|---------------:|---------:|---------:|
|  0 | keras_cv.models.ResNet50    | False                | imagenet           | kpl         | 0.88654  |   0.733746 |                  0.377153 |       0.356593 | 0.286879 | 0.352644 |
|  0 | keras_cv.models.ResNet50    | False                | simsiam.openimages | kpl         | 1.18228  |   0.98579  |                  0.538176 |       0.447615 | 0.2062   | 0.233672 |
|  0 | keras.applications.ResNet50 | True                 | imagenet           | function    | 0.844981 |   0.711848 |                  0.375795 |       0.336054 | 0.336244 | 0.390878 |
|  0 | keras.applications.ResNet50 | False                | imagenet           | kpl         | 0.913027 |   0.740511 |                  0.407273 |       0.333237 | 0.321774 | 0.385703 |
|  0 | keras_cv.models.ResNet50    | True                 | imagenet           | kpl         | 0.855949 |   0.679754 |                  0.362206 |       0.317549 | 0.33908  | 0.395572 |
|  0 | keras_cv.models.ResNet50    | True                 | simsiam.openimages | kpl         | 1.08943  |   0.971045 |                  0.56499  |       0.406054 | 0.282208 | 0.32863  |
|  0 | keras.applications.ResNet50 | False                | imagenet           | function    | 0.900034 |   0.71635  |                  0.394131 |       0.322218 | 0.326701 | 0.396513 |
|  0 | keras.applications.ResNet50 | True                 | imagenet           | kpl         | 0.845851 |   0.734732 |                  0.388184 |       0.346548 | 0.317074 | 0.372956 |