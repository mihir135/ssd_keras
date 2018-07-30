import cv2
import os
import matplotlib

matplotlib.use('TkAgg')
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast


def inti_tensorflow():
    # init tensorflow session
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto()
    # set gpu mem usage ratio
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf_session = tf.Session(config=config)


class SSD:
    def __init__(self, weights_path, img_width, img_height, auto_build=True):
        self.weights_path = weights_path
        # Set the image size.
        self.img_width = img_width
        self.img_height = img_height
        self.img_path = None

        self.classes = ['background',
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat',
                        'chair', 'cow', 'diningtable', 'dog',
                        'horse', 'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']

        # auto build model
        if auto_build:
            if self.img_width == 300 and self.img_width == 300:
                self.build_model_300()
            elif self.img_width == 512 and self.img_width == 512:
                self.build_model_512()

    def build_model_300(self):
        # 1: Build the Keras model

        K.clear_session()  # Clear previous models from memory.

        self.model = ssd_300(image_size=(self.img_height, self.img_width, 3),
                        n_classes=20,
                        mode='inference',
                        l2_regularization=0.0005,
                        scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                        # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                 [1.0, 2.0, 0.5],
                                                 [1.0, 2.0, 0.5]],
                        two_boxes_for_ar1=True,
                        steps=[8, 16, 32, 64, 100, 300],
                        offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        clip_boxes=False,
                        variances=[0.1, 0.1, 0.2, 0.2],
                        normalize_coords=True,
                        subtract_mean=[123, 117, 104],
                        swap_channels=[2, 1, 0],
                        confidence_thresh=0.5,
                        iou_threshold=0.45,
                        top_k=200,
                        nms_max_output_size=400)

        # 2: Load the trained weights into the model.

        # TODO: Set the path of the trained weights.
        weights_path = self.weights_path

        self.model.load_weights(weights_path, by_name=True)

        # 3: Compile the model so that Keras won't complain the next time you load it.

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

        self.model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    def build_model_512(self):
        # 1: Build the Keras model
        K.clear_session()  # Clear previous models from memory.

        self.model = ssd_512(image_size=(self.img_height, self.img_width, 3),
                             n_classes=20,
                             mode='inference',
                             l2_regularization=0.0005,
                             scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
                             # The scales for MS COCO are [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]
                             aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                      [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                      [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                      [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                      [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                      [1.0, 2.0, 0.5],
                                                      [1.0, 2.0, 0.5]],
                             two_boxes_for_ar1=True,
                             steps=[8, 16, 32, 64, 128, 256, 512],
                             offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                             clip_boxes=False,
                             variances=[0.1, 0.1, 0.2, 0.2],
                             normalize_coords=True,
                             subtract_mean=[123, 117, 104],
                             swap_channels=[2, 1, 0],
                             confidence_thresh=0.5,
                             iou_threshold=0.45,
                             top_k=200,
                             nms_max_output_size=400)

        # 2: Load the trained weights into the model.

        # TODO: Set the path of the trained weights.
        self.model.load_weights(self.weights_path, by_name=True)

        # 3: Compile the model so that Keras won't complain the next time you load it.

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

        self.model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    def load_img(self, img_path):
        self.img_path = img_path
        self.input_images = []  # Store resized versions of the images here.
        img = image.load_img(img_path, target_size=(self.img_height, self.img_width))
        img = image.img_to_array(img)
        self.input_images.append(img)
        self.input_images = np.array(self.input_images)

    def perdict(self):
        y_pred = self.model.predict(self.input_images)

        confidence_threshold = 0.5

        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

        # class, confidence, x1, y1, x2, y2
        self.result = y_pred_thresh[0]
        return self.result

    def display(self):
        self.orig_image = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)

        # Display the image and draw the predicted boxes onto it.

        # Set the colors for the bounding boxes
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

        plt.figure(figsize=(20, 12))
        plt.imshow(self.orig_image)

        current_axis = plt.gca()

        for box in self.result:
            # Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
            xmin = box[-4] * self.orig_image.shape[1] / self.img_width
            ymin = box[-3] * self.orig_image.shape[0] / self.img_height
            xmax = box[-2] * self.orig_image.shape[1] / self.img_width
            ymax = box[-1] * self.orig_image.shape[0] / self.img_height
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(self.classes[int(box[0])], box[1])
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})

        plt.show()

    def list_roi(self):
        roi_list = []
        for r in self.result:
            className = self.classes[int(r[0])]
            confi = r[1]

            # Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
            xmin = int(r[-4] * self.orig_image.shape[1] / self.img_width)
            ymin = int(r[-3] * self.orig_image.shape[0] / self.img_height)
            xmax = int(r[-2] * self.orig_image.shape[1] / self.img_width)
            ymax = int(r[-1] * self.orig_image.shape[0] / self.img_height)

            roi = self.orig_image[ymin:ymax, xmin:xmax]
            roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            roi_list.append([className, confi, roi])

        return roi_list


if __name__ == '__main__':
    # init tensorflow
    inti_tensorflow()
    ssd = SSD('pre_trained/VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.h5', 512, 512)
    # ssd = SSD('pre_trained/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.h5', 300, 300)
    ssd.load_img('examples/cars/0322.jpg')
    result = ssd.perdict()
    ssd.display()
    print(result)
    roi_list = ssd.list_roi()
    print()
