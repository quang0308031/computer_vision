import glob as glob
import os
import tensorflow as tf
import cv2 as cv
from ultralytics import YOLO

def crop_and_save(model_path, conf, data_path, det, img_size):
    model = YOLO(model_path)
    for k in glob.glob(data_path + '/*'):
        for t in glob.glob(k + '/*'):
            pre = model.predict(t, conf = conf)
            for i in pre:
                path = str.split(i.path, '/')
                link = det + '/' + path[-3] + '/' + path[-2]
                os.makedirs(link, exist_ok=True)

                for j in i.boxes:
                    tensor_shape = tf.convert_to_tensor(j.orig_shape)
                    thresh_hold = [tensor_shape / 2 + tensor_shape / 4, tensor_shape / 2 - tensor_shape / 4]
                    for k in j.xyxy:
                        list_xy = [int(k[o]) for o in range(len(k))]
                        if (list_xy[0] + list_xy[-2]) / 2 < thresh_hold[0][1] and (list_xy[0] + list_xy[-2]) / 2 > thresh_hold[1][1] and (list_xy[1] + list_xy[-1]) / 2 < thresh_hold[0][0] and (list_xy[1] + list_xy[-1]) / 2 > thresh_hold[1][0]:
                            cv.imwrite(link + '/' + path[-1], cv.resize(cv.imread(i.path, 1)[list_xy[1]:list_xy[-1], list_xy[0]:list_xy[-2]], (img_size[0], img_size[1])))