from __future__ import print_function
from __future__ import division
import cv2 as cv
import argparse
import copy
import csv
import numpy as np

cars = []
class ImageLoader():
    def __init__(self, img_dir, mask_dir, csv_path):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.csv_path = csv_path

        # camera intrinsic
        self.fx = 2304.5479
        self.fy = 2305.8757
        self.cx = 1686.2379
        self.cy = 1354.9849
        self.camera_matrix = np.array([[self.fx, 0, self.cx],
                                       [0, self.fy, self.cy],
                                       [0, 0, 1]], dtype=np.float32)

        self.prediction_data_world_frame = {}  # x, y, z in world frame
        self.prediction_data_cam_frame = {}  # x, y in camera frame
        with open(csv_path, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            csv_table = [r for r in csv_reader][1:]  # ignore the first line
            for row in csv_table:
                img_id = row[0]
                pred_world_frame = np.array(row[1].split()).reshape([-1, 7]).astype(
                    np.float)  # array of [model type, yaw, pitch, roll, x, y, z]
                self.prediction_data_world_frame[img_id] = pred_world_frame

                pred_cam_frame = []
                for car_data in pred_world_frame:
                    x, y, z = car_data[4], car_data[5], car_data[6]
                    pred_cam_frame_car = np.concatenate(
                        (car_data[0:4], [x * self.fx / z + self.cx, y * self.fy / z + self.cy],car_data[4:7]), axis=None)
                    pred_cam_frame.append(pred_cam_frame_car)
                self.prediction_data_cam_frame[img_id] = pred_cam_frame

    def visualize(self, img_id):
        print("reading", self.img_dir + img_id + ".jpg")
        img = cv.imread(self.img_dir + img_id + ".jpg")
        print("img.shape", img.shape)

        print("self.prediction_data_cam_frame[img_id]", self.prediction_data_cam_frame[img_id])
        fontScale = 2
        for car in self.prediction_data_cam_frame[img_id]:
            cars.append(car)
            img = cv.circle(img, (int(car[4])-50, int(car[5])), 10, (0, 255, 0), -1)
        #cv.imshow("img", cv.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4))))
        #cv.waitKey(0)
        return img

train_pic_dir = "C:/Users/kench/Desktop/PKU_Data/train_images/"
train_mask_dir = "C:/Users/kench/Desktop/PKU_Data/train_masks/"
csv_path = "C:/Users/kench/Desktop/PKU_Data/train.csv"
img_id = "ID_0db53ef6d"


image_loader = ImageLoader(train_pic_dir, train_mask_dir, csv_path)
image_ = image_loader.visualize(img_id)

fontScale = 2
color = (255, 200, 100)
thickness = 3
font = cv.FONT_HERSHEY_SIMPLEX
alpha_slider_max = 100

title_window = 'Compare_Photos'

def on_trackbar(val):
    car = cars[val]
    text = "1) "+ str(round(car[1], 2)) + "//" + str(round(car[2], 2)) + "//" + str(round(car[3], 2))+"||"+str(int(car[6]))+"//"+str(int(car[7]))+"//"+str(int(car[8]))
    car2 = cv.getTrackbarPos('Car2', title_window)
    car2 = cars[car2]
    text2 = "2) "+str(round(car2[1], 2)) + "//" + str(round(car2[2], 2)) + "//" + str(round(car2[3], 2))+"||"+str(int(car2[6]))+"//"+str(int(car2[7]))+"//"+str(int(car2[8]))
    color = cv.getTrackbarPos('Color', title_window)
    color = (color,color,color)
    image_ = image_loader.visualize(img_id)
    img = cv.rectangle(image_, (int(car2[4]- 50), int(car2[5])),(int(car2[4]- 50), int(car2[5])),color, 25)
    img = cv.putText(img, text2, (100,200), font,
                      fontScale, color, thickness, cv.LINE_AA)


    img = cv.circle(image_, (int(car[4]) - 50, int(car[5])), 32, color, -1)
    img = cv.putText(img, text, (100,100), font,
                      fontScale, color, thickness, cv.LINE_AA)
    cv.imshow(title_window, img)
def on_trackbar2(val):
    color = (val, val, val)
def on_trackbar3(val):
    val = cv.getTrackbarPos('Car', title_window)
    on_trackbar(val)

cv.namedWindow(title_window,200)
trackbar_name = "Car"
trackbar2_name = "Car2"
trackbar3_name = "Color"

cv.createTrackbar(trackbar_name, title_window , 0, len(cars), on_trackbar)
cv.createTrackbar(trackbar2_name, title_window , 0, len(cars), on_trackbar3)
cv.createTrackbar(trackbar3_name, title_window , 0, 255, on_trackbar2)

# Show some stuff
on_trackbar(0)
# Wait until user press some key
cv.waitKey()