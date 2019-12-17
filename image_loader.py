import os
import cv2
import csv
import numpy as np
import copy

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
    self.camera_matrix = np.array([[self.fx, 0,  self.cx],
                                    [0, self.fy, self.cy],
                                    [0, 0, 1]], dtype=np.float32)
    
    self.prediction_data_world_frame = {} # x, y, z in world frame
    self.prediction_data_cam_frame = {} # x, y in camera frame
    with open(csv_path, mode='r') as csv_file:
      csv_reader = csv.reader(csv_file)
      csv_table = [r for r in csv_reader][1:] # ignore the first line
      for row in csv_table:
        img_id = row[0]
        pred_world_frame = np.array(row[1].split()).reshape([-1, 7]).astype(np.float) # array of [model type, yaw, pitch, roll, x, y, z]
        self.prediction_data_world_frame[img_id] = pred_world_frame
        
        pred_cam_frame = []
        for car_data in pred_world_frame:
          x,y,z = car_data[4],car_data[5],car_data[6]
          pred_cam_frame_car = np.concatenate((car_data[0:4], [x * self.fx / z + self.cx, y * self.fy / z + self.cy]), axis=None)
          pred_cam_frame.append(pred_cam_frame_car)
        self.prediction_data_cam_frame[img_id] = pred_cam_frame
    
  def visualize(self, img_id):
    print("reading", self.img_dir + img_id + ".jpg")
    img = cv2.imread(self.img_dir + img_id + ".jpg")
    print("img.shape", img.shape)

    print("self.prediction_data_cam_frame[img_id]", self.prediction_data_cam_frame[img_id])

    for car in self.prediction_data_cam_frame[img_id]:
      print(int(car[4]),int(car[5]))
      img = cv2.circle(img,(int(car[4]),int(car[5])), 32, (0,255,0), -1)
    cv2.imshow("img",cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4))))
    cv2.waitKey(0)


if __name__ == "__main__":
  train_pic_dir = "C:\\Users\\wenbo\\Downloads\\pku-autonomous-driving\\train_images\\"
  train_mask_dir = "C:\\Users\\wenbo\\Downloads\\pku-autonomous-driving\\train_masks\\"
  csv_path = "C:\\Users\\wenbo\\Downloads\\pku-autonomous-driving\\train.csv"
  image_loader = ImageLoader(train_pic_dir, train_mask_dir, csv_path)

  image_loader.visualize("ID_0ab2b4c01")
