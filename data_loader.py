import os
import cv2
import csv
import numpy as np
import json
from car_models import *
from math import floor

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return json.JSONEncoder.default(self, obj)

def rotation_matrix(yaw, pitch, roll):
  cu, cv, cw = np.cos(float(roll)), np.cos(float(pitch)), np.cos(float(yaw))
  su, sv, sw = np.sin(float(roll)), np.sin(float(pitch)), np.sin(float(yaw))
  return np.array([[cv*cw, su*sv*cw-cu*sw, su*sw+cu*sv*cw],
                   [cv*sw, cu*cw+su*sv*sw, cu*sv*sw-su*cw],
                   [-1*sv, su*cv, cu*cv]])

def nothing(a,b):
  return b

class DataLoader():
  def __init__(self, img_dir, mask_dir, csv_path, json_file_path):
    self.img_dir = img_dir
    self.mask_dir = mask_dir
    self.csv_path = csv_path
    self.json_file_path = json_file_path

    # camera intrinsic
    self.fx = 2304.5479
    self.fy = 2305.8757
    self.cx = 1686.2379
    self.cy = 1354.9849

    self.json_data = {}
    # some example
    json_data_example = {
      "ID_0ab2b4c01": [
        {
          "model_type": 16,
          "yaw": 0.254839,
          "pitch": -2.57534,
          "roll": -3.10256,
          "world_frame": {
            "center": [7.96539, 3.20066, 11.0225] # x y z
          },
          "camera_frame": {
            "center": [1223,692] # x y
          }
        },
        {
          "data for car #2 in the image": 0
        }
      ]
    }

  def load_raw_data(self):
    with open(self.csv_path, mode='r') as csv_file:
      csv_reader = csv.reader(csv_file)
      csv_table = [r for r in csv_reader][1:]  # ignore the first line
      for row in csv_table:
        # first element i.e.row[0], is the image id
        img_id = row[0]
        self.json_data[img_id] = []
        cars_in_image = np.array(row[1].split()).reshape([-1, 7]).astype(np.float)
        for car_data_raw in cars_in_image:
          # the original data
          self.json_data[img_id].append({
            "model_type": car_data_raw[0], # car id number
            "yaw": car_data_raw[1],
            "pitch": car_data_raw[2],
            "roll": car_data_raw[3],
            "world_frame": {
              "center": car_data_raw[4:7],
            },
            "camera_frame": {
              "center": [-1, -1],
            }
          })

          # extra keypoints for current car
          car_name = Id_to_Car[int(car_data_raw[0])]
          car_dimension = car_dimension_data.loc[car_dimension_data['Model'] == car_name]
          
          # rotation matrix based on current car pose
          rot = rotation_matrix(car_data_raw[3], car_data_raw[2], car_data_raw[1]) # yaw, pitch, roll

          # Note: X-axis => front back, Y-axis => left right, Z-axis => top bottom
          # extra coords
          self.json_data[img_id][-1]["world_frame"]["front"] = car_data_raw[4:7] + np.dot(rot, np.array([float(car_dimension['X-max']), 0,0]))
          self.json_data[img_id][-1]["world_frame"]["rear"] = car_data_raw[4:7] + np.dot(rot, np.array([float(car_dimension['X-max']), 0,0]))
          self.json_data[img_id][-1]["world_frame"]["left"] = car_data_raw[4:7] + np.dot(rot, np.array([0, float(car_dimension['Y-max']), 0]))
          self.json_data[img_id][-1]["world_frame"]["right"] = car_data_raw[4:7] + np.dot(rot, np.array([0, float(car_dimension['Y-min']), 0]))
          self.json_data[img_id][-1]["world_frame"]["top"] = car_data_raw[4:7] + np.dot(rot, np.array([0, 0, float(car_dimension['Z-max'])]))
          self.json_data[img_id][-1]["world_frame"]["bottom"] = car_data_raw[4:7] + np.dot(rot, np.array([0, 0, float(car_dimension['Z-min'])]))
          
          # bounding cube for the car
          self.json_data[img_id][-1]["world_frame"]["front_left_top"] = car_data_raw[4:7] + np.dot(rot, np.array([float(car_dimension['X-max']), float(car_dimension['Y-max']), float(car_dimension['Z-max'])]))
          self.json_data[img_id][-1]["world_frame"]["front_left_bottom"] = car_data_raw[4:7] + np.dot(rot, np.array([float(car_dimension['X-max']), float(car_dimension['Y-max']), float(car_dimension['Z-min'])]))
          self.json_data[img_id][-1]["world_frame"]["front_right_top"] = car_data_raw[4:7] + np.dot(rot, np.array([float(car_dimension['X-max']), float(car_dimension['Y-min']), float(car_dimension['Z-max'])]))
          self.json_data[img_id][-1]["world_frame"]["front_right_bottom"] = car_data_raw[4:7] + np.dot(rot, np.array([float(car_dimension['X-max']), float(car_dimension['Y-min']), float(car_dimension['Z-min'])]))
          self.json_data[img_id][-1]["world_frame"]["rear_left_top"] = car_data_raw[4:7] + np.dot(rot, np.array([float(car_dimension['X-min']), float(car_dimension['Y-max']), float(car_dimension['Z-max'])]))
          self.json_data[img_id][-1]["world_frame"]["rear_left_bottom"] = car_data_raw[4:7] + np.dot(rot, np.array([float(car_dimension['X-min']), float(car_dimension['Y-max']), float(car_dimension['Z-min'])]))
          self.json_data[img_id][-1]["world_frame"]["rear_right_top"] = car_data_raw[4:7] + np.dot(rot, np.array([float(car_dimension['X-min']), float(car_dimension['Y-min']), float(car_dimension['Z-max'])]))
          self.json_data[img_id][-1]["world_frame"]["rear_right_bottom"] = car_data_raw[4:7] + np.dot(rot, np.array([float(car_dimension['X-min']), float(car_dimension['Y-min']), float(car_dimension['Z-min'])]))

          # put above into camera frame
          for keypoint_id, val in self.json_data[img_id][-1]["world_frame"].items():
            self.json_data[img_id][-1]["camera_frame"][keypoint_id] = self.coord_world_to_cam(val[0], val[1], val[2])
    
    # dump data
    print("Parsed all training data")
    with open(self.json_file_path, 'w') as f:
        json.dump(self.json_data, f, cls=NumpyEncoder, indent=2)

    print("Data dumped to {}".format(self.json_file_path))


  def load_json_data(self):
    with open(self.json_file_path) as json_file:
      self.json_data = json.load(json_file)
      print("loaded json from {}".format(self.json_file_path))
    

  def coord_world_to_cam(self, x, y, z):
    # input: coordinate from world frame
    # output: x, y in camera frame
    return [x * self.fx / z + self.cx, y * self.fy / z + self.cy]

  def visualize(self, img_id):
    # get data
    print("reading", self.img_dir + img_id + ".jpg")
    img = cv2.imread(self.img_dir + img_id + ".jpg")
    print("img.shape", img.shape)

    for car in self.json_data[img_id]:
      keypoints = car["camera_frame"]
      img = draw_cube(img, keypoints)
    cv2.imwrite("bounding_box_demo.jpg", img) 
    cv2.imshow("img",cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4))))
    cv2.waitKey(0)

def list_to_cv_pixel_coord(lst):
  return (floor(lst[0]), floor(lst[1]))

def draw_cube(img, keypoints):

  img = cv2.circle(img,list_to_cv_pixel_coord(keypoints["center"]), 10, (0,0,254), -1)

  lineThickness = 2
  cv2.line(img, list_to_cv_pixel_coord(keypoints["front_left_top"]), list_to_cv_pixel_coord(keypoints["front_left_bottom"]), (0,255,0), lineThickness)
  cv2.line(img, list_to_cv_pixel_coord(keypoints["front_left_bottom"]), list_to_cv_pixel_coord(keypoints["front_right_bottom"]), (0,255,0), lineThickness)
  cv2.line(img, list_to_cv_pixel_coord(keypoints["front_right_bottom"]), list_to_cv_pixel_coord(keypoints["front_right_top"]), (0,255,0), lineThickness)
  cv2.line(img, list_to_cv_pixel_coord(keypoints["front_right_top"]), list_to_cv_pixel_coord(keypoints["front_left_top"]), (0,255,0), lineThickness)

  cv2.line(img, list_to_cv_pixel_coord(keypoints["rear_left_top"]), list_to_cv_pixel_coord(keypoints["rear_left_bottom"]), (0,255,0), lineThickness)
  cv2.line(img, list_to_cv_pixel_coord(keypoints["rear_left_bottom"]), list_to_cv_pixel_coord(keypoints["rear_right_bottom"]), (0,255,0), lineThickness)
  cv2.line(img, list_to_cv_pixel_coord(keypoints["rear_right_bottom"]), list_to_cv_pixel_coord(keypoints["rear_right_top"]), (0,255,0), lineThickness)
  cv2.line(img, list_to_cv_pixel_coord(keypoints["rear_right_top"]), list_to_cv_pixel_coord(keypoints["rear_left_top"]), (0,255,0), lineThickness)

  cv2.line(img, list_to_cv_pixel_coord(keypoints["front_left_top"]), list_to_cv_pixel_coord(keypoints["rear_left_top"]), (0,255,0), lineThickness)
  cv2.line(img, list_to_cv_pixel_coord(keypoints["front_left_bottom"]), list_to_cv_pixel_coord(keypoints["rear_left_bottom"]), (0,255,0), lineThickness)
  cv2.line(img, list_to_cv_pixel_coord(keypoints["front_right_bottom"]), list_to_cv_pixel_coord(keypoints["rear_right_bottom"]), (0,255,0), lineThickness)
  cv2.line(img, list_to_cv_pixel_coord(keypoints["front_right_top"]), list_to_cv_pixel_coord(keypoints["rear_right_top"]), (0,255,0), lineThickness)

  return img



if __name__ == "__main__":
  train_pic_dir = "C:\\Users\\wenbo\\Downloads\\pku-autonomous-driving\\train_images\\"
  train_mask_dir = "C:\\Users\\wenbo\\Downloads\\pku-autonomous-driving\\train_masks\\"
  csv_path = "C:\\Users\\wenbo\\Downloads\\pku-autonomous-driving\\train.csv"
  json_file_path = "C:\\Users\\wenbo\\workspace\\Kaggle-Peking-University-Baidu---Autonomous-Driving\\training_data.json"

  data_loader = DataLoader(train_pic_dir, train_mask_dir, csv_path, json_file_path)
  data_loader.load_raw_data()
  # data_loader.load_json_data()


  data_loader.visualize("ID_a381bf4d0")
