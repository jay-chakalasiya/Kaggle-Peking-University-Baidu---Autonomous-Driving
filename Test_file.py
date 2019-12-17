import numpy as np
import pandas as pd
import cv2
import json
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2
import csv
import numpy as np
import copy

DATASET_DIR = 'C:/Users/kench/Desktop/PKU_Data/'
JSON_DIR = os.path.join(DATASET_DIR, 'car_models_json')
NUM_IMG_SAMPLES = 10

df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))

models = {
    #           name                id
         'baojun-310-2017':          0,
            'biaozhi-3008':          1,
      'biaozhi-liangxiang':          2,
       'bieke-yinglang-XT':          3,
            'biyadi-2x-F0':          4,
           'changanbenben':          5,
            'dongfeng-DS5':          6,
                 'feiyate':          7,
     'fengtian-liangxiang':          8,
            'fengtian-MPV':          9,
       'jilixiongmao-2015':         10,
       'lingmu-aotuo-2009':         11,
            'lingmu-swift':         12,
         'lingmu-SX4-2012':         13,
          'sikeda-jingrui':         14,
    'fengtian-weichi-2006':         15,
               '037-CAR02':         16,
                 'aodi-a6':         17,
               'baoma-330':         18,
               'baoma-530':         19,
        'baoshijie-paoche':         20,
         'bentian-fengfan':         21,
             'biaozhi-408':         22,
             'biaozhi-508':         23,
            'bieke-kaiyue':         24,
                    'fute':         25,
                 'haima-3':         26,
           'kaidilake-CTS':         27,
               'leikesasi':         28,
           'mazida-6-2015':         29,
              'MG-GT-2015':         30,
                   'oubao':         31,
                    'qiya':         32,
             'rongwei-750':         33,
              'supai-2016':         34,
         'xiandai-suonata':         35,
        'yiqi-benteng-b50':         36,
                   'bieke':         37,
               'biyadi-F3':         38,
              'biyadi-qin':         39,
                 'dazhong':         40,
          'dazhongmaiteng':         41,
                'dihao-EV':         42,
  'dongfeng-xuetielong-C6':         43,
 'dongnan-V3-lingyue-2011':         44,
'dongfeng-yulong-naruijie':         45,
                 '019-SUV':         46,
               '036-CAR01':         47,
             'aodi-Q7-SUV':         48,
              'baojun-510':         49,
                'baoma-X5':         50,
         'baoshijie-kayan':         51,
         'beiqi-huansu-H3':         52,
          'benchi-GLK-300':         53,
            'benchi-ML500':         54,
     'fengtian-puladuo-06':         55,
        'fengtian-SUV-gai':         56,
'guangqi-chuanqi-GS4-2015':         57,
    'jianghuai-ruifeng-S3':         58,
              'jili-boyue':         59,
                  'jipu-3':         60,
              'linken-SUV':         61,
               'lufeng-X8':         62,
             'qirui-ruihu':         63,
             'rongwei-RX5':         64,
         'sanling-oulande':         65,
              'sikeda-SUV':         66,
        'Skoda_Fabia-2011':         67,
        'xiandai-i25-2016':         68,
        'yingfeinidi-qx80':         69,
         'yingfeinidi-SUV':         70,
              'benchi-SUR':         71,
             'biyadi-tang':         72,
       'changan-CS35-2012':         73,
             'changan-cs5':         74,
      'changcheng-H6-2016':         75,
             'dazhong-SUV':         76,
 'dongfeng-fengguang-S560':         77,
   'dongfeng-fengxing-SX6':         78
}

models_map = dict((y, x) for x, y in models.items())

def get_no_predicted_cars():
    cars = []
    image_ids = np.array(df['ImageId'])
    prediction_strings = np.array(df['PredictionString'])
    prediction_strings = [
        np.array(prediction_string.split(' ')).astype(np.float32).reshape(-1, 7) \
        for prediction_string in prediction_strings
    ]

    for prediction_string in prediction_strings:
        for car in prediction_string:
            cars.append(car)
    cars = np.array(cars)

    unique, counts = np.unique(cars[..., 0].astype(np.uint8), return_counts=True)
    all_model_types = zip(unique, counts)

    for i, model_type in enumerate(all_model_types):
        print('{}.\t Model type: {:<22} | {} cars'.format(i, models_map[model_type[0]], model_type[1]))

def render_car_model(path_to_json):
    with open(path_to_json) as json_file:
        data = json.load(json_file)
        vertices = np.array(data['vertices'])
        fig = plt.figure()

        ax = fig.add_subplot(2, 1, 1, projection='3d')
        ax.set_title('car_type: '+data['car_type'])
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([0, 3])
        ax.scatter(vertices[:,0], vertices[:,1]*-1, vertices[:,2],zdir="y")

        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        vertices_ = vertices[::10, :]
        vertices_ = vertices[75:95, :]

        ax2.set_title('car_type: '+data['car_type'])
        ax2.set_xlim([-3, 3])
        ax2.set_ylim([-3, 3])
        ax2.set_zlim([0, 3])
        ax2.scatter(vertices_[:,0], vertices_[:,1]*-1, vertices_[:,2],zdir="y")
        plt.show()

#model_dir = "C:/Users/kench/Desktop/PKU_Data/car_models_json/"
#model = "019-SUV.json"
#model = "036-CAR01.json"
#path_to_json = model_dir + model
#render_car_model(path_to_json)



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
                        (car_data[0:4], [x * self.fx / z + self.cx, y * self.fy / z + self.cy]), axis=None)
                    pred_cam_frame.append(pred_cam_frame_car)
                self.prediction_data_cam_frame[img_id] = pred_cam_frame

    def visualize(self, img_id):
        print("reading", self.img_dir + img_id + ".jpg")
        img = cv2.imread(self.img_dir + img_id + ".jpg")
        print("img.shape", img.shape)

        print("self.prediction_data_cam_frame[img_id]", self.prediction_data_cam_frame[img_id])
        fontScale = 2
        color = (255, 200, 100)
        thickness = 3
        font = cv2.FONT_HERSHEY_SIMPLEX
        for car in self.prediction_data_cam_frame[img_id][:5][::1]:
            text = str(round(car[1],2))+"//"+str(round(car[2],2))+"//"+str(round(car[3],2))
            img = cv2.circle(img, (int(car[4])-50, int(car[5])), 32, (0, 255, 0), -1)
            img = cv2.putText(img, text, (int(car[4]), int(car[5])), font,
                                fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow("img", cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4))))
        cv2.waitKey(0)

train_pic_dir = "C:/Users/kench/Desktop/PKU_Data/train_images/"
train_mask_dir = "C:/Users/kench/Desktop/PKU_Data/train_masks/"
csv_path = "C:/Users/kench/Desktop/PKU_Data/train.csv"
image_loader = ImageLoader(train_pic_dir, train_mask_dir, csv_path)

image_loader.visualize("ID_0db53ef6d")