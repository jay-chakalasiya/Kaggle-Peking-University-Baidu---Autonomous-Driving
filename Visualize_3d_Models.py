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
import Find_Wheels
from matplotlib.widgets import Slider, Button, RadioButtons



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

def render_car_model(path_to_json, start_index = 75, end_index = 95):
    with open(path_to_json) as json_file:
        data = json.load(json_file)
        fig = plt.figure()
        vertices = np.array(data['vertices'])
        (max_X, max_Y, max_Z, min_X, min_Y, min_Z) = build_wheel_bounding_boxes(75, 90, vertices)
        max_X, max_Y, max_Z, min_X, min_Y, min_Z = (max_X, max_Y, max_Z, min_X, min_Y, min_Z)
        new_verts = np.zeros((4,3))
        new_verts[0,:] = np.array([max_X,max_Y,min_Z])
        new_verts[1,:] = np.array([min_X,min_Y,min_Z])
        new_verts[2,:] = np.array([max_X,max_Y,max_Z])
        new_verts[3,:] = np.array([min_X,min_Y,max_Z])
        ax = fig.add_subplot(2, 1, 1, projection='3d')
        ax.set_title('car_type: '+data['car_type'])
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([0, 3])
        ax.scatter(vertices[:,0], vertices[:,1]*-1, vertices[:,2],zdir="y")

        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        vertices_ = vertices[::10, :]
        vertices_ = vertices[start_index:end_index, :]
        ax2.set_title('car_type: '+data['car_type'])
        ax2.set_xlim([-3, 3])
        ax2.set_ylim([-3, 3])
        ax2.set_zlim([0, 3])
        vertices_ = np.concatenate((vertices_,new_verts), axis=0)
        c = ["b" if y >= vertices_.shape[0]-4 else "r" for y in range(0,vertices_.shape[0])]
        ax2.scatter(vertices_[:,0], vertices_[:,1]*-1, vertices_[:,2],zdir="y",c = c)
        plt.show()

def render_car_wheel(path_to_json, start_index,end_index):
    with open(path_to_json) as json_file:
        data = json.load(json_file)
        vertices = np.array(data['vertices'])
        fig = plt.figure()
        (max_X, max_Y, max_Z, min_X, min_Y, min_Z) = build_wheel_bounding_boxes(start_index, end_index, vertices)
        max_X, max_Y, max_Z, min_X, min_Y, min_Z = (max_X, max_Y, max_Z, min_X, min_Y, min_Z)
        new_verts = np.zeros((5,3))
        new_verts[0,:] = np.array([max_X,max_Y,min_Z])
        new_verts[1,:] = np.array([max_X,min_Y,min_Z])
        new_verts[2,:] = np.array([max_X,max_Y,max_Z])
        new_verts[3,:] = np.array([max_X,min_Y,max_Z])
        new_verts[4,:] = np.array([max_X,(max_Y-(max_Y-min_Y)/2),(max_Z-(max_Z-min_Z)/2)])

        ax2 = fig.add_subplot(1, 1, 1, projection='3d')
        vertices_ = vertices[::10, :]
        vertices_ = vertices[start_index:end_index, :]
        ax2.set_title('car_type: '+data['car_type'])
        ax2.set_xlim([-3, 3])
        ax2.set_ylim([-3, 3])
        ax2.set_zlim([0, 3])
        vertices_ = np.concatenate((vertices_,new_verts), axis=0)
        c = ["b" if y >= vertices_.shape[0]-5 else "r" for y in range(0,vertices_.shape[0])]
        ax2.scatter(vertices_[:,0], vertices_[:,1]*-1, vertices_[:,2],zdir="y",c = c)
        plt.show()

def render_car_wheel_verticies(vertices_):
        fig = plt.figure()
        ax2 = fig.add_subplot(1, 1, 1, projection='3d')
        ax2.set_xlim([-3, 3])
        ax2.set_ylim([-3, 3])
        ax2.set_zlim([0, 3])
        ax2.scatter(vertices_[:,0], vertices_[:,1]*-1, vertices_[:,2],zdir="y")
        plt.show()

def build_wheel_bounding_boxes(start_index,end_index, vertices):
    vertices = vertices[start_index:end_index,:]
    X = vertices[:,0]
    Y = vertices[:,1]
    Z = vertices[:,2]
    max_X = np.max(X)
    max_Y = np.max(Y)
    max_Z = np.max(Z)
    min_X = np.min(X)
    min_Y = np.min(Y)
    min_Z = np.min(Z)
    return (max_X,max_Y,max_Z,min_X,min_Y,min_Z)



if __name__ == "__main__":
    DATASET_DIR = 'C:/Users/kench/Desktop/PKU_Data/'
    JSON_DIR = os.path.join(DATASET_DIR, 'car_models_json')
    NUM_IMG_SAMPLES = 10
    df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
    model_dir = "C:/Users/kench/Desktop/PKU_Data/car_models_json/"
    #model = "019-SUV.json"
    #render_car_wheel(path_to_json,75,95)

    #model = "036-CAR01.json"
    #model = "037-CAR02.json"
    #model = "changan-cs5.json"
    model = "changcheng-H6-2016.json"
    model = "dazhong-SUV.json"
    path_to_json = model_dir + model
    #render_car_model(path_to_json, start_index = 700, end_index=1000)
    Find_Wheels.get_wheels(model)
    #render_car_wheel(path_to_json)
