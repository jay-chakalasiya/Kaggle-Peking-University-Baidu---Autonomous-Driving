import json
import numpy as np
import Visualize_3d_Models as vis
import matplotlib.pyplot as plt

def get_example_wheel(model = "019-SUV.json", start = 75, end = 95):
    model_dir = "C:/Users/kench/Desktop/PKU_Data/car_models_json/"
    path_to_json = model_dir + model
    with open(path_to_json) as json_file:
        data = json.load(json_file)
        vertices = np.array(data['vertices'])
        vertices_ = vertices[start:end, :]
    return vertices_

def find_wheels(path_to_json,e_wheel):
    with open(path_to_json) as json_file:
        data = json.load(json_file)
        vertices = np.array(data['vertices'])
        memory = np.zeros((19,3))
        memory_counter = 0
        candidate_sections = []
        flag = True
        movement_factor = .7
        example_wheel_shape = np.zeros((e_wheel.shape[0],e_wheel.shape[1]))
        for i in range(0,e_wheel.shape[0]):
            if i > 0:
                temp = e_wheel[i] - e_wheel[i-1]
                example_wheel_shape[i,:] = temp
        example_wheel_shape = example_wheel_shape[1:,:]
        emax_X, emax_Y, emax_Z, emin_X, emin_Y, emin_Z = find_vert_max_min(example_wheel_shape)

        for v in range(0,vertices.shape[0]):
            if flag == True:
                flag = False
            elif flag == False:
                advance = False
                temp = vertices[v] - vertices[v - 1]
                x_shift = abs(temp[0]- example_wheel_shape[memory_counter][0])
                y_shift = abs(temp[1]- example_wheel_shape[memory_counter][1])
                z_shift = abs(temp[2]- example_wheel_shape[memory_counter][2])
                x_agreement = temp[0]* example_wheel_shape[memory_counter][0]
                y_agreement = temp[1]* example_wheel_shape[memory_counter][1]
                z_agreement = temp[2]* example_wheel_shape[memory_counter][2]
#                if abs(x_shift) < movement_factor and abs(y_shift) < movement_factor and abs(z_shift) < movement_factor and y_agreement>0 and z_agreement>0:
                if abs(x_shift) < movement_factor and abs(y_shift) < movement_factor and abs(
                        z_shift) < movement_factor:
                    advance = True
                #if np.sum(abs(temp-example_wheel_shape[memory_counter])) < movement_factor:
                if advance == True:
                    memory[memory_counter,:] = vertices[v]
                    memory_counter += 1
                if memory_counter >= 18:
                    memory_counter = 0
                    max_X, max_Y, max_Z, min_X, min_Y, min_Z = find_vert_max_min(memory)
                    if abs(max_Y-min_Y)/abs(emax_Y-emin_Y) >.7:
                        candidate_sections.append(memory[:-1])
                if advance == False:
                    memory_counter = 0
                    if memory_counter >= 1:
                        max_X, max_Y, max_Z, min_X, min_Y, min_Z = find_vert_max_min(memory)
                        if abs(max_Y-min_Y)/abs(emax_Y-emin_Y) >.7:
                            candidate_sections.append(memory[:-1])
                    memory = np.zeros((19, 3))
        return candidate_sections


def find_wheels2(path_to_json, e_wheel):
    with open(path_to_json) as json_file:
        data = json.load(json_file)
        vertices = np.array(data['vertices'])
        memory_counter = 0
        candidate_sections = []
        flag = True
        movement_factor = .7
        example_wheel_shape = np.zeros((e_wheel.shape[0], e_wheel.shape[1]))
        for i in range(0, e_wheel.shape[0]):
            if i > 0:
                temp = e_wheel[i] - e_wheel[i - 1]
                example_wheel_shape[i, :] = temp
        example_wheel_shape = example_wheel_shape[1:, :]
        emax_X, emax_Y, emax_Z, emin_X, emin_Y, emin_Z = find_vert_max_min(example_wheel_shape)

        for v in range(0, vertices.shape[0]):
            memory = np.zeros((e_wheel.shape[0], 3))
            if flag == True:
                flag = False
            elif flag == False:
                end = v+200
                if v+200 > vertices.shape[0]:
                    end = vertices.shape[0]
                for j in range(v, end):
                    advance = False
                    temp = vertices[v] - vertices[j]
                    x_shift = abs(temp[0] - example_wheel_shape[memory_counter][0])
                    y_shift = abs(temp[1] - example_wheel_shape[memory_counter][1])
                    z_shift = abs(temp[2] - example_wheel_shape[memory_counter][2])
                    x_agreement = temp[0] * example_wheel_shape[memory_counter][0]
                    y_agreement = temp[1] * example_wheel_shape[memory_counter][1]
                    z_agreement = temp[2] * example_wheel_shape[memory_counter][2]
                    #                if abs(x_shift) < movement_factor and abs(y_shift) < movement_factor and abs(z_shift) < movement_factor and y_agreement>0 and z_agreement>0:
                    if abs(x_shift) < movement_factor and abs(y_shift) < movement_factor and abs(
                            z_shift) < movement_factor:
                        advance = True

                    # if np.sum(abs(temp-example_wheel_shape[memory_counter])) < movement_factor:
                    if advance == True:
                        memory[memory_counter, :] = vertices[j]
                        memory_counter += 1
                    if memory_counter >= e_wheel.shape[0]-2:
                        memory_counter = 0
                        max_X, max_Y, max_Z, min_X, min_Y, min_Z = find_vert_max_min(memory)
                        if abs(max_Y - min_Y) / abs(emax_Y - emin_Y) > .75:
                            candidate_sections.append(memory[:-2])
                        # memory = np.zeros((19, 3))
                    if advance == False:
                        memory_counter = 0
                        if memory_counter >= 1:
                            max_X, max_Y, max_Z, min_X, min_Y, min_Z = find_vert_max_min(memory)
                            if abs(max_Y - min_Y) / abs(emax_Y - emin_Y) > .7:
                                candidate_sections.append(memory[:-1])
                        memory = np.zeros((19, 3))
        return candidate_sections

def find_vert_max_min(vertices):
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



def get_wheels(model):
    model_dir = "C:/Users/kench/Desktop/PKU_Data/car_models_json/"
    path_to_json = model_dir + model
    e_wheel = get_example_wheel()
    sections = find_wheels2(path_to_json, e_wheel)

    if len(sections) < 3:
        sectionstemp = find_wheels(path_to_json, e_wheel)
        print("A",len(sectionstemp))
        sections = sections.extend(sectionstemp)
    fig = plt.figure()
    verts_X = []
    verts_Y = []
    verts_Z = []

    for i in sections:
        #vis.render_car_wheel_verticies(i)
        verts_X.extend(i[:,0])
        verts_Y.extend(i[:,1])
        verts_Z.extend(i[:,2])
    ax2 = fig.add_subplot(1, 1, 1, projection='3d')
    ax2.set_xlim([-3, 3])
    ax2.set_ylim([-3, 3])
    ax2.set_zlim([0, 3])
    ax2.scatter(verts_X,verts_Y, verts_Z, zdir="y")
    plt.show()
