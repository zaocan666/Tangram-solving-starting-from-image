import numpy as np
import cv2
import copy
from random import randint

VERBOSE=True

#被填充图像所对应的类，只实例化一次
class Graph():
    def __init__(self, pic_path):
        self.pic_path = pic_path
        self.init_graph = self.read_gray_pic(pic_path)
        self.current_graph = copy.deepcopy(self.init_graph)
        self.basic_length = self.get_basic_len()
        contour_nodes = get_contour_nodes(graph = self.init_graph, basic_length = self.basic_length)
        '''all_nodes = self.contour_nodes
        for i in range(6):
            all_nodes = self.get_inner_nodes(all_nodes)
        self.all_nodes = all_nodes'''

        self.contour_points = contour_nodes#define_neighbor(contour_nodes)

        if VERBOSE:
            print_nodes(pic_path=self.pic_path, nodes=self.contour_points, file_name="points.jpg")

        self.pieces_colors = []
        for i in range(7):
            self.pieces_colors.append((randint(100, 200), randint(100, 200), randint(100, 200)))

    def read_gray_pic(self, pic_path):
        img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
        return img

    #将被填充部分分成16个小直角三角形，basic_length是小直角三角形的直角边长
    def get_basic_len(self):
        m,n=self.init_graph.shape
        black_area = ((255-self.init_graph)//255).sum()
        lenghth = np.sqrt((float(black_area)/16.0)*2)
        return lenghth

    #得到当前顶点之外的点
    '''def get_inner_nodes(self, now_nodes):
        all_p = copy.deepcopy(now_nodes)
        inner_nodes=[]
        for c_p in now_nodes:
            possible_p=[]
            possible_p.append((c_p[0] + self.basic_length*np.sqrt(2), c_p[1]))
            possible_p.append((c_p[0] - self.basic_length * np.sqrt(2), c_p[1]))
            possible_p.append((c_p[0], c_p[1] + self.basic_length * np.sqrt(2)))
            possible_p.append((c_p[0], c_p[1] - self.basic_length * np.sqrt(2)))

            possible_p.append((c_p[0] + self.basic_length / np.sqrt(2), c_p[1] + self.basic_length / np.sqrt(2)))
            possible_p.append((c_p[0] + self.basic_length / np.sqrt(2), c_p[1] - self.basic_length / np.sqrt(2)))
            possible_p.append((c_p[0] - self.basic_length / np.sqrt(2), c_p[1] + self.basic_length / np.sqrt(2)))
            possible_p.append((c_p[0] - self.basic_length / np.sqrt(2), c_p[1] - self.basic_length / np.sqrt(2)))

            for p in possible_p:
                flag=False
                for exist_p in all_p:
                    if pic_process.tooClose(p, exist_p, threhold=self.basic_length/4):
                        flag=True
                        break
                if flag==False:
                    dis = 3
                    black=0
                    for i in range(int(p[0]) - dis, int(p[0]) + dis):
                        if i < 0 or i >= self.init_graph.shape[1]:
                            continue
                        for j in range(int(p[1]) - dis, int(p[1]) + dis):
                            if j < 0 or j >= self.init_graph.shape[0]:
                                continue
                            if self.init_graph[j][i]<125:
                               black += 1

                    if black >= dis:
                        int_p = (int(p[0]), int(p[1]))
                        all_p.append(int_p)

        return all_p'''


#图像中的特征点，包括边缘点和内部点，内部点是根据最小三角形的尺寸确定
class Pic_feature():
    def __init__(self, pos):
        self.pos = tuple(pos)
        #self.neighbor_point1 = neighbor_point1
        #self.neighbor_point2 = neighbor_point2


def define_neighbor(contour_nodes):
    nodes_all = []
    node_sum = 0
    for contour_group in contour_nodes:
        nodes = []
        for p in contour_group:
            node = Pic_feature(pos=p)
            nodes.append(node)

        nodes[0].neighbor_point1 = node_sum + 1
        nodes[0].neighbor_point2 = node_sum + len(nodes) - 1
        for i in range(1, len(nodes) - 1):
            nodes[i].neighbor_point1 = node_sum + i - 1
            nodes[i].neighbor_point2 = node_sum + i + 1
        nodes[-1].neighbor_point1 = node_sum + len(nodes) - 2
        nodes[-1].neighbor_point2 = node_sum + 0
        nodes_all += nodes

        node_sum += len(nodes)
    return nodes_all

def poly_out_move(ps, out_len):
    center_x = 0
    center_y = 0
    for x, y in ps:
        center_x += x
        center_y += y
    center_x /= len(ps)
    center_y /= len(ps)

    result = []
    for p in ps:
        arc = (p[0] - center_x , p[1] - center_y)
        lens = arc_len(arc)
        arc_x = arc[0]/lens
        arc_y = arc[1]/lens
        result.append((int(p[0]+arc_x*out_len), int(p[1]+arc_y*out_len)))
    return result

#三角形的重心
def get_middle_point_tri(p1, p2, p3):
    x = (p1[0] + p2[0] + p3[0])/3
    y = (p1[1] + p2[1] + p3[1])/3
    return (x, y)

#向量长度
def arc_len(arc):
    return np.sqrt(arc[0]**2+arc[1]**2)

#p1指向p2的向量的角度, [-180, 180]
def get_angle(p1, p2):
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    angle = np.arctan2(delta_y, delta_x) * 180 / np.pi
    return angle

def print_nodes(pic_path, nodes, file_name):
    if isinstance(pic_path, str):
        img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = pic_path
    for i, p in enumerate(nodes):
        if isinstance(p, tuple) or isinstance(p, list):
            p = p
        else:
            p=p.pos
        p = (int(p[0]), int(p[1]))
        cv2.circle(img, p, 5, 125, -1)
        cv2.putText(img, "N%d" % i, tuple(p), cv2.FONT_HERSHEY_PLAIN,
                    2.0, 125, thickness=2)

    cv2.imwrite(file_name, img)

def num_tooClose(n1, n2, threhold=8):
    if np.abs(n1-n2)<threhold:
        return True
    else:
        return False

def tooClose(p1, p2, threhold=8):
    dis = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    if dis<threhold:
        return True
    else:
        return False

def get_contour_nodes(graph, basic_length):
    ret, binary = cv2.threshold(graph, 127, 255, cv2.THRESH_BINARY)
    #_, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #边缘检测
    corners = cv2.goodFeaturesToTrack(graph, 22, 0.17, 7)#basic_length/8)
    corners=np.int0(corners)[:,0,:].tolist()

    contour_nodes = []
    for point in corners:
        contour_nodes.append(Pic_feature(point))

    return contour_nodes
    '''points = []
    draw_points = []
    for i in range(1, len(contours)):
        points.append(contours[i][:,0,:].tolist())
        draw_points += contours[i][:,0,:].tolist()
    valid_points=[]

    print_nodes(copy.deepcopy(graph), draw_points, "get_contour_contour.jpg")
    print_nodes(copy.deepcopy(graph), corners, "get_contour_corners.jpg")

    #去除冗余的点
    for group in points:
        valid_group = []
        for corner in group:
            for p in corners:
                if tooClose(p, corner, threhold=7):
                    valid_group.append(tuple(corner))
        if len(valid_group)>2:
            valid_points.append(valid_group)

    for valid_group in valid_points:
        index = 0
        for i in range(len(valid_group)-1):
            if tooClose(valid_group[index], valid_group[index+1], threhold=7):
                valid_group.pop(index+1)
            else:
                index += 1

    return valid_points'''