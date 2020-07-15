import numpy as np
import cv2
import copy
from random import randint

VERBOSE=False

#被填充图像所对应的类，只实例化一次
class Graph():
    def __init__(self, pic_path, small_tri_num):
        self.pic_path = pic_path
        if type(pic_path) == str:
            self.init_color_img = cv2.imread(pic_path, cv2.COLOR_BGRA2BGR)
            self.init_graph = self.read_gray_pic(pic_path)
        else:
            self.init_color_img = copy.deepcopy(pic_path)
            self.init_graph = cv2.cvtColor(pic_path, cv2.COLOR_BGRA2GRAY)

        #cv2.imwrite("graph_finish.jpg", self.init_color_img)
        self.current_graph = copy.deepcopy(self.init_graph)
        self.basic_length = self.get_basic_len(small_tri_num)

        contour_nodes = get_contour_nodes(graph = self.init_graph, basic_length = self.basic_length)
        #contour_nodes = eliminate_nodes_tooClose(contour_nodes[0], self.basic_length*4/5)
        contour_nodes_c = []
        for contour_group in contour_nodes:
            for node in contour_group:
                contour_nodes_c.append(Pic_feature(pos=tuple(node)))
        all_nodes = contour_nodes_c

        all_len = 0
        while all_len != len(all_nodes): #找出所有内部节点
            all_len = len(all_nodes)
            all_nodes = self.get_inner_nodes(all_nodes)


        flag_90_45 = self.check_inner_nodes(all_nodes) #检查斜边横向的分割方法是否合适，如果不合适，换用斜边斜向的方法分割
        if flag_90_45 == False:
            all_nodes = contour_nodes_c
            all_len = 0
            while all_len != len(all_nodes):
                all_len=len(all_nodes)
                all_nodes = self.get_inner_nodes(all_nodes, k_90=1, k_45=1)
        self.all_nodes = all_nodes

        self.all_tri = self.get_all_tri(flag_90_45) #找出所有基本三角形
        if VERBOSE:
            print_nodes(pic_path=self.pic_path, nodes=self.all_nodes, file_name="points.jpg")
            if isinstance(self.pic_path, str):
                draw_tri_num(self.all_nodes, self.all_tri, self.pic_path, "tris.jpg")

        self.pieces_colors = [] #随机生成板块颜色

        for i in range(100):
            self.pieces_colors.append((randint(100, 200), randint(100, 200), randint(100, 200)))

    def check_inner_nodes(self, all_nodes):
        for i in range(len(all_nodes)):
            for j in range(i+1, len(all_nodes)):
                node_i = all_nodes[i].pos
                node_j = all_nodes[j].pos
                dis = arc_len((node_i[0]-node_j[0], node_i[1]-node_j[1]))
                if dis < self.basic_length/2:   #点与点之间的距离不应该小于basic length
                    return False
        return True

    def get_all_tri(self, flag_90_45):
        all_tri = []
        for node_i, node in enumerate(self.all_nodes):
            neighbors = get_point_neighbors(node, self.all_nodes)

            if flag_90_45:
                list_90 = [Pic_feature.n_left_up, Pic_feature.n_right_up, Pic_feature.n_right_down, Pic_feature.n_left_down, Pic_feature.n_left_up]
            else:
                list_90 = [Pic_feature.n_left, Pic_feature.n_up, Pic_feature.n_right,Pic_feature.n_down, Pic_feature.n_left]
            for i in range(len(list_90)-1):
                neighbor_1 = neighbors[list_90[i]]
                neighbor_2 = neighbors[list_90[i+1]]
                neighbor_1_num = node.n_neighbors[list_90[i]]
                neighbor_2_num = node.n_neighbors[list_90[i+1]]
                if neighbor_1 == None or neighbor_2 ==None:
                    continue
                neighbor_two_pair = [list_90[i], list_90[i+1]]

                middle_point = (int((neighbor_1.pos[0]+neighbor_2.pos[0]+node.pos[0])/3), int((neighbor_1.pos[1]+neighbor_2.pos[1]+node.pos[1])/3))
                middle_point_1 = (int((neighbor_1.pos[0]+middle_point[0]+node.pos[0])/3), int((neighbor_1.pos[1]+middle_point[1]+node.pos[1])/3))
                middle_point_2 = (int((neighbor_2.pos[0] + middle_point[0] + node.pos[0]) / 3),int((neighbor_2.pos[1] + middle_point[1] + node.pos[1]) / 3))
                if self.init_graph[middle_point_1[1]][middle_point_1[0]]>125: #三角形在图形外面
                    continue
                if self.init_graph[middle_point_2[1]][middle_point_2[0]]>125: #三角形在图形外面
                    continue

                tri = Small_tri(point_right=node_i, point_45_1=neighbor_1_num, point_45_2=neighbor_2_num)
                exit_flag = -1
                for neighbor_i in neighbor_two_pair:
                    for tri_other in neighbors[neighbor_i].tri_45:
                        if tri_other == -1:
                            continue
                        elif tri == all_tri[tri_other]:
                            exit_flag=tri_other
                            break
                        elif tri == all_tri[tri_other]:
                            exit_flag=tri_other
                            break
                if exit_flag != -1:
                    self.all_nodes[node_i].tri_90[list_90[i] - list_90[0] + Pic_feature.tri_up_90] = exit_flag
                else:
                    all_tri.append(tri)
                    self.all_nodes[node_i].tri_90[list_90[i] - list_90[0] + Pic_feature.tri_up_90] = len(all_tri) - 1
                    self.all_nodes[neighbor_1_num].tri_45[i * 2] = len(all_tri)-1
                    self.all_nodes[neighbor_2_num].tri_45[i * 2 + 1] = len(all_tri) - 1

        return all_tri


    def read_gray_pic(self, pic_path):
        img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
        return img

    #将被填充部分分成16个小直角三角形，basic_length是小直角三角形的直角边长
    def get_basic_len(self, small_tri_num):
        black_area = ((255-self.init_graph)//255).sum()
        lenghth = np.sqrt((float(black_area)/small_tri_num)*2)
        return lenghth

    #得到当前顶点之外的点
    def get_inner_nodes(self, now_nodes, k_90=np.sqrt(2), k_45=1/np.sqrt(2)):
        all_p = copy.deepcopy(now_nodes)
        for c_i, c_p in enumerate(now_nodes):
            possible_p=[]

            possible_p.append((c_p.pos[0] - self.basic_length * k_90, c_p.pos[1]))
            possible_p.append((c_p.pos[0], c_p.pos[1] - self.basic_length * k_90))
            possible_p.append((c_p.pos[0] + self.basic_length * k_90, c_p.pos[1]))
            possible_p.append((c_p.pos[0], c_p.pos[1] + self.basic_length * k_90))

            possible_p.append((c_p.pos[0] - self.basic_length *k_45, c_p.pos[1] - self.basic_length *k_45))
            possible_p.append((c_p.pos[0] + self.basic_length *k_45, c_p.pos[1] - self.basic_length *k_45))
            possible_p.append((c_p.pos[0] + self.basic_length *k_45, c_p.pos[1] + self.basic_length *k_45))
            possible_p.append((c_p.pos[0] - self.basic_length *k_45, c_p.pos[1] + self.basic_length *k_45))


            for p_i, p in enumerate(possible_p):
                flag=False
                for e_i, exist_p in enumerate(all_p):
                    if tooClose(p, exist_p.pos, threhold=self.basic_length/6):
                        all_p[c_i].n_neighbors[p_i] = e_i
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
                        int_p = Pic_feature(pos=tuple(int_p))
                        all_p.append(int_p)

        return all_p


#图形节点，包括边缘点和内部点，内部点是根据最小三角形的尺寸确定
class Pic_feature():
    n_left = 0 #self.n_neighbors列表中0号元素表示左邻居节点，下同
    n_up = 1
    n_right = 2
    n_down = 3
    n_left_up = 4
    n_right_up = 5
    n_right_down = 6
    n_left_down = 7

    tri_up_90 = 0 #self.tri_90列表中，0号元素表示上邻居三角形，下同
    tri_right_90 = 1
    tri_down_90 = 2
    tri_left_90 = 3

    tri_right_45_down = 0 #self.tri_45列表中，0号元素表示右下邻居三角形，下同
    tri_left_45_down = 1
    tri_down_45_left = 2
    tri_up_45_left = 3
    tri_left_45_up = 4
    tri_right_45_up = 5
    tri_up_45_right = 6
    tri_down_45_right = 7

    def __init__(self, pos):
        self.pos = tuple(pos)
        self.n_neighbors = [-1 for i in range(8)] #邻居图形节点

        self.tri_90 = [-1 for i in range(4)] #以90度角相邻的三角形，即该节点是该三角形的90度角顶点
        self.tri_45 = [-1 for i in range(8)] #以45度角相邻的三角形

#基本小三角形的数据结构
class Small_tri():
    def __init__(self, point_right, point_45_1, point_45_2): #45度的点是顺时针方向
        self.point_right = point_right #直角顶点序号
        self.point_45_1 = point_45_1
        self.point_45_2 = point_45_2
        self.occupied = False

    def __eq__(self, other): #判断两个小三角形是否相等
        if self.point_right != other.point_right:
            return False
        if self.point_45_1 != other.point_45_1:
            return False
        elif self.point_45_2 != other.point_45_2:
            return False
        return True


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
                    1.0, 125, thickness=1)

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
    #ret, binary = cv2.threshold(graph, 127, 255, cv2.THRESH_BINARY)
    #_, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #边缘检测
    corners = cv2.goodFeaturesToTrack(graph, 22, 0.17, 7)#basic_length/8)
    corners=np.int0(corners)[:,0,:].tolist()
    #print_nodes(copy.deepcopy(graph), corners, "get_contour_corners.jpg")

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
                index += 1'''

    return [corners]

def get_point_neighbors(point, all_nodes):
    points = []
    for i in range(8):
        if point.n_neighbors[i] != -1:
            p = all_nodes[point.n_neighbors[i]]
        else:
            p = None
        points.append(p)
    return points

def draw_tri_num(points, all_tri, graph, save_path):
    img = cv2.imread(graph, cv2.IMREAD_GRAYSCALE)
    for i, tri in enumerate(all_tri):
        point_right = points[tri.point_right].pos
        point_45_1 = points[tri.point_45_1].pos
        point_45_2 = points[tri.point_45_2].pos
        p_x = int((point_right[0]+point_45_1[0]+point_45_2[0])/3)
        p_y = int((point_right[1] + point_45_1[1] + point_45_2[1]) / 3)

        cv2.line(img, point_right, point_45_1, (100,150,160), thickness=2)
        cv2.line(img, point_right, point_45_2, (100, 150, 160), thickness=2)
        cv2.line(img, point_45_1, point_45_2, (100, 150, 160), thickness=2)

        #cv2.putText(img, "T%d" % i, (p_x, p_y), cv2.FONT_HERSHEY_PLAIN, 1.0, 125, thickness=1)
    cv2.imwrite(save_path, img)

#Pic feature类中与tri_90_num相关的45度三角形序号
def related_tri_90to45(tri_90_num):
    if tri_90_num == Pic_feature.tri_up_90:
        return [Pic_feature.tri_up_45_left, Pic_feature.tri_up_45_right]
    elif tri_90_num == Pic_feature.tri_right_90:
        return [Pic_feature.tri_right_45_down, Pic_feature.tri_right_45_up]
    elif tri_90_num == Pic_feature.tri_down_90:
        return [Pic_feature.tri_down_45_left, Pic_feature.tri_down_45_right]
    elif tri_90_num == Pic_feature.tri_left_90:
        return [Pic_feature.tri_left_45_down, Pic_feature.tri_left_45_up]
    else:
        raise Exception("relate tri 90 parameter wrong")

def related_tri_45to90(tri_45_num):
    if tri_45_num in [Pic_feature.tri_up_45_left, Pic_feature.tri_up_45_right]:
        return Pic_feature.tri_up_90
    elif tri_45_num in [Pic_feature.tri_right_45_down, Pic_feature.tri_right_45_up]:
        return Pic_feature.tri_right_90
    elif tri_45_num in [Pic_feature.tri_down_45_left, Pic_feature.tri_down_45_right]:
        return Pic_feature.tri_down_90
    elif tri_45_num in [Pic_feature.tri_left_45_down, Pic_feature.tri_left_45_up]:
        return Pic_feature.tri_left_90
    else:
        raise Exception("relate tri 45 parameter wrong")

def eliminate_nodes_tooClose(nodes, threhold):
    index = 0
    while True:
        if index >= len(nodes):
            break
        for i, p in enumerate(nodes):
            if i==index:
                continue
            dis = arc_len((p[0]-nodes[index][0], p[1]-nodes[index][1]))
            if dis < threhold:
                nodes.pop(index)
                index -= 1
                break
        index += 1
    return nodes
