import cv2
import copy
import numpy as np

import pic_process
from pic_process import Graph, Pic_feature

#存储在open表中的节点
class Node_open_list():
    def __init__(self, contour_nodes):
        self.closed = False #是否已被放入closed表
        self.parent_node = -1
        self.contour_nodes = contour_nodes  #图形的边缘点
        self.child_num = -1

        self.large_triangle_1 = None
        self.large_triangle_2 = None
        self.middle_triangle = None
        self.small_triangle_1 = None
        self.small_triangle_2 = None
        self.square = None
        self.parallelogram = None

    def inherent_pieces(self, open_list):
        parent = open_list[self.parent_node]

        self.large_triangle_1 = copy.deepcopy(parent.large_triangle_1)
        self.large_triangle_2 = copy.deepcopy(parent.large_triangle_2)
        self.middle_triangle = copy.deepcopy(parent.middle_triangle)
        self.small_triangle_1 = copy.deepcopy(parent.small_triangle_1)
        self.small_triangle_2 = copy.deepcopy(parent.small_triangle_2)
        self.square = copy.deepcopy(parent.square)
        self.parallelogram = copy.deepcopy(parent.parallelogram)


#解决方案
class Solver():
    LARGE_TRIANGLE = 0
    MIDDLE_TRIANGLE = 1
    PARALLELOGRAM = 2
    SQUARE = 3
    SMALL_TRIANGLE = 4
    def __init__(self, pic_path, pieces_num = [2, 1, 1, 1, 2]):
        self.graph = Graph(pic_path)
        self.open_all_list = []  #open表
        self.open_list_num = []
        self.trace = []

        '''self.place_func = [self.place_large_triangle, self.place_large_triangle, self.place_middle_triangle,
                           self.place_parallelogram, self.place_square, self.place_small_triangle,
                           self.place_small_triangle]'''
        self.place_func = [self.place_large_triangle for i in range(pieces_num[self.LARGE_TRIANGLE])] + \
                          [self.place_middle_triangle for i in range(pieces_num[self.MIDDLE_TRIANGLE])] + \
                          [self.place_parallelogram for i in range(pieces_num[self.PARALLELOGRAM])] + \
                          [self.place_square for i in range(pieces_num[self.SQUARE])] + \
                          [self.place_small_triangle for i in range(pieces_num[self.SMALL_TRIANGLE])]

    def solve_dfs(self):
        init_node = Node_open_list(copy.deepcopy(self.graph.contour_points))
        self.open_all_list.append(init_node)
        self.open_list_num.append(0)


        children_node = self.place_poly(0, self.place_large_triangle)
        self.open_all_list.append(children_node[0])
        self.open_list_num.append(1)
        '''
        self.draw_current_graph(self.graph.current_graph, self.open_all_list[-1], (255, 255, 255))
        children_node = self.place_poly(self.open_list_num[-1], self.place_large_triangle)
        self.open_all_list.append(children_node[2])
        self.open_list_num.append(2)

        
        self.draw_current_graph(self.graph.current_graph, self.open_all_list[-1], (255, 255, 255))
        children_node = self.place_poly(self.open_list_num[-1], self.place_middle_triangle)
        self.open_all_list.append(children_node[4])
        self.open_list_num.append(3)

        self.draw_current_graph(self.graph.current_graph, self.open_all_list[-1], (255, 255, 255))
        children_node = self.place_poly(self.open_list_num[-1], self.place_parallelogram)
        self.open_all_list.append(children_node[1])
        self.open_list_num.append(4)'''
        '''for node in children_node:
            graph = copy.deepcopy(self.graph.init_graph)
            self.draw_current_graph(graph, node, (255, 255, 255))
            cv2.imwrite("current graph.jpg", graph)
            print(2)'''

        while len(self.open_list_num) > 0:
            node_num = self.open_list_num[-1]
            current_node = self.open_all_list[node_num]
            self.trace.append(node_num)
            self.open_all_list[node_num].closed = True

            self.graph.current_graph = copy.deepcopy(self.graph.init_graph)
            self.draw_current_graph(self.graph.current_graph, current_node, (255,255,255))
            cv2.imwrite("current graph.jpg", self.graph.current_graph)
            pic_process.print_nodes("current graph.jpg", current_node.contour_nodes, "current graph.jpg")

            finish_flag = self.get_finish_flag(current_node)
            print("finish:%d len of open list:%d" % (finish_flag, len(self.open_list_num)))
            if finish_flag == 7:
                print("finish")
                graph = cv2.imread(self.graph.pic_path, cv2.IMREAD_COLOR)
                self.draw_current_graph(graph, current_node)
                self.graph.current_graph = graph
                cv2.imwrite("result graph.jpg", graph)
                return 0

            children_node = []
            #for i, func in enumerate(self.place_func):
            for i in range(current_node.child_num+1, len(self.place_func)):
                self.open_all_list[node_num].child_num = i
                children_node = self.place_poly(node_num, self.place_func[i])

                '''valid_node = []
                for child in children_node:
                    if self.eliminate_exist_nodes(child)==False:
                        valid_node.append(child)'''

                if children_node != []:
                    break

            if children_node==[]:
                self.open_list_num.pop(-1)

            len_open_list = len(self.open_all_list)
            self.open_all_list += children_node
            self.open_list_num += [i for i in range(len_open_list, len(self.open_all_list))]

        print("can not solve")
        return -1

    #def func_has_tried(self, func, node):


    def solve(self):
        init_node = Node_open_list(copy.deepcopy(self.graph.contour_points))
        self.open_all_list.append(init_node)
        children_node, finish_flag = self.get_children_node(0)
        self.open_all_list += children_node
        self.open_all_list[0].closed = True

        while True:
            if len(self.open_all_list)==0:
                print("can not solve")
                return False
            for node_num, node in enumerate(self.open_all_list):
                if node.closed == True:  #节点在closed表中
                    continue
                self.draw_current_graph(self.graph.current_graph, node, (255,255,255))
                cv2.imwrite("current graph.jpg", self.graph.current_graph)

                children_node, finish_flag = self.get_children_node(node_num)
                self.open_all_list[node_num].closed = True

                print("finish:%d len of open list:%d"%(finish_flag, len(self.open_all_list)))
                if finish_flag==7:
                    print("finish")
                    return True
                else:
                    self.open_all_list += children_node

    #如果存在 返回True
    def eliminate_exist_nodes(self, node):

        for exist_node in self.open_all_list:
            check_polys = [exist_node.large_triangle_1, exist_node.large_triangle_2, exist_node.middle_triangle, exist_node.small_triangle_1, exist_node.small_triangle_2,
                           exist_node.square, exist_node.parallelogram]
            node_polys = [node.large_triangle_1, node.large_triangle_2, node.middle_triangle, node.small_triangle_1, node.small_triangle_2,
                          node.square, node.parallelogram]

            same_flag=True
            for i in range(len(check_polys)):
                if check_polys[i] != None and node_polys[i] != None:
                    for k in range(3):
                        point_1 = check_polys[i][k]
                        point_2 = node_polys[i][k]
                        if not pic_process.tooClose(point_1, point_2):
                            same_flag = False
                            break
                elif check_polys[i] ==None and node_polys[i] ==None:
                    continue
                else:
                    same_flag=False
                    break

            if same_flag==True:
                return True
        return False

    def get_finish_flag(self, node):
        result = 7
        if node.large_triangle_1 == None:
            result -= 1
        if node.large_triangle_2 == None:
            result -= 1
        if node.middle_triangle == None:
            result -= 1
        if node.small_triangle_1 == None:
            result -= 1
        if node.small_triangle_2 == None:
            result -= 1
        if node.square == None:
            result -= 1
        if node.parallelogram == None:
            result -= 1
        return result

    def place_poly(self, node_open_num, func):
        node_open = self.open_all_list[node_open_num]

        result_nodes = []
        for point_num, point in enumerate(node_open.contour_nodes):
            nodes = func(point, node_open_num)
            result_nodes += nodes

        if func==self.place_large_triangle:
            result_nodes = self.delete_same_node(result_nodes, node_open, "large_triangle_")
        elif func == self.place_small_triangle:
            result_nodes = self.delete_same_node(result_nodes, node_open, "small_triangle_")
        elif func == self.place_parallelogram:
            result_nodes = self.delete_same_node(result_nodes, node_open, "parallelogram")
        elif func == self.place_middle_triangle:
            result_nodes = self.delete_same_node(result_nodes, node_open, "middle_triangle")
        elif func == self.place_square:
            result_nodes = self.delete_same_node(result_nodes, node_open, "square")
        else:
            raise Exception("func error")


        return result_nodes

    def get_children_node(self, node_open_num):
        #node_open: open list 中的一个节点，self.graph.current_graph对应的节点
        node_open = self.open_all_list[node_open_num]

        large_triangle_nodes = []
        small_triangle_nodes = []
        middle_triangle_nodes = []
        all_nodes = []
        for point in node_open.contour_nodes:
            point_angle = self.get_point_angle(point=point, node_open=node_open)

            finish_flag = 7
            if node_open.large_triangle_1==None or node_open.large_triangle_2==None:
                finish_flag -= 1
                if node_open.large_triangle_1 == None:
                    finish_flag -= 1
                nodes = self.place_large_triangle(point_angle, point, node_open_num)
                large_triangle_nodes += nodes

            if node_open.small_triangle_1 == None or node_open.small_triangle_2 == None:
                finish_flag -= 1
                if node_open.small_triangle_1 == None:
                    finish_flag -= 1
                nodes = self.place_small_triangle(point_angle, point, node_open_num)
                small_triangle_nodes += nodes

            if node_open.middle_triangle == None:
                finish_flag -= 1
                nodes = self.place_middle_triangle(point_angle, point, node_open_num)
                middle_triangle_nodes += nodes

            if node_open.square == None:
                finish_flag -= 1
                nodes = self.place_square(point_angle, point, node_open_num)
                all_nodes += nodes

            if node_open.parallelogram == None:
                finish_flag -= 1
                nodes = self.place_parallelogram(point_angle, point, node_open_num)
                all_nodes += nodes

        large_triangle_nodes = self.delete_same_node(large_triangle_nodes, node_open, "large_triangle_")
        small_triangle_nodes = self.delete_same_node(small_triangle_nodes, node_open, "small_triangle_")
        middle_triangle_nodes = self.delete_same_node(middle_triangle_nodes, node_open, "middle_triangle")
        all_nodes += large_triangle_nodes
        all_nodes += small_triangle_nodes
        all_nodes += middle_triangle_nodes

        return all_nodes, finish_flag

        '''for node in large_triangle_nodes:
            img = copy.deepcopy(self.graph.current_graph)
            pic_process.print_nodes(img, node.large_triangle_1, "triangle_nodes.jpg")'''

    def delete_same_node(self, open_nodes, parent_node, class_key):

        for key,var in parent_node.__dict__.items():
            if var == None and (class_key in key):
                for i in range(len(open_nodes)):
                    j=i+1
                    while j < len(open_nodes):
                        same_flag = True
                        for k in range(3):
                            point_1 = open_nodes[i].__dict__[key][k]
                            point_2 = open_nodes[j].__dict__[key][k]
                            if not pic_process.tooClose(point_1, point_2):
                                same_flag=False
                                break
                        if same_flag:
                            open_nodes.pop(j)
                            j=j-1
                        j=j+1
                break
        return open_nodes

    def draw_current_graph(self, graph, node, color=None):

        if node.large_triangle_1 != None:
            self.draw_vital_points(graph, node.large_triangle_1, self.graph.pieces_colors[0] if color==None else color)
        if node.large_triangle_2 != None:
            self.draw_vital_points(graph, node.large_triangle_2, self.graph.pieces_colors[1] if color==None else color)
        if node.middle_triangle != None:
            self.draw_vital_points(graph, node.middle_triangle, self.graph.pieces_colors[2] if color==None else color)
        if node.small_triangle_1 != None:
            self.draw_vital_points(graph, node.small_triangle_1, self.graph.pieces_colors[3] if color==None else color)
        if node.small_triangle_2 != None:
            self.draw_vital_points(graph, node.small_triangle_2, self.graph.pieces_colors[4] if color==None else color)
        if node.square != None:
            self.draw_vital_points(graph, node.square, self.graph.pieces_colors[5] if color==None else color)
        if node.parallelogram != None:
            self.draw_vital_points(graph, node.parallelogram, self.graph.pieces_colors[6] if color==None else color)

    #返回pos位置处curre_graph的值
    def check_point(self, pos):
        if pos[1]<0 or pos[1]>=self.graph.current_graph.shape[0] or pos[0]<0 or pos[0] >= self.graph.current_graph.shape[1]:
            return 255
        return self.graph.current_graph[int(pos[1])][int(pos[0])]

    #返回pos位置周围curre_graph黑色像素的个数
    def check_point_around(self, pos, dis):
        black = 0
        for i in range(int(pos[0]) - dis, int(pos[0]) + dis):
            if i < 0 or i >= self.graph.current_graph.shape[1]:
                continue
            for j in range(int(pos[1]) - dis, int(pos[1]) + dis):
                if j < 0 or j >= self.graph.current_graph.shape[0]:
                    continue
                if self.graph.current_graph[j][i] < 125:
                    black += 1
        return black

    # 检查多边形能否放置在current_graph中，check_points是内部点，vital_points是顶点
    def check_poly(self, check_points, vital_points):
        #placeable_flag = True  # 五个内部检查点判断能否放置
        for check_point in check_points:
            if check_point == None:
                continue
            if self.check_point(check_point) > 125:
                #placeable_flag = False
                #break
                return False
        for check_point in vital_points:  # 四个边缘检查点判断能否放置
            if check_point == None:
                continue
            if self.check_point_around(check_point, dis=3) == 0: #周围黑色像素个数
                #placeable_flag = False
                #break
                return False
        for i in range(len(vital_points)-1):
            if self.check_arc(vital_points[i], vital_points[i+1])==False:
                #placeable_flag=False
                #break
                return False

        #return placeable_flag
        return True

    #检查多边形的边，point 1 和 point 2连线所组成的边
    def check_arc(self, point_1, point_2):
        x_dis = point_1[0] - point_2[0]
        y_dis = point_1[1] - point_2[1]
        check_num = 8
        for i in range(1, check_num):
            check_point = (int(point_2[0]+x_dis/check_num*i), int(point_2[1]+y_dis/check_num*i))
            if self.check_point_around(check_point, dis=4)==0:
                return False
        return True

    def draw_vital_points(self, src_img, vital_points, color = (255, 255, 255)):
        #vital_points = pic_process.poly_out_move(vital_points, 2)

        '''for i, point in enumerate(vital_points):
                        for exit_point in self.open_all_list[node_open_num].contour_nodes:
                            if pic_process.tooClose(point, exit_point.pos, threhold = 20):
                                vital_points[i] = exit_point.pos'''
        int_points=[]
        for point in vital_points:
            int_points.append((int(point[0]), int(point[1])))

        cv2.fillPoly(src_img, pts=[np.array(int_points)], color=color)
        return src_img


    #检查多边形能否放置，如果可以，返回Node_open_list
    def poly_place_getNewNode(self, check_points, vital_points, node_open_num, shape_sign=-1):
        placeable_flag = self.check_poly(check_points=check_points, vital_points=vital_points)

        if placeable_flag:
            img = copy.deepcopy(self.graph.current_graph)
            src_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if shape_sign == self.LARGE_TRIANGLE or shape_sign == self.MIDDLE_TRIANGLE:
                vital_points = vital_points[:3]

            self.draw_vital_points(src_img, vital_points)
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGRA2GRAY)
            #cv2.imwrite("fillPoly_large_tri.jpg",src_img)

            contour_points = pic_process.get_contour_nodes(graph=src_img, basic_length=self.graph.basic_length)
            #contour_points_c = pic_process.define_neighbor(contour_points)
            #pic_process.print_nodes("fillPoly_large_tri.jpg", contour_points, "large_tri_points.jpg")

            new_node = Node_open_list(contour_nodes=contour_points)
            new_node.parent_node = node_open_num
            new_node.child_num = self.open_all_list[node_open_num].child_num
            new_node.inherent_pieces(self.open_all_list)

            return new_node

        return placeable_flag

    def get_neighbor_arc(self, point, node_open):
        neighbor1 = node_open.contour_nodes[point.neighbor_point1]
        neighbor2 = node_open.contour_nodes[point.neighbor_point2]
        base_arc1 = (neighbor1.pos[0] - point.pos[0], neighbor1.pos[1] - point.pos[1])
        base_arc2 = (neighbor2.pos[0] - point.pos[0], neighbor2.pos[1] - point.pos[1])
        base_arc1 = (base_arc1[0] / pic_process.arc_len(base_arc1), base_arc1[1] / pic_process.arc_len(base_arc1))
        base_arc2 = (base_arc2[0] / pic_process.arc_len(base_arc2), base_arc2[1] / pic_process.arc_len(base_arc2))
        return base_arc1, base_arc2

    def get_135_arcs(self):
        arcs = []
        arcs.append([1, 0])
        arcs.append([1/np.sqrt(2), 1/np.sqrt(2)])
        arcs.append([0, 1])
        arcs.append([-1 / np.sqrt(2), 1 / np.sqrt(2)])
        arcs.append([-1, 0])
        arcs.append([-1 / np.sqrt(2), -1 / np.sqrt(2)])
        arcs.append([0, -1])
        arcs.append([1 / np.sqrt(2), -1 / np.sqrt(2)])
        arcs.append([1, 0])

        result = []
        for i in range(6):
            result.append([arcs[i], arcs[i+3]])

        return result

    def get_90_arcs(self):
        arcs = []
        arcs.append([1, 0])
        arcs.append([1/np.sqrt(2), 1/np.sqrt(2)])
        arcs.append([0, 1])
        arcs.append([-1 / np.sqrt(2), 1 / np.sqrt(2)])
        arcs.append([-1, 0])
        arcs.append([-1 / np.sqrt(2), -1 / np.sqrt(2)])
        arcs.append([0, -1])
        arcs.append([1 / np.sqrt(2), -1 / np.sqrt(2)])
        arcs.append([1, 0])

        result = []
        for i in range(7):
            result.append([arcs[i], arcs[i+2]])

        return result

    def get_45_arcs(self):
        arcs = []
        arcs.append([1, 0])
        arcs.append([1/np.sqrt(2), 1/np.sqrt(2)])
        arcs.append([0, 1])
        arcs.append([-1 / np.sqrt(2), 1 / np.sqrt(2)])
        arcs.append([-1, 0])
        arcs.append([-1 / np.sqrt(2), -1 / np.sqrt(2)])
        arcs.append([0, -1])
        arcs.append([1 / np.sqrt(2), -1 / np.sqrt(2)])
        arcs.append([1, 0])

        result = []
        for i in range(8):
            result.append([arcs[i], arcs[i+1]])

        return result

    # 在self.graph.current_graph以point为顶点放置一个小三角形
    def place_small_triangle(self, point, node_open_num):
        node_open = self.open_all_list[node_open_num]
        if node_open.small_triangle_2 != None and node_open.small_triangle_1 !=None:
            return []

        triangle_nodes = []

        arcs_45_pair = self.get_45_arcs()
        for arc_pair in arcs_45_pair:
            nodes_45 = self.place_small_triangle_45(point, node_open_num, arc_pair[0], arc_pair[1])
            triangle_nodes += nodes_45

        arcs_90_pair = self.get_90_arcs()
        for arc_pair in arcs_90_pair:
            nodes_90 = self.place_small_triangle_90(point, node_open_num, arc_pair[0], arc_pair[1])
            triangle_nodes += nodes_90

        return triangle_nodes

    # 在self.graph.current_graph以point为顶点放置一个中三角形
    def place_middle_triangle(self, point, node_open_num):
        node_open = self.open_all_list[node_open_num]
        if node_open.middle_triangle != None:
            return []

        triangle_nodes = []

        arcs_45_pair = self.get_45_arcs()
        for arc_pair in arcs_45_pair:
            nodes_45 = self.place_middle_triangle_45(point, node_open_num, arc_pair[0], arc_pair[1])
            triangle_nodes += nodes_45

        arcs_90_pair = self.get_90_arcs()
        for arc_pair in arcs_90_pair:
            nodes_90 = self.place_middle_triangle_90(point, node_open_num, arc_pair[0], arc_pair[1])
            triangle_nodes += nodes_90

        return triangle_nodes

    # 在self.graph.current_graph以point为顶点放置一个正方形
    def place_square(self, point, node_open_num):
        node_open = self.open_all_list[node_open_num]
        if node_open.square != None:
            return []

        square_nodes = []

        arcs_90_pair = self.get_90_arcs()
        for arc_pair in arcs_90_pair:
            nodes_90 = self.place_square_90(point, node_open_num, arc_pair[0], arc_pair[1])
            square_nodes += nodes_90

        return square_nodes

    # 在self.graph.current_graph以point为顶点放置一个大三角形
    def place_parallelogram(self, point, node_open_num):
        node_open = self.open_all_list[node_open_num]
        if node_open.parallelogram != None:
            return []

        parallelogram_nodes = []

        arcs_45_pair = self.get_45_arcs()
        for arc_pair in arcs_45_pair:
            nodes_45 = self.place_parallelogram_45(point, node_open_num, arc_pair[0], arc_pair[1])
            parallelogram_nodes += nodes_45

        arcs_135_pair = self.get_135_arcs()
        for arc_pair in arcs_135_pair:
            nodes_90 = self.place_parallelogram_135(point, node_open_num, arc_pair[0], arc_pair[1])
            parallelogram_nodes += nodes_90

        return parallelogram_nodes

    #在self.graph.current_graph以point为顶点放置一个大三角形
    def place_large_triangle(self, point, node_open_num):
        node_open = self.open_all_list[node_open_num]
        if node_open.large_triangle_1 != None and node_open.large_triangle_2 != None:
            return []

        triangle_nodes = []

        arcs_45_pair = self.get_45_arcs()
        for arc_pair in arcs_45_pair:
            nodes_45 = self.place_large_triangle_45(point, node_open_num, arc_pair[0], arc_pair[1])
            triangle_nodes += nodes_45

        arcs_90_pair = self.get_90_arcs()
        for arc_pair in arcs_90_pair:
            nodes_90 = self.place_large_triangle_90(point, node_open_num, arc_pair[0], arc_pair[1])
            triangle_nodes += nodes_90

        return triangle_nodes

    def place_parallelogram_135(self, point, node_open_num, base_arc1, base_arc2):
        return self.place_parallelogram_45(point, node_open_num, base_arc1, base_arc2)

    def place_square_90(self, point, node_open_num, base_arc1, base_arc2):
        #get_5_points_90(base_90_point, right_arc1, right_arc2):
        #base_90_point: 90度角顶点
        #right_arc1: 直角边向量

        point_near_1 = (point.pos[0] + base_arc1[0] * self.graph.basic_length,
                      point.pos[1] + base_arc1[1] * self.graph.basic_length) #与point相邻的顶点
        point_near_2 = (point.pos[0] + base_arc2[0] * self.graph.basic_length,
                      point.pos[1] + base_arc2[1] * self.graph.basic_length)
        opposite_point = (point.pos[0] + base_arc1[0] * self.graph.basic_length + base_arc2[0] * self.graph.basic_length,
                          point.pos[1] + base_arc1[1] * self.graph.basic_length + base_arc2[1] * self.graph.basic_length) #与point相对的顶点

        check_points = []
        check_points.append(pic_process.get_middle_point_tri(point_near_1, opposite_point, point.pos))
        check_points.append(pic_process.get_middle_point_tri(point_near_2, opposite_point, point.pos))
        vital_points = [point.pos, point_near_1, opposite_point, point_near_2]

        new_node = self.poly_place_getNewNode(check_points=check_points, vital_points=vital_points, node_open_num=node_open_num)

        if new_node:
            vital_points = vital_points[:4]
            new_node.square = vital_points
            return [new_node]
        return []

    def place_large_triangle_90(self, point, node_open_num, base_arc1, base_arc2):
        #get_5_points_90(base_90_point, right_arc1, right_arc2):
        #base_90_point: 90度角顶点
        #right_arc1: 直角边向量

        point_45_1 = (point.pos[0] + base_arc1[0] * self.graph.basic_length * 2,
                      point.pos[1] + base_arc1[1] * self.graph.basic_length * 2) #45度角顶点
        point_45_2 = (point.pos[0] + base_arc2[0] * self.graph.basic_length * 2,
                      point.pos[1] + base_arc2[1] * self.graph.basic_length * 2)
        hypotenuse_midpoint = ((point_45_1[0]+point_45_2[0])/2.0, (point_45_1[1]+point_45_2[1])/2.0) #斜边中点
        right_midpoint_1 = ((point.pos[0]+point_45_1[0])/2.0, (point.pos[1]+point_45_1[1])/2.0) #直角边中点
        right_midpoint_2 = ((point.pos[0] + point_45_2[0]) / 2.0, (point.pos[1] + point_45_2[1]) / 2.0)  # 直角边中点

        check_points = []
        check_points.append(pic_process.get_middle_point_tri(point_45_1, hypotenuse_midpoint, right_midpoint_1))
        check_points.append(pic_process.get_middle_point_tri(point_45_2, hypotenuse_midpoint, right_midpoint_2))
        check_points.append(pic_process.get_middle_point_tri(point.pos, right_midpoint_1, right_midpoint_2))
        check_points.append(pic_process.get_middle_point_tri(hypotenuse_midpoint, right_midpoint_1, right_midpoint_2))
        vital_points = [point.pos, point_45_1, point_45_2, hypotenuse_midpoint, right_midpoint_1, right_midpoint_2]

        new_node = self.poly_place_getNewNode(check_points=check_points, vital_points=vital_points, node_open_num=node_open_num, shape_sign=self.LARGE_TRIANGLE)

        if new_node:
            vital_points = vital_points[:3]
            vital_points.sort()
            if new_node.large_triangle_1 == None:
                new_node.large_triangle_1 = vital_points
            else:
                new_node.large_triangle_2 = vital_points
            return [new_node]
        return []

    def place_middle_triangle_90(self, point, node_open_num, base_arc1, base_arc2):
        #get_5_points_90(base_90_point, right_arc1, right_arc2):
        #base_90_point: 90度角顶点
        #right_arc1: 直角边向量

        point_45_1 = (point.pos[0] + base_arc1[0] * self.graph.basic_length * (2**0.5),
                      point.pos[1] + base_arc1[1] * self.graph.basic_length * (2**0.5)) #45度角顶点
        point_45_2 = (point.pos[0] + base_arc2[0] * self.graph.basic_length * (2**0.5),
                      point.pos[1] + base_arc2[1] * self.graph.basic_length * (2**0.5))
        hypotenuse_midpoint = ((point_45_1[0]+point_45_2[0])/2.0, (point_45_1[1]+point_45_2[1])/2.0) #斜边中点

        check_points = []
        check_points.append(pic_process.get_middle_point_tri(point_45_1, hypotenuse_midpoint, point.pos))
        check_points.append(pic_process.get_middle_point_tri(point_45_2, hypotenuse_midpoint, point.pos))
        vital_points = [point.pos, point_45_1, point_45_2, hypotenuse_midpoint]

        new_node = self.poly_place_getNewNode(check_points=check_points, vital_points=vital_points, node_open_num=node_open_num, shape_sign=self.MIDDLE_TRIANGLE)

        if new_node:
            vital_points = vital_points[:3]
            vital_points.sort()
            new_node.middle_triangle = vital_points
            return [new_node]
        return []

    def place_small_triangle_90(self, point, node_open_num, base_arc1, base_arc2):
        #get_2_points_90(base_90_point, right_arc1, right_arc2):
        #base_90_point: 90度角顶点
        #right_arc1: 直角边向量

        point_45_1 = (point.pos[0] + base_arc1[0] * self.graph.basic_length,
                      point.pos[1] + base_arc1[1] * self.graph.basic_length) #45度角顶点
        point_45_2 = (point.pos[0] + base_arc2[0] * self.graph.basic_length,
                      point.pos[1] + base_arc2[1] * self.graph.basic_length)

        check_points = []
        check_points.append(pic_process.get_middle_point_tri(point_45_1, point.pos, point_45_2))
        vital_points = [point.pos, point_45_1, point_45_2]

        new_node = self.poly_place_getNewNode(check_points=check_points, vital_points=vital_points, node_open_num=node_open_num)

        if new_node:
            vital_points = vital_points[:3]
            vital_points.sort()
            if new_node.small_triangle_1 == None:
                new_node.small_triangle_1 = vital_points
            else:
                new_node.small_triangle_2 = vital_points
            return [new_node]
        return []

    def place_large_triangle_45(self, point, node_open_num, base_arc1, base_arc2):
        node_open = self.open_all_list[node_open_num]

        def get_5_points_45(base_45_point, hypotenuse_arc, right_arc):
            #base_45_point: 45度角顶点
            #hypotenuse_arc: 斜边向量
            #right_arc: 直角边向量

            other_45_point = (base_45_point.pos[0] + hypotenuse_arc[0] * self.graph.basic_length * (2 ** 0.5) * 2,
                              base_45_point.pos[1] + hypotenuse_arc[1] * self.graph.basic_length * (2 ** 0.5) * 2) #另一个45度角顶点
            hypotenuse_midpoint = ((base_45_point.pos[0]+other_45_point[0])/2.0, (base_45_point.pos[1]+other_45_point[1])/2.0) #斜边中点
            right_point = (base_45_point.pos[0] + right_arc[0] * self.graph.basic_length * 2,
                              base_45_point.pos[1] + right_arc[1] * self.graph.basic_length * 2) #直角顶点
            right_midpoint = ((base_45_point.pos[0]+right_point[0])/2.0, (base_45_point.pos[1]+right_point[1])/2.0) #与point相邻直角边中点
            far_right_midpoint = ((other_45_point[0] + right_point[0]) / 2.0, (other_45_point[1] + right_point[1]) / 2.0)  # 远直角边中点

            return [other_45_point, hypotenuse_midpoint, right_point, right_midpoint, far_right_midpoint]

        arcs = [base_arc1, base_arc2]
        new_nodes = []
        for i in range(2):
            base_arc1 = arcs[i]
            base_arc2 = arcs[1-i]
            vital_points = get_5_points_45(point, hypotenuse_arc=base_arc1, right_arc=base_arc2)
            other_45_point, hypotenuse_midpoint, right_point, right_midpoint, far_right_midpoint = vital_points

            #pic_process.print_nodes(self.graph.pic_path, [point.pos]+vital_points, "large_tri_points.jpg")

            check_points = []
            check_points.append(pic_process.get_middle_point_tri(point.pos, hypotenuse_midpoint, right_midpoint))
            check_points.append(pic_process.get_middle_point_tri(right_point, far_right_midpoint, right_midpoint))
            check_points.append(pic_process.get_middle_point_tri(hypotenuse_midpoint, other_45_point, far_right_midpoint))
            check_points.append(pic_process.get_middle_point_tri(hypotenuse_midpoint, far_right_midpoint, right_midpoint))
            vital_points = [point.pos, other_45_point, right_point, hypotenuse_midpoint, far_right_midpoint, right_midpoint]

            new_node = self.poly_place_getNewNode(check_points=check_points, vital_points=vital_points, node_open_num=node_open_num, shape_sign=self.LARGE_TRIANGLE)

            if new_node:
                vital_points = vital_points[:3]
                vital_points.sort()
                if new_node.large_triangle_1 == None:
                    new_node.large_triangle_1 = vital_points
                else:
                    new_node.large_triangle_2 = vital_points
                new_nodes.append(new_node)

        return new_nodes

    def place_middle_triangle_45(self, point, node_open_num, base_arc1, base_arc2):
        node_open = self.open_all_list[node_open_num]

        def get_5_points_45(base_45_point, hypotenuse_arc, right_arc):
            #base_45_point: 45度角顶点
            #hypotenuse_arc: 斜边向量
            #right_arc: 直角边向量

            other_45_point = (base_45_point.pos[0] + hypotenuse_arc[0] * self.graph.basic_length * 2,
                              base_45_point.pos[1] + hypotenuse_arc[1] * self.graph.basic_length * 2) #另一个45度角顶点
            hypotenuse_midpoint = ((base_45_point.pos[0]+other_45_point[0])/2.0, (base_45_point.pos[1]+other_45_point[1])/2.0) #斜边中点
            right_point = (base_45_point.pos[0] + right_arc[0] * self.graph.basic_length * (2 ** 0.5),
                              base_45_point.pos[1] + right_arc[1] * self.graph.basic_length * (2 ** 0.5)) #直角顶点

            return [other_45_point, hypotenuse_midpoint, right_point]

        arcs = [base_arc1, base_arc2]
        new_nodes = []
        for i in range(2):
            base_arc1 = arcs[i]
            base_arc2 = arcs[1-i]
            vital_points = get_5_points_45(point, hypotenuse_arc=base_arc1, right_arc=base_arc2)
            other_45_point, hypotenuse_midpoint, right_point = vital_points

            #pic_process.print_nodes(self.graph.pic_path, [point.pos]+vital_points, "large_tri_points.jpg")

            check_points = []
            check_points.append(pic_process.get_middle_point_tri(point.pos, hypotenuse_midpoint, right_point))
            check_points.append(pic_process.get_middle_point_tri(right_point, hypotenuse_midpoint, other_45_point))
            vital_points = [point.pos, other_45_point, right_point, hypotenuse_midpoint]

            new_node = self.poly_place_getNewNode(check_points=check_points, vital_points=vital_points, node_open_num=node_open_num, shape_sign=self.MIDDLE_TRIANGLE)

            if new_node:
                vital_points = vital_points[:3]
                vital_points.sort()
                new_node.middle_triangle = vital_points

                new_nodes.append(new_node)
        return new_nodes

    def place_small_triangle_45(self, point, node_open_num, base_arc1, base_arc2):
        def get_2_points_45(base_45_point, hypotenuse_arc, right_arc):
            #base_45_point: 45度角顶点
            #hypotenuse_arc: 斜边向量
            #right_arc: 直角边向量

            other_45_point = (base_45_point.pos[0] + hypotenuse_arc[0] * self.graph.basic_length * (2 ** 0.5),
                              base_45_point.pos[1] + hypotenuse_arc[1] * self.graph.basic_length * (2 ** 0.5)) #另一个45度角顶点
            right_point = (base_45_point.pos[0] + right_arc[0] * self.graph.basic_length,
                              base_45_point.pos[1] + right_arc[1] * self.graph.basic_length) #直角顶点

            return [other_45_point, right_point]

        arcs = [base_arc1, base_arc2]
        new_nodes = []
        for i in range(2):
            base_arc1 = arcs[i]
            base_arc2 = arcs[1-i]
            vital_points = get_2_points_45(point, hypotenuse_arc=base_arc1, right_arc=base_arc2)
            other_45_point, right_point = vital_points

            #pic_process.print_nodes(self.graph.pic_path, [point.pos]+vital_points, "large_tri_points.jpg")

            check_points = []
            check_points.append(pic_process.get_middle_point_tri(point.pos, other_45_point, right_point))
            vital_points = [point.pos, other_45_point, right_point]

            new_node = self.poly_place_getNewNode(check_points=check_points, vital_points=vital_points, node_open_num=node_open_num)

            if new_node:
                vital_points = vital_points[:3]
                vital_points.sort()
                if new_node.small_triangle_1 == None:
                    new_node.small_triangle_1 = vital_points
                else:
                    new_node.small_triangle_2 = vital_points
                new_nodes.append(new_node)

        return new_nodes

    def place_parallelogram_45(self, point, node_open_num, base_arc1, base_arc2):
        def get_3_points_45(base_45_point, hypotenuse_arc, right_arc):
            #base_45_point: 45度角顶点
            #hypotenuse_arc: 斜边向量
            #right_arc: 直角边向量

            close_135_point = (base_45_point.pos[0] + hypotenuse_arc[0] * self.graph.basic_length,
                              base_45_point.pos[1] + hypotenuse_arc[1] * self.graph.basic_length) #较近的135度角顶点
            far_135_point = (base_45_point.pos[0] + right_arc[0] * self.graph.basic_length * (2**0.5),
                              base_45_point.pos[1] + right_arc[1] * self.graph.basic_length * (2**0.5)) #较远的135度角顶点
            far_45_point = (base_45_point.pos[0] + hypotenuse_arc[0] * self.graph.basic_length + right_arc[0] * self.graph.basic_length * (2 ** 0.5),
                             base_45_point.pos[1] + hypotenuse_arc[1] * self.graph.basic_length + right_arc[1] * self.graph.basic_length * (2 ** 0.5))  # 较远的45度角顶点

            return [close_135_point, far_135_point, far_45_point]

        arcs = [base_arc1, base_arc2]
        new_nodes = []
        for i in range(2):
            base_arc1 = arcs[i]
            base_arc2 = arcs[1-i]
            vital_points = get_3_points_45(point, hypotenuse_arc=base_arc1, right_arc=base_arc2)
            close_135_point, far_135_point, far_45_point = vital_points

            #pic_process.print_nodes(self.graph.pic_path, [point.pos]+vital_points, "large_tri_points.jpg")

            check_points = []
            check_points.append(pic_process.get_middle_point_tri(point.pos, close_135_point, far_135_point))
            check_points.append(pic_process.get_middle_point_tri(far_45_point, close_135_point, far_135_point))
            vital_points = [point.pos, close_135_point, far_45_point, far_135_point]

            new_node = self.poly_place_getNewNode(check_points=check_points, vital_points=vital_points, node_open_num=node_open_num)

            if new_node:
                vital_points = vital_points[:4]
                new_node.parallelogram = vital_points
                new_nodes.append(new_node)

        return new_nodes

    #返回顶点point所对应的角, 0到360度
    def get_point_angle(self, point, node_open):
        neighbor1 = node_open.contour_nodes[point.neighbor_point1]
        neighbor2 = node_open.contour_nodes[point.neighbor_point2]
        angle1 = pic_process.get_angle(p1=point.pos, p2=neighbor1.pos)
        angle2 = pic_process.get_angle(p1=point.pos, p2=neighbor2.pos)
        delta_angle = np.abs(angle1 - angle2)

        if pic_process.num_tooClose(delta_angle, 180, threhold=5):
            return delta_angle

        middle_arc = ((neighbor1.pos[0] + neighbor2.pos[0]) / 2.0 - point.pos[0],
                      (neighbor1.pos[1] + neighbor2.pos[1]) / 2.0 - point.pos[1])  # 中间向量
        middle_len = pic_process.arc_len(middle_arc)
        check_point = (point.pos[0] + middle_arc[0] * self.graph.basic_length / 4.0 / middle_len,
                       point.pos[1] + middle_arc[1] * self.graph.basic_length / 4.0 / middle_len)
        inner_angle = True
        if check_point[0] < 0 or check_point[0] > self.graph.init_graph.shape[1] or check_point[1] < 0 or check_point[1] > self.graph.init_graph.shape[0]:
            inner_angle = False
        if self.graph.current_graph[int(check_point[1])][int(check_point[0])] > 125:
            inner_angle = False
        if not inner_angle:
            delta_angle = 360 - delta_angle if delta_angle < 180 else delta_angle  # 该角是外角
        else:
            delta_angle = 360 - delta_angle if delta_angle > 180 else delta_angle  # 该角是内角
        return delta_angle

if __name__ == "__main__":
    import time
    start_time = time.time()
    solver = Solver("E:\\pycharmProject\\AI\\tangram_solve\\tangrams\\tangram2.png")
    solver.solve_dfs()
    end_time = time.time()
    print("time:", end_time-start_time)