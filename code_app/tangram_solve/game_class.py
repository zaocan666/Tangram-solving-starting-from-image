import cv2
import copy
import numpy as np
import time

import pic_process
from pic_process import Graph, Pic_feature

#搜索图节点，存储在open表中
class Node_open_list():
    def __init__(self, all_tri):
        self.closed = False #是否已被放入closed表
        self.parent_node = -1
        self.all_tri = all_tri
        self.child_num = -1
        self.score = 0

        self.large_triangles = []
        self.middle_triangles = []
        self.small_triangles = []
        self.squares = []
        self.parallelograms = []

    def inherent_pieces(self, open_all_list):
        parent = open_all_list[self.parent_node]

        self.large_triangles = copy.deepcopy(parent.large_triangles)
        self.middle_triangles = copy.deepcopy(parent.middle_triangles)
        self.small_triangles = copy.deepcopy(parent.small_triangles)
        self.squares = copy.deepcopy(parent.squares)
        self.parallelograms = copy.deepcopy(parent.parallelograms)


#解决方案
class Solver():
    LARGE_TRIANGLE = 0
    MIDDLE_TRIANGLE = 1
    PARALLELOGRAM = 2
    SQUARE = 3
    SMALL_TRIANGLE = 4
    def __init__(self, pic_path, pieces_num = [2, 1, 1, 1, 2], output_line = None): #pieces_num 代表每种形状的板要放几个
        self.pieces_num = pieces_num
        self.small_tri_num = get_small_tri_num(pieces_num)
        self.output_line = output_line
        self.graph = Graph(pic_path, small_tri_num = self.small_tri_num)
        self.open_all_list = []  #open表 全部节点
        self.open_list_num = []
        self.trace = [] #记录求解经过的节点序号

        A = [1,2]+[2,3]

        self.place_func = [self.place_large_triangle for i in range(pieces_num[self.LARGE_TRIANGLE])]+ \
                          [self.place_middle_triangle for i in range(pieces_num[self.MIDDLE_TRIANGLE])]+ \
                          [self.place_parallelogram for i in range(pieces_num[self.PARALLELOGRAM])]+ \
                          [self.place_square for i in range(pieces_num[self.SQUARE])]+ \
                          [self.place_small_triangle for i in range(pieces_num[self.SMALL_TRIANGLE])]

    def solve_dfs2(self):
        init_node = Node_open_list(all_tri=copy.deepcopy(self.graph.all_tri))
        self.open_all_list.append(init_node)
        self.open_list_num.append(0)
        self.debug_refresh_index = 0
        while len(self.open_list_num) > 0:
            node_num = self.open_list_num.pop(-1)
            self.trace.append(node_num)
            node = self.open_all_list[node_num]
            self.open_all_list[node_num].closed = True

            #self.graph.current_graph = copy.deepcopy(self.graph.init_color_img)
            #self.draw_current_graph(self.graph.current_graph,node)
            #cv2.imwrite("current graph.jpg", self.graph.current_graph)

            finish_flag = self.get_finish_flag(node)
            debug_text = "finish:%d/%d len of open list:%d" % (finish_flag, sum(self.pieces_num), len(self.open_list_num))
            self.debug_output(debug_text)
            if finish_flag == sum(self.pieces_num):
                self.debug_refresh_index = 10000
                self.debug_output(debug_text)

                graph = copy.deepcopy(self.graph.init_color_img)
                self.draw_current_graph(graph, node)
                self.graph.current_graph = graph
                #cv2.imwrite("result graph.jpg", graph)
                return 0

            children_node = []
            for func in self.place_func:
                children_node = func(node_num)
                if type(children_node) != list:
                    continue
                else:
                    break

            len_open_list = len(self.open_all_list)
            self.open_all_list += children_node
            self.open_list_num += [i for i in range(len_open_list, len(self.open_all_list))]

    def debug_output(self,debug_text):
        if self.output_line != None:
            self.debug_refresh_index += 1
            if self.debug_refresh_index >= 50:
                self.debug_refresh_index = 0
                self.output_line.setText(debug_text)
                self.output_line.repaint()
        else:
            print(debug_text)

    #返回已摆放板块的个数
    def get_finish_flag(self, node):
        result = 0
        result += len(node.large_triangles)
        result += len(node.middle_triangles)
        result += len(node.small_triangles)
        result += len(node.squares)
        result += len(node.parallelograms)

        '''for tri in node.all_tri:
            if tri.occupied == True:
                result += 1'''
        return result

    #在graph上画出搜索节点node对应的拼接图形
    def draw_current_graph(self, graph, node):
        index = 0
        index = self.draw_vital_points(graph, node.large_triangles, self.graph.pieces_colors, index)
        index = self.draw_vital_points(graph, node.middle_triangles, self.graph.pieces_colors, index)
        index = self.draw_vital_points(graph, node.small_triangles, self.graph.pieces_colors, index)
        index = self.draw_vital_points(graph, node.squares, self.graph.pieces_colors, index)
        index = self.draw_vital_points(graph, node.parallelograms, self.graph.pieces_colors, index)
        return graph

    # 在src_img上用同样的颜色color画出small_tris小三角形组
    def draw_vital_points(self, src_img, small_tris, color, index=0):
        for small_tris_group in small_tris:
            index += 1
            for tri_n in small_tris_group:
                tri = self.graph.all_tri[tri_n]
                p1 = self.graph.all_nodes[tri.point_right].pos
                p2 = self.graph.all_nodes[tri.point_45_1].pos
                p3 = self.graph.all_nodes[tri.point_45_2].pos
                cv2.fillPoly(src_img, pts=[np.array([p1,p2,p3])], color=color[index])
        return index

    # 在搜索节点node的基础上，查找可以摆放小三角形板块的位置，并得到一批后继节点
    def place_small_triangle(self, node_open_num):
        result_nodes = []
        node = self.open_all_list[node_open_num]
        if len(node.small_triangles) >= self.pieces_num[self.SMALL_TRIANGLE]:
            return -1
        for tri_i, tri in enumerate(node.all_tri):
            if tri.occupied == True:
                continue
            new_tris = copy.deepcopy(node.all_tri)
            new_tris[tri_i].occupied = True

            new_tris, _, _ = self.make_related_tri45_occupied(new_tris, tri_i)

            new_node = Node_open_list(all_tri=new_tris)
            new_node.parent_node = node_open_num
            new_node.inherent_pieces(self.open_all_list)

            new_node.small_triangles.append([tri_i])

            result_nodes.append(new_node)

        return result_nodes

    # 在搜索节点node的基础上，查找可以摆放中三角形板块的位置，并得到一批后继节点
    def place_middle_triangle(self, node_open_num):
        result_nodes = []
        node = self.open_all_list[node_open_num]
        if len(node.middle_triangles) >=  self.pieces_num[self.MIDDLE_TRIANGLE]:
            return -1
        for tri_i, tri in enumerate(node.all_tri):
            if tri.occupied == True:
                continue

            point_right = tri.point_right
            neighbor_tris = []  # 符合条件的相邻三角形
            for tri_90_num in self.graph.all_nodes[point_right].tri_90:
                if tri_90_num == -1:
                    continue
                tri_90 = node.all_tri[tri_90_num]
                if tri_90.occupied == True:
                    continue
                if tri_90.point_45_1 == tri.point_45_2:
                    neighbor_tris.append(tri_90_num)
                if tri_90.point_45_2 == tri.point_45_1:
                    neighbor_tris.append(tri_90_num)

            for neighbor_tri in neighbor_tris:
                new_tris = copy.deepcopy(node.all_tri)
                new_tris[tri_i].occupied = True

                new_tris[neighbor_tri].occupied = True

                new_tris, _, _ = self.make_related_tri45_occupied(new_tris, tri_i)
                new_tris, _, _ = self.make_related_tri45_occupied(new_tris, neighbor_tri)

                new_node = Node_open_list(all_tri=new_tris)
                new_node.parent_node = node_open_num
                new_node.inherent_pieces(self.open_all_list)

                tris = [tri_i, neighbor_tri]
                tris.sort()
                new_node.middle_triangles.append(tris)

                result_nodes.append(new_node)

        i = 0
        j = 0
        while i < len(result_nodes):
            j = i + 1
            while j < len(result_nodes):
                same_flag = True
                for k in range(2):
                    if result_nodes[i].middle_triangles[-1][k] != result_nodes[j].middle_triangles[-1][k]:
                        same_flag = False
                if same_flag:
                    result_nodes.pop(j)
                    j -= 1
                j += 1
            i += 1

        return result_nodes

    # 在搜索节点node的基础上，查找可以摆放正方形板块的位置，并得到一批后继节点
    def place_square(self, node_open_num):
        result_nodes = []
        node = self.open_all_list[node_open_num]
        if len(node.squares) >= self.pieces_num[self.SQUARE]:
            return -1
        for tri_i, tri in enumerate(node.all_tri):
            if tri.occupied == True:
                continue

            new_tris = copy.deepcopy(node.all_tri)
            new_tris[tri_i].occupied = True

            new_tris, tris_45, occupied_flag = self.make_related_tri45_occupied(new_tris, tri_i)
            if -1 in tris_45:
                continue
            if occupied_flag==True:
                continue

            new_tris, tris_45_2, _ = self.make_related_tri45_occupied(new_tris, tris_45[0])

            new_node = Node_open_list(all_tri=new_tris)
            new_node.parent_node = node_open_num
            new_node.inherent_pieces(self.open_all_list)

            tris = tris_45+tris_45_2
            tris.sort()
            new_node.squares.append(tris)

            result_nodes.append(new_node)

        i=0
        j=0
        while i < len(result_nodes):
            j=i+1
            while j< len(result_nodes):
                same_flag = True
                for k in range(4):
                    if result_nodes[i].squares[-1][k] != result_nodes[j].squares[-1][k]:
                        same_flag = False
                if same_flag:
                    result_nodes.pop(j)
                    j -= 1
                j += 1
            i += 1

        return result_nodes

    # 在搜索节点node的基础上，查找可以摆放平行四边形板块的位置，并得到一批后继节点
    def place_parallelogram(self, node_open_num):
        result_nodes = []
        node = self.open_all_list[node_open_num]
        if len(node.parallelograms) >= self.pieces_num[self.PARALLELOGRAM]:
            return -1
        for tri_i, tri in enumerate(node.all_tri):
            if tri.occupied == True:
                continue

            point_right = tri.point_right
            neighbor_tris = [] #符合条件的相邻三角形
            for tri_45 in self.graph.all_nodes[point_right].tri_45:
                if tri_45==-1:
                    continue
                if node.all_tri[tri_45].occupied==True:
                    continue
                tri_45_right_point = node.all_tri[tri_45].point_right

                if tri_45_right_point == tri.point_45_1:
                    neighbor_tris.append(tri_45)
                if tri_45_right_point == tri.point_45_2:
                    neighbor_tris.append(tri_45)

            for neighbor_tri in neighbor_tris:
                new_tris = copy.deepcopy(node.all_tri)
                new_tris[tri_i].occupied = True
                new_tris[neighbor_tri].occupied = True

                new_tris, tri_cover, _ = self.make_related_tri45_occupied(new_tris, tri_i)
                if neighbor_tri in tri_cover:
                    continue
                new_tris,_, _ = self.make_related_tri45_occupied(new_tris, neighbor_tri)

                new_node = Node_open_list(all_tri=new_tris)
                new_node.parent_node = node_open_num
                new_node.inherent_pieces(self.open_all_list)

                tris = [tri_i, neighbor_tri]
                tris.sort()
                new_node.parallelograms.append(tris)

                result_nodes.append(new_node)

        i = 0
        j = 0
        while i < len(result_nodes):
            j = i + 1
            while j < len(result_nodes):
                same_flag = True
                for k in range(2):
                    if result_nodes[i].parallelograms[-1][k] != result_nodes[j].parallelograms[-1][k]:
                        same_flag = False
                if same_flag:
                    result_nodes.pop(j)
                    j -= 1
                j += 1
            i += 1

        return result_nodes

    #在搜索节点node的基础上，查找可以摆放大三角形板块的位置，并得到一批后继节点
    def place_large_triangle(self, node_open_num):
        result_nodes = []
        node = self.open_all_list[node_open_num]
        if len(node.large_triangles) >= self.pieces_num[self.LARGE_TRIANGLE]: #搜索节点node已不需要摆放大三角形板块
            return -1
        for tri_i, tri in enumerate(node.all_tri):
            if tri.occupied == True:
                continue

            new_tris = copy.deepcopy(node.all_tri)
            new_tris[tri_i].occupied = True
            #tri_i是大三角形的直角对应的三角形
            new_tris, inner_two_tris, occupied_flag = self.make_related_tri45_occupied(new_tris, tri_i)
            if -1 in inner_two_tris:
                continue
            if occupied_flag == True:
                continue

            outer_two_tris = []

            for inner_tri_num in inner_two_tris:
                inner_tri = node.all_tri[inner_tri_num]
                point_right = inner_tri.point_right
                for tri_90_num in self.graph.all_nodes[point_right].tri_90:
                    if tri_90_num == -1:
                        continue
                    if node.all_tri[tri_90_num].occupied == True:
                        continue
                    tri_90 = node.all_tri[tri_90_num]
                    if tri_90.point_45_1 == tri.point_right or tri_90.point_45_2 == tri.point_right:
                        continue
                    if tri_90.point_45_1 == inner_tri.point_45_2:
                        outer_two_tris.append(tri_90_num)
                        break
                    if tri_90.point_45_2 == inner_tri.point_45_1:
                        outer_two_tris.append(tri_90_num)
                        break
            complete_4_tris = outer_two_tris + inner_two_tris
            if len(complete_4_tris) != 4:
                continue

            for small_tri in complete_4_tris:
                new_tris, _, _ = self.make_related_tri45_occupied(new_tris, small_tri)
                new_tris[small_tri].occupied = True

            new_node = Node_open_list(all_tri=new_tris)
            new_node.parent_node = node_open_num
            new_node.inherent_pieces(self.open_all_list)
            complete_4_tris.sort()

            new_node.large_triangles.append(complete_4_tris)

            result_nodes.append(new_node)

        return result_nodes

    #使得与tri90_n有重合的tri状态变成occupied
    def make_related_tri45_occupied(self, tris, tri90_n):
        tri = tris[tri90_n]
        point_right = tri.point_right
        point = self.graph.all_nodes[point_right]
        index = -1
        for i, tri_n in enumerate(point.tri_90):
            if tri_n == tri90_n:
                index = i
                break
        tri_45s_num = pic_process.related_tri_90to45(index)
        tri_45s = []
        for tri_num in tri_45s_num:
            tri_n = point.tri_45[tri_num]
            tri_45s.append(tri_n)
            flag = False
            if tri_n != -1:
                flag = (tris[tri_n].occupied) or flag
                tris[tri_n].occupied = True
        return tris, tri_45s, flag

def get_small_tri_num(small_tri_num):
    result = 0
    result += small_tri_num[Solver.LARGE_TRIANGLE] * 4
    result += small_tri_num[Solver.MIDDLE_TRIANGLE] * 2
    result += small_tri_num[Solver.PARALLELOGRAM] * 2
    result += small_tri_num[Solver.SQUARE] * 2
    result += small_tri_num[Solver.SMALL_TRIANGLE] * 1

    return result

if __name__ == "__main__":
    start_time = time.time()
    solver = Solver("tangrams\\tangram2.png")#, pieces_num=[2,0,1,0,2+10+2+2]) #4 11 14 16 17 18 19 27
    solver.solve_dfs2()
    end_time = time.time()
    print("time:", end_time-start_time)