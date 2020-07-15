import cv2
import numpy as np
from pic_process import tooClose, Pic_feature, print_nodes, get_point_neighbors, Small_tri, draw_tri_num, related_tri_90to45
import copy
AREA_THREHOLD = 0.75

#图形二值化
def binaryzation(grey_img, threhold_low, threhold_high):
    black_area = np.bitwise_and(grey_img >= threhold_low, grey_img <= threhold_high)
    white_area = np.bitwise_not(black_area)
    grey_img[black_area] = 0
    grey_img[white_area] = 255
    '''for i in range(grey_img.shape[0]):
        for j in range(grey_img.shape[1]):
            if grey_img[i][j]>threhold_low and grey_img[i][j]<threhold_high:
                grey_img[i][j] = 0
            else:
                grey_img[i][j] = 255'''
    return grey_img

#得到图形内部节点
def get_inner_nodes(now_nodes, basic_length, shape, k_90=np.sqrt(2), k_45=1/np.sqrt(2)):
    all_p = copy.deepcopy(now_nodes)
    for c_i, c_p in enumerate(now_nodes):
        possible_p=[]

        possible_p.append((c_p.pos[0] - basic_length * k_90, c_p.pos[1]))
        possible_p.append((c_p.pos[0], c_p.pos[1] - basic_length * k_90))
        possible_p.append((c_p.pos[0] + basic_length * k_90, c_p.pos[1]))
        possible_p.append((c_p.pos[0], c_p.pos[1] + basic_length * k_90))

        possible_p.append((c_p.pos[0] - basic_length *k_45, c_p.pos[1] - basic_length *k_45))
        possible_p.append((c_p.pos[0] + basic_length *k_45, c_p.pos[1] - basic_length *k_45))
        possible_p.append((c_p.pos[0] + basic_length *k_45, c_p.pos[1] + basic_length *k_45))
        possible_p.append((c_p.pos[0] - basic_length *k_45, c_p.pos[1] + basic_length *k_45))

        for p_i, p in enumerate(possible_p):
            if p[0]>shape[1] or p[0]<0 or p[1]>shape[0] or p[1]<0:
                continue

            flag=False
            for e_i, exist_p in enumerate(all_p):
                if tooClose(p, exist_p.pos, threhold=basic_length/8):
                    all_p[c_i].n_neighbors[p_i] = e_i
                    flag=True
                    break
            if flag==False:
                int_p = (int(p[0]), int(p[1]))
                int_p = Pic_feature(pos=tuple(int_p))
                all_p.append(int_p)

    return all_p

#黑色区域面积
def get_black_area(grey_img):
    black_area = ((255 - grey_img) // 255).sum()
    return black_area

#返回图形中所有的基本三角形
def get_all_tri(flag_90_45, all_nodes):
    all_tri = []
    for node_i, node in enumerate(all_nodes):
        neighbors = get_point_neighbors(node, all_nodes)

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
                all_nodes[node_i].tri_90[list_90[i] - list_90[0] + Pic_feature.tri_up_90] = exit_flag
            else:
                all_tri.append(tri)
                all_nodes[node_i].tri_90[list_90[i] - list_90[0] + Pic_feature.tri_up_90] = len(all_tri) - 1
                all_nodes[neighbor_1_num].tri_45[i * 2] = len(all_tri)-1
                all_nodes[neighbor_2_num].tri_45[i * 2 + 1] = len(all_tri) - 1

    return all_tri

#返回覆盖面积超过阀值AREA_THREHOLD的基本三角形
def get_valid_tris(basic_len, all_tri, grey_img, all_nodes):
    valid_tri = []
    one_tri_area = (basic_len**2)/2
    black_area_sum = get_black_area(grey_img)
    for tri_i, tri in enumerate(all_tri):
        p1 = all_nodes[tri.point_right].pos
        p2 = all_nodes[tri.point_45_1].pos
        p3 = all_nodes[tri.point_45_2].pos

        src_img = copy.deepcopy(grey_img)
        cv2.fillPoly(src_img, pts=[np.array([p1, p2, p3])], color=255)
        black_area_mTri = get_black_area(src_img)
        cover_area = black_area_sum-black_area_mTri #该三角形覆盖的面积

        if cover_area/one_tri_area > AREA_THREHOLD:
            valid_tri.append(tri_i)

    return valid_tri

#在白色图形中用黑色画出基本三角形all_tri
def get_transformed_img(all_tri, all_nodes, shape):
    img = np.array([255], dtype=np.uint8)
    img = np.resize(img ,shape)
    for tri in all_tri:
        p1 = all_nodes[tri.point_right].pos
        p2 = all_nodes[tri.point_45_1].pos
        p3 = all_nodes[tri.point_45_2].pos

        cv2.fillPoly(img, pts=[np.array([p1, p2, p3])], color=0)
    return img

#避免得到的基本三角形出现无法拼接的情况：基本三角形有一半被覆盖
def regulate_valid_tri(all_tri, valid_tri_index, all_nodes):
    for tri_i in valid_tri_index:
        tri = all_tri[tri_i]
        point_right = all_nodes[tri.point_right]
        index = -1
        for i, tri_n in enumerate(point_right.tri_90):
            if tri_n == tri_i:
                index = i
                break

        tri_45_num = related_tri_90to45(index)
        tri_45_0=point_right.tri_45[tri_45_num[0]]
        tri_45_1=point_right.tri_45[tri_45_num[1]]

        if tri_45_0 in valid_tri_index and (not tri_45_1 in valid_tri_index):
            valid_tri_index.append(tri_45_1)
        elif tri_45_1 in valid_tri_index and (not tri_45_0 in valid_tri_index):
            valid_tri_index.append(tri_45_0)

    regulated_tri = []
    for tri_i in valid_tri_index:
        regulated_tri.append(all_tri[tri_i])

    return regulated_tri

#从二值化图形得到规则化图形
def Transform_bi(basic_len, bi_img, flag_90_45 = True):
    all_nodes = [Pic_feature(pos=(int(bi_img.shape[1] / 2), int(bi_img.shape[0] / 2)))]
    all_len = 0
    while all_len != len(all_nodes):
        all_len = len(all_nodes)
        if flag_90_45 == True:
            all_nodes = get_inner_nodes(all_nodes, basic_length=basic_len, shape=bi_img.shape)
        else:
            all_nodes = get_inner_nodes(all_nodes, basic_length=basic_len, shape=bi_img.shape, k_90=1, k_45=1)

    all_tri = get_all_tri(flag_90_45, all_nodes)

    valid_tri_index = get_valid_tris(basic_len=basic_len, all_tri=all_tri, grey_img=bi_img, all_nodes=all_nodes)
    regulated_tri = regulate_valid_tri(all_tri=all_tri, valid_tri_index=valid_tri_index, all_nodes=all_nodes)

    img = get_transformed_img(all_tri=regulated_tri, all_nodes=all_nodes, shape=bi_img.shape)
    #cv2.imwrite("transform.jpg", img)
    #print_nodes(pic_path=bi_img, nodes=all_nodes, file_name="points.jpg")
    #draw_tri_num(all_nodes, all_tri, "points.jpg", "points.jpg")
    return img

#使用二值化图形bi_img得到规则化图形，总程序
def Transform_img(bi_img):
    black_area = get_black_area(bi_img)
    tri_num = 26
    basic_len = int(((black_area / tri_num) * 2) ** 0.5)  # 26个小三角形 117

    t_img_crosswise = Transform_bi(basic_len=basic_len, bi_img=bi_img, flag_90_45=True)
    t_img_lenthway = Transform_bi(basic_len=basic_len, bi_img=bi_img, flag_90_45=False)

    area_cross = get_black_area(t_img_crosswise)
    area_lenth = get_black_area(t_img_lenthway)
    if abs(area_cross - black_area) < abs(area_lenth - black_area):
        area_img = area_cross
        transform_img = t_img_crosswise
    else:
        area_img = area_lenth
        transform_img = t_img_lenthway

    small_tri_num = area_img * 2 / (basic_len ** 2)
    #print("small_tri_num:", small_tri_num)

    #cv2.imwrite("transform.jpg", transform_img)
    return transform_img, round(small_tri_num)

if __name__=="__main__":
    grey_img = cv2.imread("dog.png", cv2.IMREAD_GRAYSCALE)
    bi_img = binaryzation(grey_img, 0, 200)
    bi_img = cv2.resize(src=bi_img, dsize=(500, int(500*bi_img.shape[0]/bi_img.shape[1])))
    #cv2.imwrite("binaryzation.jpg", bi_img)

    Transform_img(bi_img)
