from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QDesktopWidget, QMessageBox, QPushButton, QLabel, QLineEdit
from PyQt5.QtGui import QPainter, QPolygon, QBrush, QIntValidator
from PyQt5.QtCore import Qt, QPoint
from PyQt5 import QtGui
import sys
import numpy as np
import copy
import cv2

from game_class import Solver, get_small_tri_num
from pic_process import tooClose, arc_len, num_tooClose
from ui_basic import Solve_frame


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui_userD = Ui_userDefine()
        self.ui_userD.setParent(self)
        self.ui_userD.initUI()

        self.resize(1200, 700)
        self.center()
        self.setWindowTitle('Tangram')
        self.show()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2)

class Ui_userDefine(QWidget):
    def __init__(self):
        super().__init__()

    def initUI(self):
        self.resize(1200, 700)

        self.Tri_area = Tri_choosing_area(self)
        self.Tri_area.move(20, 50)

        hint_label = QLabel(parent=self)
        hint_label.setText("请点击单位长度距离的点，每\r\n点击三个点构成一个小三角形\r\n。可解图形应包括位置合适的\r\n若干个小三角形,左边是范例。\r\n画完后在右边框内输入所画板\r\n块个数。")
        hint_label.resize(200, 100)
        hint_label.move(self.Tri_area.geometry().x()+self.Tri_area.geometry().width()+40, self.Tri_area.geometry().y() + 20)

        flip_button = QPushButton(parent=self)
        flip_button.setText("翻转坐标点")
        flip_button.move(hint_label.geometry().x(), hint_label.geometry().y()+hint_label.geometry().height() + 20)
        flip_button.pressed.connect(self.Tri_area.flipPoints)

        revocation_button = QPushButton(parent=self)
        revocation_button.setText("撤销")
        revocation_button.move(flip_button.geometry().x(), flip_button.geometry().y() + flip_button.geometry().height() + 20)
        revocation_button.pressed.connect(self.Tri_area.revocate)

        clear_button = QPushButton(parent=self)
        clear_button.setText("清除")
        clear_button.move(revocation_button.geometry().x(),revocation_button.geometry().y() + revocation_button.geometry().height() + 20)
        clear_button.pressed.connect(self.Tri_area.clear_tri)

        finish_button = QPushButton(parent=self)
        finish_button.setText("完成")
        finish_button.move(clear_button.geometry().x(),clear_button.geometry().y() + clear_button.geometry().height() + 20)
        finish_button.pressed.connect(lambda :self.Tri_area.finish(qlabel=self.pic_area, parent=self, small_tri_num=get_small_tri_num(self.pieces_num_input.get_line_nums())))

        self.pieces_num_input = Pieces_input_lines(self)
        self.pieces_num_input.move(flip_button.x()+flip_button.width()+40, flip_button.y())

        self.pic_area = QLabel(parent=self)
        self.pic_area.resize(self.Tri_area.geometry().width(), self.Tri_area.geometry().height())
        self.pic_area.move(self.Tri_area.geometry().x(), self.Tri_area.geometry().y())
        self.pic_area.setVisible(False)

        self.solver_class = Solve_frame(self, size=(300, 200), pic_area=self.pic_area)
        self.solver_class.move(finish_button.geometry().x(), finish_button.geometry().y() + finish_button.geometry().height() + 40)
        self.solver_class.solve_problem_button.pressed.connect(lambda: self.solver_class.solve_button_pressed(path=self.Tri_area.finish_pic, pieces_num = self.pieces_num_input.get_line_nums()))

        restart_button = QPushButton(parent=self)
        restart_button.setText("新图形")
        restart_button.move(self.solver_class.geometry().x(), self.solver_class.geometry().y() + self.solver_class.geometry().height() + 40)
        restart_button.pressed.connect(self.restart)

    def restart(self):
        self.solver_class.playing_index = -1
        self.solver_class.solver = None
        self.solver_class.timer.stop()

        self.Tri_area.pressed_point = []
        self.Tri_area.init_tris()
        self.Tri_area.points = self.Tri_area.init_points
        self.Tri_area.flip_point = False
        self.Tri_area.finish_pic = None

        self.pic_area.setVisible(False)
        self.Tri_area.setVisible(True)
        self.repaint()

class Pieces_input_lines(QWidget):
    def __init__(self, parent):
        super().__init__()

        self.setParent(parent)
        self.resize(250, 400)

        large_tri_label = QLabel(parent = self)
        large_tri_label.setText("大三角形个数：")
        large_tri_label.move(0, 0)
        self.large_tri_line = QLineEdit(parent = self)
        self.large_tri_line.setText("2")
        self.large_tri_line.move(large_tri_label.geometry().x()+large_tri_label.geometry().width()+30, large_tri_label.geometry().y())

        middle_tri_label = QLabel(parent=self)
        middle_tri_label.setText("中三角形个数：")
        middle_tri_label.move(large_tri_label.x(), large_tri_label.y()+large_tri_label.height()+5)
        self.middle_tri_line = QLineEdit(parent=self)
        self.middle_tri_line.setText("1")
        self.middle_tri_line.move(self.large_tri_line.x(), middle_tri_label.y())

        parallelogram_label = QLabel(parent=self)
        parallelogram_label.setText("平行四边形个数：")
        parallelogram_label.move(large_tri_label.x(), middle_tri_label.y() + middle_tri_label.height() + 5)
        self.parallelogram_line = QLineEdit(parent=self)
        self.parallelogram_line.setText("1")
        self.parallelogram_line.move(self.large_tri_line.x(), parallelogram_label.y())

        square_label = QLabel(parent=self)
        square_label.setText("正方形个数：")
        square_label.move(large_tri_label.x(), parallelogram_label.y() + parallelogram_label.height() + 5)
        self.square_line = QLineEdit(parent=self)
        self.square_line.setText("1")
        self.square_line.move(self.large_tri_line.x(), square_label.y())

        small_tri_label = QLabel(parent=self)
        small_tri_label.setText("小三角形个数：")
        small_tri_label.move(large_tri_label.x(), square_label.y() + square_label.height() + 5)
        self.small_tri_line = QLineEdit(parent=self)
        self.small_tri_line.setText("2")
        self.small_tri_line.move(self.large_tri_line.x(), small_tri_label.y())

        self.large_tri_line.setValidator(QIntValidator(0, 99))
        self.middle_tri_line.setValidator(QIntValidator(0, 99))
        self.small_tri_line.setValidator(QIntValidator(0, 99))
        self.square_line.setValidator(QIntValidator(0, 99))
        self.parallelogram_line.setValidator(QIntValidator(0, 99))

    def get_line_nums(self):
        result = [0 for i in range(5)]
        result[Solver.LARGE_TRIANGLE] = int(self.large_tri_line.text())
        result[Solver.MIDDLE_TRIANGLE] = int(self.middle_tri_line.text())
        result[Solver.PARALLELOGRAM] = int(self.parallelogram_line.text())
        result[Solver.SQUARE] = int(self.square_line.text())
        result[Solver.SMALL_TRIANGLE] = int(self.small_tri_line.text())

        return result

class Triangle():
    def __init__(self, p1, p2, p3):
        self.ps = [p1, p2, p3]
        self.ps.sort()

class Tri_choosing_area(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.basic_length = 60
        self.resize(self.basic_length * 12, self.basic_length * 10)
        self.setParent(parent)
        self.init_points = self.get_all_points()
        self.diff_points = self.get_all_points(k_90=1, k_45=1)
        self.points=self.init_points
        self.choosed_tri = []
        self.init_tris()
        self.pressed_point = []
        self.flip_point = False
        self.finish_pic = None

    def init_tris(self):
        tris = [[(273, 128), (316, 171), (358, 128)], [(316, 171), (358, 128), (359, 214)],
         [(273, 128), (274, 214), (316, 171)], [(274, 214), (316, 171), (359, 214)],
         [(358, 128), (401, 171), (443, 128)], [(358, 128), (359, 214), (401, 171)],
         [(401, 171), (443, 128), (444, 214)], [(359, 214), (401, 171), (444, 214)],
         [(274, 214), (317, 257), (359, 214)], [(274, 214), (274, 299), (317, 257)],
         [(317, 257), (359, 214), (360, 300)], [(274, 299), (317, 257), (360, 300)],
         [(359, 214), (402, 257), (444, 214)], [(359, 214), (360, 300), (402, 257)],
         [(402, 257), (444, 214), (444, 299)], [(360, 300), (402, 257), (444, 299)]]
        self.choosed_tri = []
        for tri_pos in tris:
            self.choosed_tri.append(Triangle(*tri_pos))

    def init_tris_flip(self):
        tris = [[(240, 180), (300, 120), (300, 180)], [(300, 120), (300, 180), (360, 180)],
         [(300, 120), (360, 120), (360, 180)], [(360, 120), (360, 180), (420, 180)],
         [(240, 180), (240, 240), (300, 180)], [(240, 240), (300, 180), (300, 240)],
         [(300, 180), (300, 240), (360, 180)], [(300, 240), (360, 180), (360, 240)],
         [(360, 180), (360, 240), (420, 180)], [(360, 240), (420, 180), (420, 240)],
         [(240, 240), (240, 300), (300, 240)], [(240, 300), (300, 240), (300, 300)],
         [(300, 240), (300, 300), (360, 240)], [(300, 300), (360, 240), (360, 300)],
         [(360, 240), (360, 300), (420, 240)], [(360, 300), (420, 240), (420, 300)]]
        self.choosed_tri = []
        for tri_pos in tris:
            self.choosed_tri.append(Triangle(*tri_pos))

    def get_all_points(self, k_90=np.sqrt(2), k_45=1/np.sqrt(2)):
        self.points=[]
        size_x = self.geometry().width()
        size_y = self.geometry().height()
        points = [(int(size_x/2), int(size_y/2))]

        for j in range(10):
            now_points = copy.deepcopy(points)
            for point in now_points:
                possible_p = []

                if k_90 == 1:
                    possible_p.append((point[0] - self.basic_length * k_90, point[1]))
                    possible_p.append((point[0], point[1] - self.basic_length * k_90))
                    possible_p.append((point[0] + self.basic_length * k_90, point[1]))
                    possible_p.append((point[0], point[1] + self.basic_length * k_90))
                else:
                    possible_p.append((point[0] - self.basic_length * k_45, point[1] - self.basic_length * k_45))
                    possible_p.append((point[0] + self.basic_length * k_45, point[1] - self.basic_length * k_45))
                    possible_p.append((point[0] + self.basic_length * k_45, point[1] + self.basic_length * k_45))
                    possible_p.append((point[0] - self.basic_length * k_45, point[1] + self.basic_length * k_45))

                len_valid_p = []
                for i in range(len(possible_p)):
                    possible_p[i] = (int(possible_p[i][0]), int(possible_p[i][1]))
                    if possible_p[i][0]>=0 and possible_p[i][0]<self.geometry().width() and possible_p[i][1]>=0 and possible_p[i][1]<self.geometry().height():
                        len_valid_p.append(possible_p[i])

                valid_p = []
                for possi in len_valid_p:
                    exist_flag = False
                    for p in points:
                        if tooClose(p, possi, 5):
                            exist_flag = True
                            break
                    if exist_flag==False:
                        valid_p.append(possi)

                points += valid_p

        return points

    def paintEvent(self, QPaintEvent):
        qp = QPainter(self)
        self.draw_all_points(qp)
        self.draw_all_tri(qp)

    def mousePressEvent(self, QMouseEvent):
        '''print("x:", QMouseEvent.x(), "y:", QMouseEvent.y())
        self.points.append((QMouseEvent.x(), QMouseEvent.y()))
        self.repaint()'''
        press_x = QMouseEvent.x()
        press_y = QMouseEvent.y()

        repaint_flag = False
        for point in self.points:
            if tooClose(point, (press_x, press_y), 10):
                self.pressed_point.append(point)
                repaint_flag = True
        if repaint_flag==True:
            self.repaint()

        if len(self.pressed_point)==3:
            pressed_point = self.pressed_point
            self.pressed_point = []
            dis1 = arc_len((pressed_point[0][0]-pressed_point[1][0], pressed_point[0][1]-pressed_point[1][1]))
            dis2 = arc_len((pressed_point[1][0] - pressed_point[2][0],
                            pressed_point[1][1] - pressed_point[2][1]))
            dis3 = arc_len((pressed_point[2][0] - pressed_point[0][0],
                            pressed_point[2][1] - pressed_point[0][1]))
            basic_len = 0
            basic_len_sqrt2 = 0
            for dis in [dis1, dis2, dis3]:
                if num_tooClose(dis, self.basic_length, self.basic_length/8):
                    basic_len += 1
                elif num_tooClose(dis, self.basic_length*np.sqrt(2), self.basic_length/8):
                    basic_len_sqrt2 += 1

            if basic_len==2 and basic_len_sqrt2==1:
                self.choosed_tri.append(Triangle(*pressed_point))
                self.repaint()
            else:
                QMessageBox.information(self, "提示", "请点击单位长度距离的三个点", QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)


    def draw_all_points(self, qp):
        brush = QBrush(Qt.SolidPattern)
        brush.setColor(Qt.black)
        qp.setBrush(brush)

        radius = 4
        for p in self.points:
            qp.drawEllipse(p[0]-radius, p[1]-radius, radius*2, radius*2)

        brush.setColor(Qt.red)
        qp.setBrush(brush)
        for c_p in self.pressed_point:
            qp.drawEllipse(c_p[0] - radius, c_p[1] - radius, radius * 2, radius * 2)

    def draw_all_tri(self, qp):
        brush = QBrush(Qt.SolidPattern)
        brush.setColor(Qt.yellow)
        qp.setBrush(brush)

        for tri in self.choosed_tri:
            points = []
            for p in tri.ps:
                points.append(QPoint(p[0], p[1]))
            tri_poly = QPolygon(points)
            qp.drawPolygon(tri_poly)

    #撤销按钮
    def revocate(self):
        if len(self.choosed_tri)>0:
            self.choosed_tri.pop(-1)
        self.pressed_point = []
        self.repaint()

    def flipPoints(self):
        reply = QMessageBox.information(self, "警告", "是否要翻转坐标点？已画的三角形将丢失", QMessageBox.Yes,
                                QMessageBox.No)

        if reply==QMessageBox.No:
            return

        self.finish_pic = None
        self.choosed_tri = []
        self.flip_point = not self.flip_point
        if self.flip_point==False:
            self.points = self.init_points
            self.init_tris()
        else:
            self.points = self.diff_points
            self.init_tris_flip()

        self.repaint()

    def clear_tri(self):
        reply = QMessageBox.information(self, "警告", "是否要清除？已画的三角形将丢失", QMessageBox.Yes,
                                        QMessageBox.No)

        if reply==QMessageBox.No:
            return
        self.choosed_tri = []

    def finish(self, qlabel, parent, small_tri_num=16):
        width = self.geometry().width()
        height = self.geometry().height()
        img = np.ndarray(shape=[height, width, 3],dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j]=np.array([255, 255, 255])

        for tri in self.choosed_tri:
            cv2.fillPoly(img, pts=[np.array(tri.ps)], color=0)
        #cv2.imwrite("finish.jpg", img)

        grey_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        black_area = ((255 - grey_img) // 255).sum()
        if self.flip_point == False:
            black_area /= 1.027
        else:
            black_area /= 1.010
        #print("black area:", black_area)
        #print("small_tri_num:", small_tri_num)
        #print("correct area:", small_tri_num*(self.basic_length**2)/2)
        if not num_tooClose(black_area, small_tri_num*(self.basic_length**2)/2, threhold = (self.basic_length**2)/4):
            QMessageBox.information(self, "警告", "所画区域面积大小与输入的七巧板片数不相符", QMessageBox.Ok)
            return

        '''tris = []
        for tri in self.choosed_tri:
            tris.append(tri.ps)
        print(tris)'''

        self.finish_pic = img
        self.showImage(qlabel, img)
        qlabel.setVisible(True)
        self.setVisible(False)
        parent.repaint()


    def showImage(self, qlabel, img):
        size = (int(qlabel.width()), int(qlabel.height()))
        shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        # cv2.imshow('img', shrink)
        shrink = cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB)
        self.QtImg = QtGui.QImage(shrink.data,
                                  shrink.shape[1],
                                  shrink.shape[0],
                                  shrink.shape[1]*3,
                                  QtGui.QImage.Format_RGB888)

        qlabel.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))


if __name__ == "__main__":
    app = QApplication([])
    ui = MainWindow()
    sys.exit(app.exec_())