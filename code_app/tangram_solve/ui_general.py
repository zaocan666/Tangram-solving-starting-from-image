from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QWidget, QLabel, QPushButton, QFileDialog, QApplication, QLineEdit, QMessageBox
from PyQt5 import QtGui
import sys
import cv2
import numpy
import copy
from generalization import binaryzation, Transform_img
from ui_basic import Solve_frame
from game_class import Solver, Node_open_list

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui_general = Ui_general()
        self.ui_general.setParent(self)
        self.ui_general.initUI()

        self.resize(1200, 700)
        self.center()
        self.setWindowTitle('Tangram')
        self.show()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2)

class Ui_general(QWidget):
    def __init__(self):
        super().__init__()
        self.grey_img = None
        self.binary_img = None
        self.transform_img = None
        self.small_tri_num = 0

    def initUI(self):
        self.resize(1200, 700)

        self.pic_area = QLabel(parent=self)
        self.pic_area.resize(800, 600)
        self.pic_area.move(30, 50)

        import_button = QPushButton(parent=self)
        import_button.setText("加载图片")
        import_button.move(self.pic_area.x() + self.pic_area.width()+40, self.pic_area.y() + 40)
        import_button.pressed.connect(self.import_pic)

        grey_label = QLabel(parent=self)
        grey_label.setText("填充区域灰度值范围：")
        grey_label.move(import_button.x(), import_button.y()+import_button.height()+40)
        self.grey_line_min = QLineEdit(parent=self)
        self.grey_line_min.setText("0")
        self.grey_line_min.move(grey_label.x()+10, grey_label.y()+grey_label.height()+10)
        self.grey_line_min.resize(40,self.grey_line_min.height())
        zhi_label = QLabel(parent=self)
        zhi_label.setText("至")
        zhi_label.resize(40, zhi_label.height())
        zhi_label.move(self.grey_line_min.x()+self.grey_line_min.width()+5, self.grey_line_min.y())
        self.grey_line_max = QLineEdit(parent=self)
        self.grey_line_max.setText("200")
        self.grey_line_max.move(zhi_label.x() + zhi_label.width() + 5, zhi_label.y())
        self.grey_line_max.resize(40, self.grey_line_max.height())

        binary_button = QPushButton(parent=self)
        binary_button.setText("二值化")
        binary_button.move(import_button.x(), self.grey_line_max.y() + self.grey_line_max.height() + 40)
        binary_button.pressed.connect(self.binary_show)

        general_button = QPushButton(parent=self)
        general_button.setText("规则化")
        general_button.move(import_button.x(), binary_button.y() + binary_button.height() + 40)
        general_button.pressed.connect(self.general_show)

        self.solver_class = Solve_frame(self, size=(300, 200), pic_area=self.pic_area)
        self.solver_class.move(general_button.x(), general_button.y() + general_button.height() + 50)
        self.solver_class.solve_problem_button.pressed.connect(self.solve_puzzle)

    def import_pic(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;*.png;*.bmp;;All Files(*)")
        if imgName=='':
            return

        self.solver_class.solver = None
        self.solver_class.playing_index = -1
        self.solver_class.timer.stop()

        self.grey_img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
        if type(self.grey_img) != numpy.ndarray:
            QMessageBox.information(self, "提示", "无法加载图片！", QMessageBox.Ok | QMessageBox.Close,
                                QMessageBox.Close)
            return
        self.showImage(self.pic_area, self.grey_img)

        self.transform_img = None
        self.binary_img = None

    def binary_show(self):
        if type(self.grey_img) != numpy.ndarray:
            QMessageBox.information(self, "提示", "请先加载图片！", QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            return

        threhold_low = int(self.grey_line_min.text())
        threhold_high = int(self.grey_line_max.text())
        self.binary_img = binaryzation(copy.deepcopy(self.grey_img), threhold_low, threhold_high)
        self.binary_img = cv2.resize(src=self.binary_img, dsize=(500, int(500 * self.binary_img.shape[0] / self.binary_img.shape[1])))
        self.showImage(self.pic_area, self.binary_img)

    def general_show(self):
        if type(self.binary_img) != numpy.ndarray:
            QMessageBox.information(self, "提示", "请先将图片二值化！", QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
        self.transform_img, self.small_tri_num = Transform_img(self.binary_img)
        self.showImage(self.pic_area, self.transform_img)


    def showImage(self, qlabel, img):
        img_copy = copy.deepcopy(img)
        size = (int(qlabel.width()), int(qlabel.height()))
        shrink = cv2.resize(img_copy, size, interpolation=cv2.INTER_AREA)
        # cv2.imshow('img', shrink)
        if len(img_copy.shape)==2:
            self.QtImg = QtGui.QImage(shrink.data,
                                      shrink.shape[1],
                                      shrink.shape[0],
                                      shrink.shape[1]*1,
                                      QtGui.QImage.Format_Grayscale8)
        else:
            shrink = cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB)
            self.QtImg = QtGui.QImage(shrink.data,
                                      shrink.shape[1],
                                      shrink.shape[0],
                                      shrink.shape[1] * 3,
                                      QtGui.QImage.Format_RGB888)

        qlabel.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

    def solve_puzzle(self):
        if self.small_tri_num == 0:
            QMessageBox.information(self, "提示", "请先将图片规则化！", QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

        pieces_num = self.try_pieces_num()
        color_img = cv2.cvtColor(self.transform_img, cv2.COLOR_GRAY2BGR)
        self.solver_class.solve_button_pressed(path=color_img, pieces_num=pieces_num)

    def try_pieces_num(self):
        pieces_num = [0 for i in range(5)]
        pieces_num[Solver.LARGE_TRIANGLE] = 2
        pieces_num[Solver.MIDDLE_TRIANGLE] = 2
        pieces_num[Solver.PARALLELOGRAM] = 2
        pieces_num[Solver.SQUARE] = 2
        pieces_num[Solver.SMALL_TRIANGLE] = int(self.small_tri_num - 4*pieces_num[Solver.LARGE_TRIANGLE]-2*pieces_num[Solver.MIDDLE_TRIANGLE]
        -2*pieces_num[Solver.PARALLELOGRAM]-2*pieces_num[Solver.SQUARE])

        color_img = cv2.cvtColor(self.transform_img, cv2.COLOR_GRAY2BGR)
        solver = Solver(color_img, pieces_num=pieces_num)

        init_node = Node_open_list(all_tri=copy.deepcopy(solver.graph.all_tri))
        solver.open_all_list.append(init_node)
        solver.open_list_num.append(0)

        #通过试错的方法得到图形上能摆放的板块数量

        for i in range(pieces_num[Solver.LARGE_TRIANGLE]):
            children_node = solver.place_large_triangle(solver.open_list_num[-1])
            if children_node == []:
                pieces_num[Solver.LARGE_TRIANGLE] -= 1
                pieces_num[Solver.SMALL_TRIANGLE] += 4
            len_open_list = len(solver.open_all_list)
            solver.open_all_list += children_node
            solver.open_list_num += [i for i in range(len_open_list, len(solver.open_all_list))]

        for i in range(pieces_num[Solver.MIDDLE_TRIANGLE]):
            children_node = solver.place_middle_triangle(solver.open_list_num[-1])
            if children_node == []:
                pieces_num[Solver.MIDDLE_TRIANGLE] -= 1
                pieces_num[Solver.SMALL_TRIANGLE] += 2
            len_open_list = len(solver.open_all_list)
            solver.open_all_list += children_node
            solver.open_list_num += [i for i in range(len_open_list, len(solver.open_all_list))]

        for i in range(pieces_num[Solver.PARALLELOGRAM]):
            children_node = solver.place_parallelogram(solver.open_list_num[-1])
            if children_node == []:
                pieces_num[Solver.PARALLELOGRAM] -= 1
                pieces_num[Solver.SMALL_TRIANGLE] += 2
            len_open_list = len(solver.open_all_list)
            solver.open_all_list += children_node
            solver.open_list_num += [i for i in range(len_open_list, len(solver.open_all_list))]

        for i in range(pieces_num[Solver.SQUARE]):
            children_node = solver.place_square(solver.open_list_num[-1])
            if children_node == []:
                pieces_num[Solver.SQUARE] -= 1
                pieces_num[Solver.SMALL_TRIANGLE] += 2
            len_open_list = len(solver.open_all_list)
            solver.open_all_list += children_node
            solver.open_list_num += [i for i in range(len_open_list, len(solver.open_all_list))]

        return pieces_num

if __name__ == "__main__":
    app = QApplication([])
    ui = MainWindow()
    sys.exit(app.exec_())