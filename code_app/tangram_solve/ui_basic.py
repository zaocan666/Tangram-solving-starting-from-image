from PyQt5.QtWidgets import QMainWindow, QApplication, QDesktopWidget, QLabel, QComboBox, QPushButton, QMessageBox, QWidget
from PyQt5 import QtGui
from PyQt5.QtCore import QBasicTimer
import sys
import cv2
from game_class import Solver
import time
import copy
import numpy


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui_basic = Ui_basic()
        self.ui_basic.setParent(self)
        self.ui_basic.initUI()

        self.resize(1200, 700)
        self.center()
        self.setWindowTitle('Tangram')
        self.show()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2)

class Ui_basic(QWidget):
    def __init__(self):
        super().__init__()

    def initUI(self):
        self.resize(1200, 700)

        self.pic_list=["tangram%d"%i for i in range(1,20+1)]

        self.pic_area = QLabel(parent=self) #图形显示区域
        self.pic_area.resize(800, 600)
        self.pic_area.move(30,50)
        self.current_img = cv2.imread("tangrams\\tangram1.png", cv2.COLOR_BGRA2BGR)
        self.showImage(self.pic_area, self.current_img)

        pic_choose_label = QLabel(self)
        pic_choose_label.move(self.pic_area.geometry().x()+self.pic_area.geometry().width()+30, self.pic_area.geometry().y()+20)
        pic_choose_label.setText("选择图片：")
        self.pic_choose_combo = QComboBox(self)
        self.pic_choose_combo.move(pic_choose_label.geometry().x()+pic_choose_label.geometry().width()+30, pic_choose_label.geometry().y())
        self.pic_choose_combo.resize(150,self.pic_choose_combo.geometry().height())
        self.pic_choose_combo.addItems(self.pic_list)
        self.pic_choose_combo.currentIndexChanged.connect(self.pic_change)

        middle_x = (pic_choose_label.geometry().x() + self.pic_choose_combo.geometry().x() + self.pic_choose_combo.geometry().width()) / 2
        self.solver_class = Solve_frame(self, size=((middle_x-pic_choose_label.geometry().x())*2, 600), pic_area=self.pic_area)
        self.solver_class.move(pic_choose_label.geometry().x(), pic_choose_label.geometry().y()+pic_choose_label.geometry().height()+40)
        self.solver_class.solve_problem_button.pressed.connect(lambda :self.solver_class.solve_button_pressed(path = "tangrams\\" + self.pic_choose_combo.currentText() + ".png"))

    def pic_change(self):
        self.solver_class.solver=None
        self.solver_class.playing_index = -1
        self.solver_class.timer.stop()

        current_text = self.pic_choose_combo.currentText()
        self.current_img = cv2.imread("tangrams\\"+current_text+".png", cv2.COLOR_BGRA2BGR)
        self.showImage(self.pic_area, self.current_img)
        #print(current_text)

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

#将解答按钮、解答过程播放按钮和速度选择框封装成一个类，供三个标签页重复使用
class Solve_frame(QWidget):
    def __init__(self, parent, size, pic_area):
        super().__init__()

        self.playing_index = -1
        self.solver = None
        self.pic_area = pic_area
        self.img_path = "tangrams\\tangram1.png"
        self.problem_solving = False

        self.setParent(parent)
        self.resize(*size)
        self.solve_problem_button = QPushButton(parent=self)
        self.solve_problem_button.setText("解答")
        middle_x = self.geometry().width()/2
        self.solve_problem_button.move(middle_x-self.solve_problem_button.geometry().width()/2, 0)
        #self.solve_problem_button.pressed.connect(self.solve_button_pressed)

        self.solve_test = QLabel(parent=self) #解答过程中的信息显示
        self.solve_test.setText("正在解答。。。")
        self.solve_test.resize(250, self.solve_test.height())
        self.solve_test.move(middle_x - self.solve_test.geometry().width() / 2,
                             self.solve_problem_button.geometry().y() + self.solve_problem_button.geometry().height() + 20)
        self.solve_test.setHidden(True)


        speed_choose_label = QLabel(self)
        speed_choose_label.move(0, self.solve_test.geometry().y() + 40)
        speed_choose_label.setText("播放速度：")
        self.play_speed_combo = QComboBox(self)
        self.play_speed_combo.move(speed_choose_label.geometry().x() + speed_choose_label.geometry().width() + 30,
                                   speed_choose_label.geometry().y())
        self.play_speed_combo.addItems(["高速", "中速", "慢速"])

        play_button = QPushButton(self)
        play_button.setText("播放解答过程")
        play_button.move(middle_x - play_button.geometry().width() / 2,
                         self.play_speed_combo.geometry().y() + self.play_speed_combo.geometry().height() + 40)
        play_button.pressed.connect(self.play_button_pressed)

        self.timer = QBasicTimer()

    def solve_button_pressed(self, path, pieces_num = [2, 1, 1, 1, 2]):
        if self.problem_solving:
            return
        if type(path) != str and type(path)!=numpy.ndarray:
            QMessageBox.information(self, "警告", "还未形成图像！", QMessageBox.Ok)
            return

        self.problem_solving = True
        self.playing_index = -1
        self.solve_test.setHidden(False)
        self.timer.stop()
        self.repaint()

        start_time = time.time()
        #path = "tangrams\\" + self.parent().pic_choose_combo.currentText() + ".png"
        solver = Solver(path, pieces_num = pieces_num, output_line = self.solve_test)
        solver.solve_dfs2()
        end_time = time.time()

        QMessageBox.information(self, "提示", "完成解答，用时：%.3f s，经%d步" % (end_time - start_time, len(solver.trace)),
                                QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)
        self.showImage(self.pic_area, img=solver.graph.current_graph)

        self.solver = solver
        self.problem_solving = False
        self.solve_test.setHidden(True)

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

    def play_button_pressed(self):
        if self.solver == None:
            QMessageBox.information(self, "提示", "未完成解答，请先按解答按钮并等待", QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            return

        speed_text = self.play_speed_combo.currentText()
        self.playing_index = 0
        if speed_text == "高速":
            self.timer.start(300, self)
        elif speed_text == "中速":
            self.timer.start(800, self)
        else:
            self.timer.start(1500, self)

    def timerEvent(self, event):
        if event.timerId() == self.timer.timerId():
            if self.playing_index < len(self.solver.trace) and self.solver != None:
                node_num = self.solver.trace[self.playing_index]
                node = self.solver.open_all_list[node_num]
                self.playing_index += 1

                graph = copy.deepcopy(self.solver.graph.init_color_img)
                self.solver.draw_current_graph(graph=graph, node=node)
                self.showImage(self.pic_area, graph)
                # print("up",self.playing_index)
        else:
            super(Solve_frame, self).timerEvent(event)

if __name__ == "__main__":
    app = QApplication([])
    ui = MainWindow()
    sys.exit(app.exec_())