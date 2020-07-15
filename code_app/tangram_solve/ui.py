from PyQt5.QtWidgets import QTabWidget, QMainWindow, QDesktopWidget, QApplication
from PyQt5.QtGui import QIcon
import sys
from ui_userDefine import Ui_userDefine
from ui_basic import Ui_basic
from ui_general import Ui_general

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1200, 750)
        self.center()
        self.setWindowTitle('Tangram')
        self.setWindowIcon(QIcon("tangram.ico"))

        self.tabW = QTabWidget(parent=self)
        ui_userD = Ui_userDefine()
        ui_basic = Ui_basic()
        ui_general = Ui_general()

        self.tabW.addTab(ui_basic, "求解七巧板")
        self.tabW.addTab(ui_userD, "用户自定义")
        self.tabW.addTab(ui_general, "任意图片拼接")
        self.tabW.resize(1200,750)
        ui_userD.initUI()
        ui_basic.initUI()
        ui_general.initUI()

        self.show()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2)

if __name__ == "__main__":
    app = QApplication([])
    ui = MainWindow()
    sys.exit(app.exec_())