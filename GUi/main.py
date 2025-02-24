########################################################################
## IMPORTS
########################################################################
import sys
import os
from PySide6 import *
from PyQt5.QtWidgets import QApplication

########################################################################
# IMPORT GUI FILE
from interface import *
########################################################################

########################################################################
# IMPORT Custom widgets
from Custom_Widgets import *
########################################################################


########################################################################
## MAIN WINDOW CLASS
########################################################################
class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        ########################################################################
        # APPLY JSON STYLESHEET
        ########################################################################
        # self = QMainWindow class
        # self.ui = Ui_MainWindow / user interface class
        loadJsonStyle(self, self.ui)
        ########################################################################

        ########################################################################

        self.show()
        self.ui.settingsBtn.clicked.connect(lambda:self.ui.centerMenuContainer.expandMenu())
        self.ui.informationBtn.clicked.connect(lambda:self.ui.centerMenuContainer.expandMenu())
        self.ui.helpBtn.clicked.connect(lambda:self.ui.centerMenuContainer.expandMenu())

        self.ui.closeCenterMenuBtn.clicked.connect(lambda:self.ui.centerMenuContainer.collapseMenu())

        self.ui.moreBtn.clicked.connect(lambda:self.ui.rightMenuContainer.expandMenu())
        self.ui.profileBtn.clicked.connect(lambda:self.ui.rightMenuContainer.expandMenu())

        self.ui.closeRightMenuBtn.clicked.connect(lambda:self.ui.rightMenuContainer.collapseMenu())
        





########################################################################
## EXECUTE APP
########################################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
########################################################################
## END===>
########################################################################  
