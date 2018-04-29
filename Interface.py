# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MoviePredictionUI.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import sys


class Ui_MoviePrediction(object):
    def setupUi(self, MoviePrediction):
        MoviePrediction.setObjectName("MoviePrediction")
        MoviePrediction.resize(600, 400)
        self.centralwidget = QtWidgets.QWidget(MoviePrediction)
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Path = QtWidgets.QLineEdit(self.centralwidget)
        self.Path.setObjectName("Path")
        self.horizontalLayout.addWidget(self.Path)
        self.Load = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Load.sizePolicy().hasHeightForWidth())
        self.Load.setSizePolicy(sizePolicy)
        self.Load.setStyleSheet(stylesheet(self))
        self.Load.clicked.connect(self.loadfile)
        self.Load.setObjectName("Load")
        self.horizontalLayout.addWidget(self.Load)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.Plot = QtWidgets.QPushButton(self.centralwidget)
        self.Plot.setStyleSheet(stylesheet(self))
        self.Plot.setObjectName("Plot")
        self.verticalLayout.addWidget(self.Plot)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.Clean_Vis = QtWidgets.QPushButton(self.centralwidget)
        self.Clean_Vis.setStyleSheet(stylesheet(self))
        self.Clean_Vis.setObjectName("Clean_Vis")
        self.verticalLayout.addWidget(self.Clean_Vis)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.Displahy_Cor = QtWidgets.QPushButton(self.centralwidget)
        self.Displahy_Cor.setStyleSheet(stylesheet(self))
        self.Displahy_Cor.setObjectName("Displahy_Cor")
        self.verticalLayout_2.addWidget(self.Displahy_Cor)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.Sim_Cor = QtWidgets.QPushButton(self.centralwidget)
        self.Sim_Cor.setStyleSheet(stylesheet(self))
        self.Sim_Cor.setObjectName("Sim_Cor")
        self.verticalLayout_2.addWidget(self.Sim_Cor)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        MoviePrediction.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MoviePrediction)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MoviePrediction.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MoviePrediction)
        self.statusbar.setObjectName("statusbar")
        MoviePrediction.setStatusBar(self.statusbar)
        self.actionSettings = QtWidgets.QAction(MoviePrediction)
        self.actionSettings.setObjectName("actionSettings")
        self.actionExit = QtWidgets.QAction(MoviePrediction)
        self.actionExit.triggered.connect(self.exit)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.actionSettings)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MoviePrediction)
        QtCore.QMetaObject.connectSlotsByName(MoviePrediction)

    def retranslateUi(self, MoviePrediction):
        _translate = QtCore.QCoreApplication.translate
        MoviePrediction.setWindowTitle(_translate("MoviePrediction", "Movie Prediction System"))
        self.Load.setText(_translate("MoviePrediction", "Load"))
        self.label.setText(_translate("MoviePrediction", "% of movie per decade"))
        self.Plot.setText(_translate("MoviePrediction", "Plot Graph"))
        self.label_2.setText(_translate("MoviePrediction", "Threshold for keyword deletion"))
        self.Clean_Vis.setText(_translate("MoviePrediction", "Clean & Visualize"))
        self.label_3.setText(_translate("MoviePrediction", "Correlation Coefficients"))
        self.Displahy_Cor.setText(_translate("MoviePrediction", "Display Correlation"))
        self.label_4.setText(_translate("MoviePrediction", "Correlation between keyword"))
        self.Sim_Cor.setText(_translate("MoviePrediction", "Similar Correlation"))
        self.menuFile.setTitle(_translate("MoviePrediction", "File"))
        self.actionSettings.setText(_translate("MoviePrediction", "Settings"))
        self.actionExit.setText(_translate("MoviePrediction", "Exit"))

    def loadfile(self):
        filepath = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, "Open CSV File", "", "CSV files (*.csv)")
        if filepath is not None:
            self.Path.setText(filepath[0])

    def exit(self):
        sys.exit(0)


def stylesheet(self):
    return """
	QPushButton {
		background-color: rgba(50, 50, 50, 150);
		color: rgb(255, 255, 255);
	}
	"""


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MoviePrediction = QtWidgets.QMainWindow()
    ui = Ui_MoviePrediction()
    ui.setupUi(MoviePrediction)
    MoviePrediction.show()
    sys.exit(app.exec_())
