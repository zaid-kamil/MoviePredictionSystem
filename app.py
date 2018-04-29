
from PyQt5 import QtCore, QtGui, QtWidgets

import pandas as pd
import sys


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(517, 603)
        self.btnBrowseDataSet = QtWidgets.QToolButton(Dialog)
        self.btnBrowseDataSet.setGeometry(QtCore.QRect(330, 90, 41, 31))
        self.btnBrowseDataSet.setObjectName("btnBrowseDataSet")
        self.editDatasetPath = QtWidgets.QPlainTextEdit(Dialog)
        self.editDatasetPath.setGeometry(QtCore.QRect(20, 160, 311, 31))
        self.editDatasetPath.setObjectName("editDatasetPath")
        self.editEmailPath = QtWidgets.QPlainTextEdit(Dialog)
        self.editEmailPath.setGeometry(QtCore.QRect(20, 90, 311, 31))
        self.editEmailPath.setObjectName("editEmailPath")
        self.btnBrowseEmail = QtWidgets.QToolButton(Dialog)
        self.btnBrowseEmail.setGeometry(QtCore.QRect(330, 160, 41, 31))
        self.btnBrowseEmail.setObjectName("btnBrowseEmail")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(20, 60, 251, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.btntraining = QtWidgets.QPushButton(Dialog)
        self.btntraining.setGeometry(QtCore.QRect(380, 90, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.btntraining.setFont(font)
        self.btntraining.setObjectName("btntraining")
        self.btnclassify = QtWidgets.QPushButton(Dialog)
        self.btnclassify.setGeometry(QtCore.QRect(380, 160, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.btnclassify.setFont(font)
        self.btnclassify.setObjectName("btnclassify")
        self.Head = QtWidgets.QLabel(Dialog)
        self.Head.setGeometry(QtCore.QRect(20, 0, 331, 61))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(18)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.Head.setFont(font)
        self.Head.setObjectName("Head")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 135, 251, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.datasetlabel = QtWidgets.QLabel(Dialog)
        self.datasetlabel.setGeometry(QtCore.QRect(330, 60, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.datasetlabel.setFont(font)
        self.datasetlabel.setObjectName("datasetlabel")
        self.emaillabel = QtWidgets.QLabel(Dialog)
        self.emaillabel.setGeometry(QtCore.QRect(330, 140, 101, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.emaillabel.setFont(font)
        self.emaillabel.setObjectName("emaillabel")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(20, 195, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(14, 220, 491, 371))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.btnsetting = QtWidgets.QPushButton(self.widget)
        self.btnsetting.setObjectName("btnsetting")
        self.gridLayout.addWidget(self.btnsetting, 1, 1, 1, 1)
        self.btnsave = QtWidgets.QPushButton(self.widget)
        self.btnsave.setObjectName("btnsave")
        self.gridLayout.addWidget(self.btnsave, 1, 4, 1, 1)
        self.btnclear = QtWidgets.QPushButton(self.widget)
        self.btnclear.setObjectName("btnclear")
        self.gridLayout.addWidget(self.btnclear, 1, 0, 1, 1)
        self.btnquit = QtWidgets.QPushButton(self.widget)
        self.btnquit.setObjectName("btnquit")
        self.gridLayout.addWidget(self.btnquit, 1, 2, 1, 1)
        self.btnopen = QtWidgets.QPushButton(self.widget)
        self.btnopen.setObjectName("btnopen")
        self.gridLayout.addWidget(self.btnopen, 1, 3, 1, 1)
        self.tableView = QtWidgets.QTableView(self.widget)
        self.tableView.setObjectName("tableView")
        self.gridLayout.addWidget(self.tableView, 0, 0, 1, 5)
        self.editDatasetPath.raise_()
        self.editEmailPath.raise_()
        self.btnBrowseDataSet.raise_()
        self.btnBrowseEmail.raise_()
        self.label_2.raise_()
        self.btntraining.raise_()
        self.btnclassify.raise_()
        self.Head.raise_()
        self.label.raise_()
        self.datasetlabel.raise_()
        self.emaillabel.raise_()
        self.label_3.raise_()
        self.btnsave.raise_()
        self.btnclear.raise_()
        self.btnsetting.raise_()
        self.btnquit.raise_()
        self.btnopen.raise_()
        self.btntraining.clicked.connect(self.excute_training)
        self.btnquit.clicked.connect(self.quit)
        self.btnclear.clicked.connect(self.clear)
        self.btnsetting.clicked.connect(self.setting)
        self.btnsave.clicked.connect(self.save)
        self.btnclassify.clicked.connect(self.classify)
        self.btnBrowseDataSet.clicked.connect(self.openfiledialog)
        self.dlg = Dialog
        self.btntraining.setDisabled(True)
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.btnBrowseDataSet.setText(_translate("Dialog", "..."))
        self.btnBrowseEmail.setText(_translate("Dialog", "..."))
        self.label_2.setText(_translate("Dialog", "Email dataset path"))
        self.btntraining.setText(_translate("Dialog", "Training"))
        self.btnclassify.setText(_translate("Dialog", "Classify"))
        self.Head.setText(_translate("Dialog", "Email classification"))
        self.label.setText(_translate("Dialog", "Select a file containing email"))
        self.datasetlabel.setText(_translate("Dialog", "Status "))
        self.emaillabel.setText(_translate("Dialog", "Status"))
        self.label_3.setText(_translate("Dialog", "Output"))
        self.btnsetting.setText(_translate("Dialog", "Setting"))
        self.btnsave.setText(_translate("Dialog", "Save"))
        self.btnclear.setText(_translate("Dialog", "Clear"))
        self.btnquit.setText(_translate("Dialog", "Quit"))
        self.btnopen.setText(_translate("Dialog", "Open Notebook"))

    def excute_training(self):
        self.datasetlabel.setText("Training...")
        training.train_email_dataset(self.df)

    def quit(self):
        sys.exit(0)

    def clear(self):
        self.editEmailPath.setPlainText("")
        self.btntraining.setDisabled(True)

    def setting(self):
        print("setting work")

    def save(self):
        print("save work")

    def classify(self):
        print("classify")

    def openfiledialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self.dlg, "Open CSV", "", "CSV Files (*.csv)", options=options)
        if filename != '':
            self.editEmailPath.setPlainText(filename)
            self.df = pd.read_csv(filename)
            model = PandasModel(self.df)
            self.tableView.setModel(model)
            self.btntraining.setEnabled(True)


class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[col]
        return None



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
