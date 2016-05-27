# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dfp_gui.ui'
#
# Created: Sun May 15 21:31:37 2016
#      by: PyQt4 UI code generator 4.11.3
#
# WARNING! All changes made in this file will be lost!
from function import Function
from figure import Figure
from fmindfp import fmindfp
import numpy as np

from multiprocessing import Process

def plot_graph(fig):
    fig.show()

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(798, 600)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.formLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(20, 20, 371, 191))
        self.formLayoutWidget.setObjectName(_fromUtf8("formLayoutWidget"))
        self.formLayout = QtGui.QFormLayout(self.formLayoutWidget)
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setMargin(0)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.cost_function = QtGui.QLineEdit(self.formLayoutWidget)
        self.cost_function.setObjectName(_fromUtf8("cost_function"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.cost_function)
        self.maxIteracjiLabel = QtGui.QLabel(self.formLayoutWidget)
        self.maxIteracjiLabel.setObjectName(_fromUtf8("maxIteracjiLabel"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.maxIteracjiLabel)
        self.max_iter = QtGui.QLineEdit(self.formLayoutWidget)
        self.max_iter.setObjectName(_fromUtf8("max_iter"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.max_iter)
        self.eplylonLabel = QtGui.QLabel(self.formLayoutWidget)
        self.eplylonLabel.setObjectName(_fromUtf8("eplylonLabel"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.eplylonLabel)
        self.epsilon = QtGui.QLineEdit(self.formLayoutWidget)
        self.epsilon.setObjectName(_fromUtf8("epsilon"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.epsilon)
        self.bDRNiczkowaniaLabel = QtGui.QLabel(self.formLayoutWidget)
        self.bDRNiczkowaniaLabel.setObjectName(_fromUtf8("bDRNiczkowaniaLabel"))
        self.formLayout.setWidget(5, QtGui.QFormLayout.LabelRole, self.bDRNiczkowaniaLabel)
        self.div_err = QtGui.QLineEdit(self.formLayoutWidget)
        self.div_err.setObjectName(_fromUtf8("div_err"))
        self.formLayout.setWidget(5, QtGui.QFormLayout.FieldRole, self.div_err)
        self.start_btn = QtGui.QPushButton(self.formLayoutWidget)
        self.start_btn.setObjectName(_fromUtf8("start_btn"))
        self.formLayout.setWidget(6, QtGui.QFormLayout.FieldRole, self.start_btn)
        self.initial_cond = QtGui.QLineEdit(self.formLayoutWidget)
        self.initial_cond.setObjectName(_fromUtf8("initial_cond"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.initial_cond)
        self.funkcjaLabel_2 = QtGui.QLabel(self.formLayoutWidget)
        self.funkcjaLabel_2.setObjectName(_fromUtf8("funkcjaLabel_2"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.funkcjaLabel_2)
        self.maxIteracjiLabel_2 = QtGui.QLabel(self.formLayoutWidget)
        self.maxIteracjiLabel_2.setObjectName(_fromUtf8("maxIteracjiLabel_2"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.maxIteracjiLabel_2)
        self.txt_browser = QtGui.QTextBrowser(self.centralwidget)
        self.txt_browser.setGeometry(QtCore.QRect(20, 220, 371, 301))
        self.txt_browser.setObjectName(_fromUtf8("txt_browser"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtGui.QToolBar(MainWindow)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.maxIteracjiLabel.setText(_translate("MainWindow", "max iteracji", None))
        self.eplylonLabel.setText(_translate("MainWindow", "epsylon", None))
        self.bDRNiczkowaniaLabel.setText(_translate("MainWindow", "błąd różniczkowania", None))
        self.start_btn.setText(_translate("MainWindow", "START", None))
        self.funkcjaLabel_2.setText(_translate("MainWindow", "funkcja celu", None))
        self.maxIteracjiLabel_2.setText(_translate("MainWindow", "warunki początkowe", None))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar", None))

        self.start_btn.clicked.connect(self.start_button_clicked)


    def start_button_clicked(self):
        cost_fun = str(self.cost_function.text())
        max_it = int(self.max_iter.text())
        eps = self.epsilon.text()
        div_err = float(self.div_err.text())
        x0 = self.initial_cond.text()

        x0 = x0.replace(" ", "")
        x0 = x0.split(",")
        if len(x0) == 1:
            x0 = float(x0[0])
        else:
            x0 = tuple(float(i) for i in x0)


        eps = eps.replace(" ", "")
        eps = eps.split(",")
        if len(eps) == 3:
            eps = tuple(float(i) for i in eps)
        else:
            eps = tuple(float(eps[0]) for i in range(3))

        #ustawic tolerancje!!!!!! w funckacji
        #gtol=1e-05, xtol=1e-09, fxtol=1e-09
        opts = {'gtol': eps[0],
                'xtol': eps[1],
                'fxtol': eps[2],
                'epsilon': div_err,
                'maxiter': max_it
        }

        fun = Function(cost_fun)
        x = fmindfp(fun, x0, disp=True, **opts)

        #normalizacjay wjscia:
        output = np.asarray(x[2])
        output = output.tolist()
        text = '\n'.join(output)
        self.txt_browser.setText(text)


        vec = x[1]
        vec = np.asanyarray(vec)
        print(vec.shape)
        print(vec)


        fig = Figure(fun, vec)
        fig.show()
        #p = Process(target=plot_graph, args=fig)
        #p.start()
        #p.join()


import sys

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ex = Ui_MainWindow()
    ex.show()
    sys.exit(app.exec_())
