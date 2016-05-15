# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dfp_gui.ui'
#
# Created: Sun May 15 20:06:41 2016
#      by: PyQt4 UI code generator 4.11.3
#
# WARNING! All changes made in this file will be lost!

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

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(798, 600)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.formLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(40, 30, 341, 155))
        self.formLayoutWidget.setObjectName(_fromUtf8("formLayoutWidget"))
        self.formLayout = QtGui.QFormLayout(self.formLayoutWidget)
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setMargin(0)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.funkcjaLabel_2 = QtGui.QLabel(self.formLayoutWidget)
        self.funkcjaLabel_2.setObjectName(_fromUtf8("funkcjaLabel_2"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.funkcjaLabel_2)
        self.cost_function_linedt = QtGui.QLineEdit(self.formLayoutWidget)
        self.cost_function_linedt.setObjectName(_fromUtf8("cost_function_linedt"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.cost_function_linedt)
        self.maxIteracjiLabel = QtGui.QLabel(self.formLayoutWidget)
        self.maxIteracjiLabel.setObjectName(_fromUtf8("maxIteracjiLabel"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.maxIteracjiLabel)
        self.max_iter_linedt = QtGui.QLineEdit(self.formLayoutWidget)
        self.max_iter_linedt.setObjectName(_fromUtf8("max_iter_linedt"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.max_iter_linedt)
        self.eplylonLabel = QtGui.QLabel(self.formLayoutWidget)
        self.eplylonLabel.setObjectName(_fromUtf8("eplylonLabel"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.eplylonLabel)
        self.epsilon_linedt = QtGui.QLineEdit(self.formLayoutWidget)
        self.epsilon_linedt.setObjectName(_fromUtf8("epsilon_linedt"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.epsilon_linedt)
        self.bDRNiczkowaniaLabel = QtGui.QLabel(self.formLayoutWidget)
        self.bDRNiczkowaniaLabel.setObjectName(_fromUtf8("bDRNiczkowaniaLabel"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.bDRNiczkowaniaLabel)
        self.div_err_linedt = QtGui.QLineEdit(self.formLayoutWidget)
        self.div_err_linedt.setObjectName(_fromUtf8("div_err_linedt"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.div_err_linedt)
        self.start_btn = QtGui.QPushButton(self.formLayoutWidget)
        self.start_btn.setObjectName(_fromUtf8("start_btn"))
        self.formLayout.setWidget(5, QtGui.QFormLayout.FieldRole, self.start_btn)
        self.output_txtbrow = QtGui.QTextBrowser(self.centralwidget)
        self.output_txtbrow.setGeometry(QtCore.QRect(40, 190, 341, 301))
        self.output_txtbrow.setObjectName(_fromUtf8("output_txtbrow"))
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
        self.funkcjaLabel_2.setText(_translate("MainWindow", "funkcja celu", None))
        self.maxIteracjiLabel.setText(_translate("MainWindow", "max iteracji", None))
        self.eplylonLabel.setText(_translate("MainWindow", "epsylon", None))
        self.bDRNiczkowaniaLabel.setText(_translate("MainWindow", "błąd różniczkowania", None))
        self.start_btn.setText(_translate("MainWindow", "START", None))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar", None))

