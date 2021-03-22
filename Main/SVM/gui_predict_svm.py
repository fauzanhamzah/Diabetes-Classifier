#!/usr/bin/env python3

"""
GUI for diabetes prediction.
"""
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, QApplication, QMessageBox
from PyQt5.QtGui import QDoubleValidator, QFont
from PyQt5.QtCore import Qt, QLine

import train_svm
from aboutUi import Ui_About
from user_guide import Ui_user_guide


class Menu(QWidget):

    def __init__(self) -> None:
        super(Menu, self).__init__()

        self.user_guide = QtWidgets.QWidget()
        self.ui_about = Ui_About()
        self.ui_guide = Ui_user_guide()
        self.About = QtWidgets.QWidget()

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../IMG/diabetes.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        # header 1
        self.header1 = QLabel("Diabetes Classification\n Support Vector Machine")
        self.header1.setAlignment(Qt.AlignCenter)
        self.font1 = QFont()
        self.font1.setFamily("Calibri Light")
        self.font1.setPointSize(24)
        self.header1.setFont(self.font1)

        # gambar
        self.img_cover = QLabel("")
        self.img_cover.setAlignment(Qt.AlignCenter)
        self.img_cover.setObjectName("img_cover")
        self.img_cover.setPixmap(QtGui.QPixmap("../../IMG/cover.png"))
        self.img_cover.setGeometry(QtCore.QRect(340, 190, 281, 121))

        # btn_predict
        self.btn_predict = QPushButton("Predict")
        self.btn_predict.setFixedWidth(160)
        self.btn_predict.setFixedHeight(41)
        self.fontpred = QFont()
        self.fontpred.setFamily("Calibri Light")
        self.fontpred.setPointSize(10)
        self.btn_predict.setFont(self.fontpred)
        self.btn_predict.move(100, 70)
        self.predicon = QtGui.QIcon()
        self.predicon.addPixmap(QtGui.QPixmap("../../IMG/predictive.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_predict.setIcon(self.predicon)

        # btn_exit
        self.btn_exit = QPushButton("Exit")
        self.btn_exit.setFixedWidth(160)
        self.btn_exit.setFixedHeight(41)
        self.fontexit = QFont()
        self.fontexit.setFamily("Calibri Light")
        self.fontexit.setPointSize(10)
        self.btn_exit.setFont(self.fontpred)
        self.btn_exit.move(100, 70)
        self.exiticon = QtGui.QIcon()
        self.exiticon.addPixmap(QtGui.QPixmap("../../IMG/logout.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_exit.setIcon(self.exiticon)

        # btn_about
        self.btn_about = QPushButton("About")
        self.btn_about.setFixedWidth(160)
        self.btn_about.setFixedHeight(41)
        self.font_about = QFont()
        self.font_about.setFamily("Calibri Light")
        self.font_about.setPointSize(10)
        self.btn_about.setFont(self.font_about)
        self.btn_about.move(100, 70)
        self.about_icon = QtGui.QIcon()
        self.about_icon.addPixmap(QtGui.QPixmap("../../IMG/about.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_about.setIcon(self.about_icon)

        # btn_help
        self.btn_help = QPushButton("User Guide")
        self.btn_help.setFixedWidth(160)
        self.btn_help.setFixedHeight(41)
        self.btn_help.setFont(self.font_about)
        self.help_icon = QtGui.QIcon()
        self.help_icon.addPixmap(QtGui.QPixmap("../../IMG/help.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_help.setIcon(self.help_icon)

        self.h4 = QHBoxLayout()
        self.h5 = QHBoxLayout()
        self.h1 = QHBoxLayout()
        self.h2 = QHBoxLayout()
        self.h3 = QHBoxLayout()
        self.h6 = QHBoxLayout()

        self.v1_box = QVBoxLayout()
        self.v2_box = QVBoxLayout()
        self.v3_box = QVBoxLayout()
        self.awal_hbox = QHBoxLayout()
        self.final_hbox = QHBoxLayout()
        self.home()

    def home(self) -> None:
        self.v1_box.addWidget(self.header1)
        self.v1_box.setGeometry(QtCore.QRect())

        self.btn_predict.clicked.connect(Diabetes)
        self.btn_exit.clicked.connect(self.close)
        self.btn_about.clicked.connect(self.opnabout)
        self.btn_help.clicked.connect(self.opnhelp)

        self.v1_box.addWidget(self.img_cover)

        self.h1.addWidget(self.btn_exit)
        self.h1.addWidget(self.btn_predict)
        self.h1.addWidget(self.btn_about)
        self.h1.addWidget(self.btn_help)

        self.v1_box.addLayout(self.h1)
        self.final_hbox.addLayout(self.v1_box)
        self.setLayout(self.final_hbox)

    def mwindow(self) -> None:
        """ window features are set here and application is loaded into display"""
        self.setFixedSize(970, 500)
        self.setWindowTitle("Diabetes Classification")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../IMG/diabetes.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.show()

    def opnabout(self) -> None:
        self.ui_about.setupUi(self.About)
        self.About.show()

    def opnhelp(self) -> None:
        self.ui_guide.setupUi(self.user_guide)
        self.user_guide.show()


class Diabetes(QWidget):

    def __init__(self) -> None:
        super(Diabetes, self).__init__()

        self.setFixedSize(970, 500)
        self.setWindowTitle("Diabetes Classification")
        self.all_font = QFont()
        self.all_font.setFamily("Calibri Light")

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../IMG/diabetes.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.show()

        self.results = QLabel(" ")
        self.model_details = QLabel(
            "Klik untuk mengecek class data")
        self.model_details.setFont(QFont("Calibri Light", 10))

        self.details = QLabel(
            "Model yang digunakan yaitu Suport Vector Machine.\nAkurasi yang didapatkan dengan model ini yaitu: "
            "79%\nDataset yang digunakan yaitu PIMA Indians "
            "diabetes dataset dari UCI archive.")
        self.details.setFont(QFont("Calibri Light"))

        self.report_subhead = QLabel("About")
        self.report_subhead.setFont(QFont(self.all_font))

        self.h6 = QHBoxLayout()
        self.sub_head = QLabel("Data Pasien")
        self.sub_head.setFont(QFont("Calibri Light", 18, weight=QFont.Bold))
        self.l0 = QLineEdit()
        self.l1 = QLineEdit()
        self.l2 = QLineEdit()
        self.l3 = QLineEdit()
        self.l4 = QLineEdit()
        self.l5 = QLineEdit()
        self.t1 = QLabel("2-Hour serum insulin:")
        self.t2 = QLabel("Plasma glucose concentration:")
        self.t3 = QLabel("Age (years):")
        self.t4 = QLabel("Pregnancies:")
        self.t5 = QLabel("Body mass index:")
        self.r1 = QLabel("(14-850 mu U/ml)")
        self.r2 = QLabel("(44-200 mg/dL)")
        self.r3 = QLabel("(21-85 year)")
        self.r4 = QLabel("(0-17 pregnancies)")
        self.r5 = QLabel("(18-68 (kg/m)^2)")
        self.h1 = QHBoxLayout()
        self.h2 = QHBoxLayout()
        self.h3 = QHBoxLayout()
        self.h4 = QHBoxLayout()
        self.h5 = QHBoxLayout()

        self.exit = QPushButton("BACK")
        self.exit.setFixedWidth(160)
        self.exit.setFixedHeight(41)
        self.font_exit = QFont()
        self.font_exit.setFamily("Calibri Light")
        self.font_exit.setPointSize(10)
        self.exit.setFont(self.font_exit)
        self.exit_icon = QtGui.QIcon()
        self.exit_icon.addPixmap(QtGui.QPixmap("../../IMG/back.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.exit.setIcon(self.exit_icon)

        self.clbtn = QPushButton("CLEAR")
        self.clbtn.setFixedWidth(160)
        self.clbtn.setFixedHeight(41)
        self.font_clr = QFont()
        self.font_clr.setFamily("Calibri Light")
        self.font_clr.setPointSize(10)
        self.clbtn.setFont(self.font_clr)
        self.clr_icon = QtGui.QIcon()
        self.clr_icon.addPixmap(QtGui.QPixmap("../../IMG/clear.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.clbtn.setIcon(self.clr_icon)

        self.submit = QPushButton("SUBMIT")
        self.submit.setFixedWidth(160)
        self.submit.setFixedHeight(41)
        self.font_sub = QFont()
        self.font_sub.setFamily("Calibri Light")
        self.font_sub.setPointSize(10)
        self.submit.setFont(self.font_sub)
        self.sub_icon = QtGui.QIcon()
        self.sub_icon.addPixmap(QtGui.QPixmap("../../IMG/enter.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.submit.setIcon(self.sub_icon)

        self.v1_box = QVBoxLayout()
        self.v2_box = QVBoxLayout()
        self.final_hbox = QHBoxLayout()
        self.initui()

    def initui(self) -> None:
        """ The gui is created and widgets elements are set here """
        self.v1_box.addWidget(self.sub_head)
        self.v1_box.addSpacing(10)
        self.v1_box.setSpacing(5)
        self.l1.setValidator(QDoubleValidator())
        self.l2.setValidator(QDoubleValidator())
        self.l3.setValidator(QDoubleValidator())
        self.l4.setValidator(QDoubleValidator())
        self.l5.setValidator(QDoubleValidator())
        self.l1.setToolTip(
            "2-Hour serum insulin: \n 70-180 mg/dl")
        self.l2.setToolTip("(44-200 mg/dL)")
        self.l3.setToolTip("(21-85 year)")
        self.l4.setToolTip("0-17 pregnancies")
        self.l5.setToolTip("weight in kg/(height in m)^2 \n 16-68")
        self.l1.setFixedSize(40, 30)
        self.l2.setFixedSize(40, 30)
        self.l3.setFixedSize(40, 30)
        self.l4.setFixedSize(40, 30)
        self.l5.setFixedSize(40, 30)
        self.h1.addWidget(self.t1)
        self.h1.addWidget(self.l1)
        self.h1.addWidget(self.r1)
        self.v1_box.addLayout(self.h1)
        self.h2.addWidget(self.t2)
        self.h2.addWidget(self.l2)
        self.h2.addWidget(self.r2)
        self.v1_box.addLayout(self.h2)
        self.h3.addWidget(self.t3)
        self.h3.addWidget(self.l3)
        self.h3.addWidget(self.r3)
        self.v1_box.addLayout(self.h3)
        self.h4.addWidget(self.t4)
        self.h4.addWidget(self.l4)
        self.h4.addWidget(self.r4)
        self.v1_box.addLayout(self.h4)
        self.h5.addWidget(self.t5)
        self.h5.addWidget(self.l5)
        self.h5.addWidget(self.r5)
        self.v1_box.addLayout(self.h5)
        self.submit.clicked.connect(lambda: self.test_input())
        self.submit.setToolTip("Klik Untuk Mengecek Class Data")
        self.clbtn.clicked.connect(lambda: self.clfn())
        self.exit.clicked.connect(self.close)
        self.h6.addWidget(self.exit)
        self.h6.addWidget(self.submit)
        self.h6.addWidget(self.clbtn)
        self.v1_box.addLayout(self.h6)
        self.report_ui()
        self.final_hbox.addLayout(self.v1_box)
        self.final_hbox.addSpacing(40)
        self.final_hbox.addLayout(self.v2_box)
        self.setLayout(self.final_hbox)

    def report_ui(self):
        self.v2_box.setSpacing(6)
        self.report_subhead.setAlignment(Qt.AlignCenter)
        self.report_subhead.setFont(QFont("Calibri Light", 20, weight=QFont.Bold))
        self.v2_box.addWidget(self.report_subhead)
        self.details.setFont(QFont("Calibri Light", 12, weight=QFont.Bold))
        self.details.setAlignment(Qt.AlignLeft)
        self.details.setWordWrap(True)
        self.model_details.setWordWrap(True)
        self.v2_box.addWidget(self.details)
        self.results.setWordWrap(True)
        self.v2_box.addWidget(self.results)
        self.v2_box.addWidget(self.model_details)

    def clfn(self):
        """ clear all the text fields via clear button"""
        self.l1.clear()
        self.l2.clear()
        self.l3.clear()
        self.l3.clear()
        self.l4.clear()
        self.l5.clear()
        self.report_subhead.setText("About")
        self.model_details.setText(
            "Isi data dan klik submit untuk mengetahui class data")
        self.results.setText(" ")
        self.details.setText(
            "Model yang digunakan yaitu Suport Vector Machine.\nAkurasi yang didapatkan dengan model ini yaitu: "
            "79%\nDataset yang digunakan yaitu PIMA Indians "
            "diabetes dataset dari UCI archive.")
        # print(self.frameGeometry().width())
        # print(self.frameGeometry().height())

    def test_input(self) -> None:
        """ test for diabetes"""
        my_dict = {"Insulin": float(self.l1.text()), "Glucose": float(self.l2.text()), "Age": float(
            self.l3.text()), "Pregnancies": float(self.l4.text()), "BMI": float(self.l5.text())}
        output = train_svm.check_input(my_dict)
        # print(self.output)
        # self.setFixedSize(850, 342)
        self.report_subhead.setText("Reports")
        self.model_details.setText(
            "Model yang digunakan yaitu Suport Vector Machine.\nAkurasi yang didapatkan dengan model ini yaitu: "
            "79%\nDataset yang digunakan yaitu PIMA Indians "
            "diabetes dataset dari UCI archive.")
        self.details.setText("{}\n2-Hour serum insulin: {} \
\nPlasma glucose concentration: {}\nAge (years): {}\nPregnancies: {}\nBody mass index: {}".format(
            self.l0.text(), self.l1.text(), self.l2.text(), self.l3.text(), self.l4.text(), self.l5.text()))
        #
        if output == 0:
            self.results.setText(
                " Class = 0 ( Negative )")
        else:
            self.results.setText(
                " Class = 1 ( Positif )")
        self.results.setFont(QFont("Calibri Light", 12, weight=QFont.Bold))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a_window = Menu()
    a_window.mwindow()
    sys.exit(app.exec_())