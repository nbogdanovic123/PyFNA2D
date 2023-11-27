# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\UI_1D.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(472, 410)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(472, 410))
        MainWindow.setMaximumSize(QtCore.QSize(472, 411))
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("")
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.mainFrame = QtWidgets.QFrame(self.centralwidget)
        self.mainFrame.setGeometry(QtCore.QRect(0, 0, 471, 391))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.mainFrame.setFont(font)
        self.mainFrame.setAutoFillBackground(False)
        self.mainFrame.setStyleSheet("")
        self.mainFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.mainFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.mainFrame.setObjectName("mainFrame")
        self.tabWidget = QtWidgets.QTabWidget(self.mainFrame)
        self.tabWidget.setGeometry(QtCore.QRect(10, 0, 451, 351))
        self.tabWidget.setFocusPolicy(QtCore.Qt.TabFocus)
        self.tabWidget.setStyleSheet("")
        self.tabWidget.setObjectName("tabWidget")
        self.ulazniPodaci_tab = QtWidgets.QWidget()
        self.ulazniPodaci_tab.setObjectName("ulazniPodaci_tab")
        self.groupBox = QtWidgets.QGroupBox(self.ulazniPodaci_tab)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 431, 101))
        self.groupBox.setObjectName("groupBox")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 20, 401, 77))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.input_file_tbtn = QtWidgets.QToolButton(self.gridLayoutWidget)
        self.input_file_tbtn.setMinimumSize(QtCore.QSize(40, 0))
        self.input_file_tbtn.setObjectName("input_file_tbtn")
        self.gridLayout.addWidget(self.input_file_tbtn, 0, 1, 1, 1)
        self.input_file_txt = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.input_file_txt.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.input_file_txt.setObjectName("input_file_txt")
        self.gridLayout.addWidget(self.input_file_txt, 0, 0, 1, 1)
        self.worksheet_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.worksheet_btn.setEnabled(False)
        self.worksheet_btn.setMinimumSize(QtCore.QSize(208, 0))
        self.worksheet_btn.setMaximumSize(QtCore.QSize(40, 16777215))
        self.worksheet_btn.setObjectName("worksheet_btn")
        self.gridLayout.addWidget(self.worksheet_btn, 1, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.ulazniPodaci_tab)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 140, 431, 131))
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.groupBox_2)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(10, 30, 421, 91))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setVerticalSpacing(2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.hdiff_start_cell_txt = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.hdiff_start_cell_txt.setMaximumSize(QtCore.QSize(32, 16777215))
        self.hdiff_start_cell_txt.setObjectName("hdiff_start_cell_txt")
        self.gridLayout_3.addWidget(self.hdiff_start_cell_txt, 1, 1, 1, 1)
        self.repers_start_cell_txt = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.repers_start_cell_txt.setMaximumSize(QtCore.QSize(32, 16777215))
        self.repers_start_cell_txt.setObjectName("repers_start_cell_txt")
        self.gridLayout_3.addWidget(self.repers_start_cell_txt, 0, 1, 1, 1)
        self.repers_info_lbl = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.repers_info_lbl.setObjectName("repers_info_lbl")
        self.gridLayout_3.addWidget(self.repers_info_lbl, 0, 0, 1, 1)
        self.hdiff_start_lbl = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.hdiff_start_lbl.setObjectName("hdiff_start_lbl")
        self.gridLayout_3.addWidget(self.hdiff_start_lbl, 1, 0, 1, 1)
        self.cell_example_lbl = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.cell_example_lbl.setObjectName("cell_example_lbl")
        self.gridLayout_3.addWidget(self.cell_example_lbl, 1, 2, 1, 1)
        self.tabWidget.addTab(self.ulazniPodaci_tab, "")
        self.tacnost_tab = QtWidgets.QWidget()
        self.tacnost_tab.setObjectName("tacnost_tab")
        self.groupBox_3 = QtWidgets.QGroupBox(self.tacnost_tab)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 10, 211, 91))
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.groupBox_3)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(10, 20, 111, 61))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.sigmah_lbl = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.sigmah_lbl.setObjectName("sigmah_lbl")
        self.gridLayout_2.addWidget(self.sigmah_lbl, 0, 0, 1, 1)
        self.sigma0_txt = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.sigma0_txt.setObjectName("sigma0_txt")
        self.gridLayout_2.addWidget(self.sigma0_txt, 1, 1, 1, 1)
        self.sigma0_lbl = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.sigma0_lbl.setObjectName("sigma0_lbl")
        self.gridLayout_2.addWidget(self.sigma0_lbl, 1, 0, 1, 1)
        self.sigmah_txt = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.sigmah_txt.setObjectName("sigmah_txt")
        self.gridLayout_2.addWidget(self.sigmah_txt, 0, 1, 1, 1)
        self.groupBox_5 = QtWidgets.QGroupBox(self.tacnost_tab)
        self.groupBox_5.setGeometry(QtCore.QRect(230, 10, 201, 101))
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayoutWidget_5 = QtWidgets.QWidget(self.groupBox_5)
        self.gridLayoutWidget_5.setGeometry(QtCore.QRect(10, 20, 181, 77))
        self.gridLayoutWidget_5.setObjectName("gridLayoutWidget_5")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.gridLayoutWidget_5)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.alpha_coef_cmb = QtWidgets.QComboBox(self.gridLayoutWidget_5)
        self.alpha_coef_cmb.setMaximumSize(QtCore.QSize(60, 16777215))
        self.alpha_coef_cmb.setEditable(False)
        self.alpha_coef_cmb.setObjectName("alpha_coef_cmb")
        self.alpha_coef_cmb.addItem("")
        self.alpha_coef_cmb.addItem("")
        self.alpha_coef_cmb.addItem("")
        self.alpha_coef_cmb.addItem("")
        self.alpha_coef_cmb.addItem("")
        self.alpha_coef_cmb.addItem("")
        self.gridLayout_5.addWidget(self.alpha_coef_cmb, 0, 1, 1, 1)
        self.alpha_coef_lbl = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.alpha_coef_lbl.setMinimumSize(QtCore.QSize(25, 0))
        self.alpha_coef_lbl.setObjectName("alpha_coef_lbl")
        self.gridLayout_5.addWidget(self.alpha_coef_lbl, 0, 0, 1, 1)
        self.beta_coef_lbl = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.beta_coef_lbl.setMinimumSize(QtCore.QSize(25, 0))
        self.beta_coef_lbl.setObjectName("beta_coef_lbl")
        self.gridLayout_5.addWidget(self.beta_coef_lbl, 1, 0, 1, 1)
        self.beta_coef_cmb = QtWidgets.QComboBox(self.gridLayoutWidget_5)
        self.beta_coef_cmb.setMaximumSize(QtCore.QSize(60, 16777215))
        self.beta_coef_cmb.setEditable(False)
        self.beta_coef_cmb.setObjectName("beta_coef_cmb")
        self.beta_coef_cmb.addItem("")
        self.beta_coef_cmb.addItem("")
        self.beta_coef_cmb.addItem("")
        self.beta_coef_cmb.addItem("")
        self.beta_coef_cmb.addItem("")
        self.gridLayout_5.addWidget(self.beta_coef_cmb, 1, 1, 1, 1)
        self.tabWidget.addTab(self.tacnost_tab, "")
        self.datum_tab = QtWidgets.QWidget()
        self.datum_tab.setObjectName("datum_tab")
        self.datum_method_selection_gb = QtWidgets.QGroupBox(self.datum_tab)
        self.datum_method_selection_gb.setEnabled(True)
        self.datum_method_selection_gb.setGeometry(QtCore.QRect(10, 10, 421, 101))
        self.datum_method_selection_gb.setObjectName("datum_method_selection_gb")
        self.gridLayoutWidget_6 = QtWidgets.QWidget(self.datum_method_selection_gb)
        self.gridLayoutWidget_6.setGeometry(QtCore.QRect(10, 20, 401, 71))
        self.gridLayoutWidget_6.setObjectName("gridLayoutWidget_6")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.gridLayoutWidget_6)
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.min_trace_rbtn = QtWidgets.QRadioButton(self.gridLayoutWidget_6)
        self.min_trace_rbtn.setChecked(True)
        self.min_trace_rbtn.setObjectName("min_trace_rbtn")
        self.gridLayout_6.addWidget(self.min_trace_rbtn, 1, 0, 1, 1)
        self.classic_method_rbtn = QtWidgets.QRadioButton(self.gridLayoutWidget_6)
        self.classic_method_rbtn.setChecked(False)
        self.classic_method_rbtn.setObjectName("classic_method_rbtn")
        self.gridLayout_6.addWidget(self.classic_method_rbtn, 0, 0, 1, 1)
        self.method_options_gb = QtWidgets.QGroupBox(self.datum_tab)
        self.method_options_gb.setEnabled(True)
        self.method_options_gb.setGeometry(QtCore.QRect(10, 140, 421, 91))
        self.method_options_gb.setObjectName("method_options_gb")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.method_options_gb)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 30, 401, 61))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.method_opt_vert = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.method_opt_vert.setContentsMargins(0, 0, 0, 0)
        self.method_opt_vert.setObjectName("method_opt_vert")
        self.datum_method_explain_lbl = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.datum_method_explain_lbl.setObjectName("datum_method_explain_lbl")
        self.method_opt_vert.addWidget(self.datum_method_explain_lbl)
        self.method_opt_grid = QtWidgets.QGridLayout()
        self.method_opt_grid.setObjectName("method_opt_grid")
        self.datum_reper_cmb = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.datum_reper_cmb.setMinimumSize(QtCore.QSize(0, 0))
        self.datum_reper_cmb.setMaximumSize(QtCore.QSize(55, 16777215))
        self.datum_reper_cmb.setObjectName("datum_reper_cmb")
        self.method_opt_grid.addWidget(self.datum_reper_cmb, 0, 1, 1, 1)
        self.datum_reper_lbl = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.datum_reper_lbl.setObjectName("datum_reper_lbl")
        self.method_opt_grid.addWidget(self.datum_reper_lbl, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.method_opt_grid.addItem(spacerItem, 0, 2, 1, 1)
        self.method_opt_vert.addLayout(self.method_opt_grid)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.method_opt_vert.addItem(spacerItem1)
        self.tabWidget.addTab(self.datum_tab, "")
        self.export_tab = QtWidgets.QWidget()
        self.export_tab.setObjectName("export_tab")
        self.word_export_gb = QtWidgets.QGroupBox(self.export_tab)
        self.word_export_gb.setGeometry(QtCore.QRect(10, 10, 421, 71))
        self.word_export_gb.setObjectName("word_export_gb")
        self.gridLayoutWidget_7 = QtWidgets.QWidget(self.word_export_gb)
        self.gridLayoutWidget_7.setGeometry(QtCore.QRect(10, 20, 401, 41))
        self.gridLayoutWidget_7.setObjectName("gridLayoutWidget_7")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.gridLayoutWidget_7)
        self.gridLayout_8.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.word_output_btn = QtWidgets.QToolButton(self.gridLayoutWidget_7)
        self.word_output_btn.setMinimumSize(QtCore.QSize(40, 0))
        self.word_output_btn.setObjectName("word_output_btn")
        self.gridLayout_8.addWidget(self.word_output_btn, 0, 2, 1, 1)
        self.word_output_txt = QtWidgets.QLineEdit(self.gridLayoutWidget_7)
        self.word_output_txt.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.word_output_txt.setObjectName("word_output_txt")
        self.gridLayout_8.addWidget(self.word_output_txt, 0, 1, 1, 1)
        self.optional_export_gb = QtWidgets.QGroupBox(self.export_tab)
        self.optional_export_gb.setGeometry(QtCore.QRect(10, 150, 421, 161))
        self.optional_export_gb.setObjectName("optional_export_gb")
        self.gridLayoutWidget_8 = QtWidgets.QWidget(self.optional_export_gb)
        self.gridLayoutWidget_8.setGeometry(QtCore.QRect(10, 20, 401, 131))
        self.gridLayoutWidget_8.setObjectName("gridLayoutWidget_8")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.gridLayoutWidget_8)
        self.gridLayout_9.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.excel_output_ckb = QtWidgets.QCheckBox(self.gridLayoutWidget_8)
        self.excel_output_ckb.setChecked(True)
        self.excel_output_ckb.setObjectName("excel_output_ckb")
        self.gridLayout_9.addWidget(self.excel_output_ckb, 0, 0, 1, 1)
        self.excel_output_btn = QtWidgets.QToolButton(self.gridLayoutWidget_8)
        self.excel_output_btn.setMinimumSize(QtCore.QSize(40, 0))
        self.excel_output_btn.setObjectName("excel_output_btn")
        self.gridLayout_9.addWidget(self.excel_output_btn, 2, 1, 1, 1)
        self.scr_output_ckb = QtWidgets.QCheckBox(self.gridLayoutWidget_8)
        self.scr_output_ckb.setChecked(True)
        self.scr_output_ckb.setObjectName("scr_output_ckb")
        self.gridLayout_9.addWidget(self.scr_output_ckb, 3, 0, 1, 1)
        self.scr_output_txt = QtWidgets.QLineEdit(self.gridLayoutWidget_8)
        self.scr_output_txt.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.scr_output_txt.setObjectName("scr_output_txt")
        self.gridLayout_9.addWidget(self.scr_output_txt, 5, 0, 1, 1)
        self.scr_output_btn = QtWidgets.QToolButton(self.gridLayoutWidget_8)
        self.scr_output_btn.setMinimumSize(QtCore.QSize(40, 0))
        self.scr_output_btn.setObjectName("scr_output_btn")
        self.gridLayout_9.addWidget(self.scr_output_btn, 5, 1, 1, 1)
        self.excel_output_txt = QtWidgets.QLineEdit(self.gridLayoutWidget_8)
        self.excel_output_txt.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.excel_output_txt.setObjectName("excel_output_txt")
        self.gridLayout_9.addWidget(self.excel_output_txt, 2, 0, 1, 1)
        self.csv_export_gb = QtWidgets.QGroupBox(self.export_tab)
        self.csv_export_gb.setGeometry(QtCore.QRect(10, 90, 421, 51))
        self.csv_export_gb.setObjectName("csv_export_gb")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.csv_export_gb)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 20, 401, 25))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.csv_output_txt = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.csv_output_txt.setObjectName("csv_output_txt")
        self.horizontalLayout.addWidget(self.csv_output_txt)
        self.csv_output_btn = QtWidgets.QToolButton(self.horizontalLayoutWidget)
        self.csv_output_btn.setMinimumSize(QtCore.QSize(40, 0))
        self.csv_output_btn.setObjectName("csv_output_btn")
        self.horizontalLayout.addWidget(self.csv_output_btn)
        self.tabWidget.addTab(self.export_tab, "")
        self.exit_btn = QtWidgets.QPushButton(self.mainFrame)
        self.exit_btn.setGeometry(QtCore.QRect(370, 360, 71, 31))
        self.exit_btn.setObjectName("exit_btn")
        self.run_btn = QtWidgets.QPushButton(self.mainFrame)
        self.run_btn.setGeometry(QtCore.QRect(290, 360, 71, 31))
        self.run_btn.setObjectName("run_btn")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 472, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuClear = QtWidgets.QMenu(self.menuFile)
        self.menuClear.setObjectName("menuClear")
        self.menuMode = QtWidgets.QMenu(self.menubar)
        self.menuMode.setObjectName("menuMode")
        self.menu1D = QtWidgets.QMenu(self.menuMode)
        self.menu1D.setObjectName("menu1D")
        self.menu2D = QtWidgets.QMenu(self.menuMode)
        self.menu2D.setObjectName("menu2D")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionNew_Project = QtWidgets.QAction(MainWindow)
        self.actionNew_Project.setObjectName("actionNew_Project")
        self.actionSave_Project = QtWidgets.QAction(MainWindow)
        self.actionSave_Project.setObjectName("actionSave_Project")
        self.actionOpen_Project = QtWidgets.QAction(MainWindow)
        self.actionOpen_Project.setObjectName("actionOpen_Project")
        self.actionClear_Current = QtWidgets.QAction(MainWindow)
        self.actionClear_Current.setObjectName("actionClear_Current")
        self.actionClear_All = QtWidgets.QAction(MainWindow)
        self.actionClear_All.setObjectName("actionClear_All")
        self.action1D_Free_Network_Adjustment = QtWidgets.QAction(MainWindow)
        self.action1D_Free_Network_Adjustment.setObjectName("action1D_Free_Network_Adjustment")
        self.action1D_Network_Adjustment = QtWidgets.QAction(MainWindow)
        self.action1D_Network_Adjustment.setObjectName("action1D_Network_Adjustment")
        self.action1D_Network_Analysis = QtWidgets.QAction(MainWindow)
        self.action1D_Network_Analysis.setObjectName("action1D_Network_Analysis")
        self.action2D_Free_Network_Adjustment = QtWidgets.QAction(MainWindow)
        self.action2D_Free_Network_Adjustment.setObjectName("action2D_Free_Network_Adjustment")
        self.action2D_Network_Adjustment = QtWidgets.QAction(MainWindow)
        self.action2D_Network_Adjustment.setObjectName("action2D_Network_Adjustment")
        self.action2D_Network_Analysis = QtWidgets.QAction(MainWindow)
        self.action2D_Network_Analysis.setObjectName("action2D_Network_Analysis")
        self.menuClear.addAction(self.actionClear_Current)
        self.menuClear.addAction(self.actionClear_All)
        self.menuFile.addAction(self.actionNew_Project)
        self.menuFile.addAction(self.actionOpen_Project)
        self.menuFile.addAction(self.actionSave_Project)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.menuClear.menuAction())
        self.menu1D.addAction(self.action1D_Free_Network_Adjustment)
        self.menu1D.addAction(self.action1D_Network_Adjustment)
        self.menu1D.addAction(self.action1D_Network_Analysis)
        self.menu2D.addAction(self.action2D_Free_Network_Adjustment)
        self.menu2D.addAction(self.action2D_Network_Adjustment)
        self.menu2D.addAction(self.action2D_Network_Analysis)
        self.menuMode.addAction(self.menu1D.menuAction())
        self.menuMode.addAction(self.menu2D.menuAction())
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuMode.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Excel fajl sa podacima"))
        self.input_file_tbtn.setText(_translate("MainWindow", "..."))
        self.worksheet_btn.setText(_translate("MainWindow", "Promenite radni list sa podacima"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Podaci i njihove pozicije u fajlu"))
        self.repers_info_lbl.setText(_translate("MainWindow", "Približne visine repera - Poćetna ćelija u fajlu:"))
        self.hdiff_start_lbl.setText(_translate("MainWindow", "Visinske razlike i dužine - Početna ćelija fajlu:"))
        self.cell_example_lbl.setText(_translate("MainWindow", "  Primer: B12"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.ulazniPodaci_tab), _translate("MainWindow", "Ulazni podaci"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Tačnost merenih veličina"))
        self.sigmah_lbl.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-style:italic;\">σ</span><span style=\" font-size:14pt; font-style:italic; vertical-align:sub;\">Δhi</span><span style=\" font-style:italic;\">: </span></p></body></html>"))
        self.sigma0_lbl.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-style:italic;\">σ</span><span style=\" font-size:14pt; font-style:italic; vertical-align:sub;\">0</span><span style=\" font-style:italic;\">: </span></p></body></html>"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Nivo značajnosti i moć testa"))
        self.alpha_coef_cmb.setItemText(0, _translate("MainWindow", "0.05"))
        self.alpha_coef_cmb.setItemText(1, _translate("MainWindow", "0.10"))
        self.alpha_coef_cmb.setItemText(2, _translate("MainWindow", "0.15"))
        self.alpha_coef_cmb.setItemText(3, _translate("MainWindow", "0.20"))
        self.alpha_coef_cmb.setItemText(4, _translate("MainWindow", "0.25"))
        self.alpha_coef_cmb.setItemText(5, _translate("MainWindow", "0.30"))
        self.alpha_coef_lbl.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:11pt; font-style:italic;\">α</span>: nivo značajnosti</p></body></html>"))
        self.beta_coef_lbl.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:11pt; font-style:italic;\">β</span>: moć testa</p></body></html>"))
        self.beta_coef_cmb.setItemText(0, _translate("MainWindow", "0.80"))
        self.beta_coef_cmb.setItemText(1, _translate("MainWindow", "0.90"))
        self.beta_coef_cmb.setItemText(2, _translate("MainWindow", "0.95"))
        self.beta_coef_cmb.setItemText(3, _translate("MainWindow", "0.975"))
        self.beta_coef_cmb.setItemText(4, _translate("MainWindow", "0.99"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tacnost_tab), _translate("MainWindow", "Tačnost"))
        self.datum_method_selection_gb.setTitle(_translate("MainWindow", "Metoda određivanja datuma"))
        self.min_trace_rbtn.setText(_translate("MainWindow", "Minimalni trag kofaktorske matrice"))
        self.classic_method_rbtn.setText(_translate("MainWindow", "Klasičan metod"))
        self.method_options_gb.setTitle(_translate("MainWindow", "Klasičan metod - opcije"))
        self.datum_method_explain_lbl.setText(_translate("MainWindow", "Izaberite reper koja će biti fiksiran"))
        self.datum_reper_lbl.setText(_translate("MainWindow", "Reper:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.datum_tab), _translate("MainWindow", "Datum"))
        self.word_export_gb.setTitle(_translate("MainWindow", "Word izveštaj"))
        self.word_output_btn.setText(_translate("MainWindow", "..."))
        self.optional_export_gb.setTitle(_translate("MainWindow", "Opcioni izlazni fajlovi"))
        self.excel_output_ckb.setText(_translate("MainWindow", "Prateći Excel fajl sa matricama i ostalim podacima"))
        self.excel_output_btn.setText(_translate("MainWindow", "..."))
        self.scr_output_ckb.setText(_translate("MainWindow", "CAD script fajl (*.scr) sa izravnatim visinama repera\n"
" i elipsama grešaka"))
        self.scr_output_btn.setText(_translate("MainWindow", "..."))
        self.csv_export_gb.setTitle(_translate("MainWindow", "CSV fajl sa izravnatim visinama repera"))
        self.csv_output_btn.setText(_translate("MainWindow", "..."))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.export_tab), _translate("MainWindow", "Export podataka"))
        self.exit_btn.setText(_translate("MainWindow", "Zatvori"))
        self.run_btn.setText(_translate("MainWindow", "Pokreni"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuClear.setTitle(_translate("MainWindow", "Clear"))
        self.menuMode.setTitle(_translate("MainWindow", "Mode"))
        self.menu1D.setTitle(_translate("MainWindow", "1D"))
        self.menu2D.setTitle(_translate("MainWindow", "2D"))
        self.actionNew_Project.setText(_translate("MainWindow", "New Project"))
        self.actionSave_Project.setText(_translate("MainWindow", "Save Project"))
        self.actionOpen_Project.setText(_translate("MainWindow", "Open Project"))
        self.actionClear_Current.setText(_translate("MainWindow", "Clear Current"))
        self.actionClear_All.setText(_translate("MainWindow", "Clear All"))
        self.action1D_Free_Network_Adjustment.setText(_translate("MainWindow", "Free Network Adjustment"))
        self.action1D_Network_Adjustment.setText(_translate("MainWindow", "Network Adjustment"))
        self.action1D_Network_Analysis.setText(_translate("MainWindow", "Network Analysis"))
        self.action2D_Free_Network_Adjustment.setText(_translate("MainWindow", "Free Network Adjustment"))
        self.action2D_Network_Adjustment.setText(_translate("MainWindow", "Network Adjustment"))
        self.action2D_Network_Analysis.setText(_translate("MainWindow", "Network Analysis"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
