from typing import Tuple
from PyQt5.QtGui import QStandardItem

from DataTypes import Point
from QtGUI import Ui_MainWindow
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QFileDialog, QMessageBox, QComboBox, \
    QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import QThread, QRect, QSize, Qt, QMetaObject, QCoreApplication, QPoint, QEventLoop, QModelIndex
from Adjustment import run2d
from FileHandler import *
import sys

# TODO Implement new UI (HA GL MF YOU GONNA NEED IT)


class MainWindow(QMainWindow, Ui_MainWindow):
    msg_signal = QtCore.pyqtSignal(str, str)
    PARAMS = {
        'inputFile': r"C:\Users\nikol\OneDrive\Documents\IG3\vezba1\vezba1.xlsx",
        'Worksheet': None,
        'PtsDataStart': None,
        'DisDataStart': None,
        'DirDataStart': None,
        'sigmaD': (None, None),
        'sigmaP': None,
        'sigma0': None,
        'DisRepeat': None,
        'DirGyrus': None,
        'alphaCoef': None,
        'betaCoef': None,
        'datumMethod': None,
        'datumCoords': None,
        'wordOutput': 'C:/Users/nikol/OneDrive/Documents/Diplomski/python/Izravnanje/testFiles/izvestaj.docx',
        'csvOutput': 'C:/Users/nikol/OneDrive/Documents/Diplomski/python/Izravnanje/testFiles/points.csv',
        'hasExcel': None,
        'excelOutput': 'C:/Users/nikol/OneDrive/Documents/Diplomski/python/Izravnanje/testFiles/excel.xlsx',
        'hasScript': None,
        'scrOutput': 'C:/Users/nikol/OneDrive/Documents/Diplomski/python/Izravnanje/testFiles/scrTest.scr'
    }

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('PyFNA2D')

        self.selected_points_lbl = QLabel()
        self.selected_points_lbl.setText(' Izabrane tačke: -')
        self.datum_point_checkbox_cmb = CheckableComboBox(self.selected_points_lbl)
        self.datum_point_checkbox_cmb.setMinimumWidth(100)

        self.window_open = False
        self.default_folder = None
        self.window: QWidget = None

        self.worker_thread = QThread()
        self.worker = Worker()
        self.worker.msg_signal = self.msg_signal

        self._main_loop()

    def _main_loop(self):
        """
        Main program loop. Handles signals and slot connections
        :return None:
        """
        # temp
        self.points_start_cell_txt.setText('B4')
        self.directions_start_cell_txt.setText('F4')
        self.distances_start_cell_txt.setText('M4')
        self.input_file_txt.setText(r"C:\Users\nikol\OneDrive\Documents\IG3\vezba1\vezba1.xlsx")
        self.excel_output_txt.setText('C:/Users/nikol/OneDrive/Documents/Diplomski/python/Izravnanje/testFiles'
                                      '/excel.xlsx')
        self.scr_output_txt.setText('C:/Users/nikol/OneDrive/Documents/Diplomski/python/Izravnanje/testFiles'
                                    '/scrTest.scr')
        self.word_output_txt.setText('C:/Users/nikol/OneDrive/Documents/Diplomski/python/Izravnanje/testFiles'
                                     '/izvestaj.docx')
        self.csv_output_txt.setText('C:/Users/nikol/OneDrive/Documents/Diplomski/python/Izravnanje/testFiles'
                                    '/points.csv')
        self.sigmap_txt.setText('5')
        self.sigma0_txt.setText('1')
        self.sigmad_base_txt.setText('3')
        self.sigmad_per_km_txt.setText('0')
        self.dir_gyrus_txt.setText('1')
        self.dis_repeat_txt.setText('1')
        self.min_trace_rbtn.setChecked(True)

        self.worksheet_btn.clicked.connect(lambda: self._get_sheetname(False))
        self.exit_btn.clicked.connect(self._exit_window)
        self.exit_btn.setShortcut('Ctrl+2')
        self.input_file_tbtn.clicked.connect(self._get_input_file)
        self.run_btn.setShortcut('Ctrl+1')
        self.run_btn.clicked.connect(self._run_worker)
        self.classic_method_rbtn.clicked.connect(self._update_datum_tab)
        self.min_trace_rbtn.clicked.connect(lambda: self._update_datum_tab(is_classic=False))
        self.tabWidget.currentChanged.connect(self._refresh_datum_tab)
        self.word_output_btn.clicked.connect(lambda: self._get_output_file('Word (*.docx)'))
        self.excel_output_btn.clicked.connect(lambda: self._get_output_file('Excel (*.xlsx)'))
        self.scr_output_btn.clicked.connect(lambda: self._get_output_file('CAD script (*.scr)'))
        self.csv_output_btn.clicked.connect(lambda: self._get_output_file('CSV (*.csv)'))
        self.excel_output_ckb.clicked.connect(self._refresh_export_tab)
        self.scr_output_ckb.clicked.connect(self._refresh_export_tab)
        self.datum_point_cmb.currentIndexChanged.connect(lambda: self._update_datum_coord_cmb(self.datum_point_cmb
                                                                                              .currentIndex()))

        self.msg_signal.connect(self.pop_up_msg)
        self.worker.finished.connect(self._exit_worker)

    def _get_default_folder(self, path):
        """
        Returns the folder in which the file provided in path is. Used to get the default folder for future files.
        Updates the self.default_folder with the path.
        :param str path: File path
        :return None:
        """
        folders = path.split('/')
        folders.pop()
        self.default_folder = '/'.join(folders) + '/'

    def _get_input_file(self):
        """
        QFileDialog - updates the self.PARAMS 'inputFile' key with file path.
        :return None:
        """
        file, ext = QFileDialog.getOpenFileName(self, 'Open', 'C:/Users/nikol/OneDrive/Documents/Diplomski/python'
                                                      '/Izravnanje/testFiles/', "Excel (*.xlsx)")

        if file:
            self._get_default_folder(file)
            self.PARAMS.update({'inputFile': file})
            self.input_file_txt.setText(file)
            self.input_file_txt.setReadOnly(True)
            self.PARAMS.update({'Worksheet': None})
            self.worksheet_btn.setEnabled(False)
        elif self.PARAMS.get('inputFile') in ['', None]:
            self.msg_signal.emit('w', 'Niste izabrali fajl')
            self.tabWidget.setCurrentIndex(0)

        if self.window_open:
            self.window.close()
            self.window_open = False

    def _get_output_file(self, extension):
        """
        QFileDialog - updates the self.PARAMS corresponding key with file path based on file extension provided.
        :param str extension: File extension
        :return None:
        """
        file, ext = QFileDialog.getSaveFileName(self, 'Save', self.default_folder, extension)

        if file and '.docx' in ext:
            self.PARAMS.update({'wordOutput': file})
            self.word_output_txt.setText(file)
            self.word_output_txt.setReadOnly(True)
        elif file and '.xlsx' in ext:
            self.PARAMS.update({'excelOutput': file})
            self.excel_output_txt.setText(file)
            self.excel_output_txt.setReadOnly(True)
        elif file and '.scr' in ext:
            self.PARAMS.update({'scrOutput': file})
            self.scr_output_txt.setText(file)
            self.scr_output_txt.setReadOnly(True)
        elif file and '.csv' in ext:
            self.PARAMS.update({'csvOutput': file})
            self.csv_output_txt.setText(file)
            self.csv_output_txt.setReadOnly(True)
        else:
            self.msg_signal.emit('w', 'Niste sačuvali sve izabrane fajlove!')
            self.tabWidget.setCurrentIndex(3)

    def _get_sheetname(self, info_flag):
        """
        Updates self.PARAMS 'Worksheet' key with sheetname.
        :param bool info_flag: Switch between emitting a self.msg_signal and not emitting it.
        :return None:
        """
        wb = get_workbook(self.PARAMS.get('inputFile'), self.msg_signal)
        if wb == 'Error':
            return 'Error'

        worksheets = wb.worksheets
        if len(worksheets) == 1:
            ws = wb.active
            self.PARAMS.update({'Worksheet': ws.title})
            self.worksheet_btn.setEnabled(False)
        else:
            if info_flag:
                self.msg_signal.emit('i', 'Iz padajućeg menija izaberite radni list na kom se nalaze podaci.')
            self.run_btn.setEnabled(False)
            self.window = InputWindow()

            self.moveToThread(self.worker_thread)
            self.window_open = True
            self.window.move(QPoint(540, 470))
            self.window.set_cb_items(worksheets)
            event_loop = QEventLoop()
            self.window.windowClosed.connect(lambda: self.run_btn.setEnabled(True))
            self.window.button.clicked.connect(event_loop.quit)  # type: ignore
            self.window.button.clicked.connect(self._update_worksheet)  # type: ignore
            self.window.show()
            event_loop.exec_()
            self.PARAMS.update({'Worksheet': self.window.combo_box.currentText()})
            self.run_btn.setEnabled(True)
            self.worksheet_btn.setEnabled(True)

    def _get_user_input(self) -> None:
        """
        Updates self.PARAMS corresponding keys with their values based on user input in UI.
        :return None:
        """
        # TAB 1
        if self.PARAMS.get('Worksheet') == '' or self.PARAMS.get('inputFile') != self.PARAMS.get('previousInputFile'):
            self.PARAMS.update({'Worksheet': ''})
            self._get_sheetname(True)

        self.PARAMS.update({'previousInputFile': self.PARAMS.get('inputFile')})  # DON'T ASK ME ANYTHING

        self.PARAMS.update({'PtsDataStart': self.points_start_cell_txt.text()})
        self.PARAMS.update({'DisDataStart': self.distances_start_cell_txt.text()})
        self.PARAMS.update({'DirDataStart': self.directions_start_cell_txt.text()})

        # TAB 2
        self.PARAMS.update({'sigmaD': (self.sigmad_base_txt.text(), self.sigmad_per_km_txt.text())})
        self.PARAMS.update({'sigmaP': self.sigmap_txt.text()})
        self.PARAMS.update({'sigma0': self.sigma0_txt.text()})
        self.PARAMS.update({'DisRepeat': self.dis_repeat_txt.text()})
        self.PARAMS.update({'DirGyrus': self.dir_gyrus_txt.text()})
        self.PARAMS.update({'alphaCoef': self.alpha_coef_cmb.currentText()})
        self.PARAMS.update({'betaCoef': self.beta_coef_cmb.currentText()})

        # TAB 3
        if self.classic_method_rbtn.isChecked():
            self.PARAMS.update({'datumMethod': 'classic'})
            self.PARAMS.update({'datumCoords':
                                f'{self.datum_point_cmb.currentText()};{self.datum_coord_cmb.currentText()}'})

        if self.min_trace_rbtn.isChecked():
            self.PARAMS.update({'datumMethod': 'min_trace'})
            self.PARAMS.update({'datumCoords': self.datum_point_checkbox_cmb.get_selected_items()})

        # TAB 4
        self._refresh_export_tab()

    def _refresh_datum_tab(self):
        """
        Refreshes Datum tab every time its clicked and checks if there is points data needed for datum tab to display
        correct information.
        :return None:
        """
        if self.tabWidget.currentIndex() == 2:
            self.PARAMS.update({'PtsDataStart': self.points_start_cell_txt.text()})
            if not self.PARAMS['inputFile']:
                self.tabWidget.setCurrentIndex(0)
                self.pop_up_msg('c', "Morate izabrati fajl sa podacima pre prelaska na tab 'Datum'")
            elif self.PARAMS['PtsDataStart'] == '':
                self.tabWidget.setCurrentIndex(0)
                self.pop_up_msg('c', "Morate popuniti polje sa početnom ćelijom tačaka pre prelaska na tab 'Datum'")

            if self.classic_method_rbtn.isChecked():
                self._update_datum_tab()
            else:
                self._update_datum_tab(is_classic=False)
                self.PARAMS.update({'datumMethod': 'min_trace'})
                self.PARAMS.update({'datumCoords': self.datum_point_checkbox_cmb.get_selected_items()})

    def _refresh_export_tab(self):
        """
        Refreshes Export tab based on checked or unchecked QCheckBox
        :return None:
        """
        if self.excel_output_ckb.isChecked():
            self.excel_output_txt.setEnabled(True)
            self.excel_output_btn.setEnabled(True)
            self.PARAMS.update({'hasExcel': True})
        else:
            self.excel_output_txt.clear()
            self.excel_output_txt.setEnabled(False)
            self.excel_output_btn.setEnabled(False)
            self.PARAMS.update({'hasExcel': False})

        if self.scr_output_ckb.isChecked():
            self.scr_output_txt.setEnabled(True)
            self.scr_output_btn.setEnabled(True)
            self.PARAMS.update({'hasScript': True})
        else:
            self.scr_output_txt.clear()
            self.scr_output_txt.setEnabled(False)
            self.scr_output_btn.setEnabled(False)
            self.PARAMS.update({'hasScript': False})

    def _update_datum_tab(self, is_classic=True):
        """
        Changes Datum tab UI layout based on which datum method is selected.
        :keyword is_classic: True if classic method is selected.
        :return None:
        """
        if self.tabWidget.currentIndex() == 2:
            classic_gb_size = (421, 121)
            # initial prep
            if not self.PARAMS.get('Worksheet'):
                if self._get_sheetname(True) == 'Error':
                    self.tabWidget.setCurrentIndex(0)
                    return

            wb = get_workbook(self.PARAMS.get('inputFile'), self.msg_signal)
            pts = extract_excel_data(wb, self.PARAMS.get('PtsDataStart'), self.PARAMS.get('Worksheet'), self.msg_signal)
            self.points = Point.to_point_list(pts)

            if is_classic:
                # UI prep
                self.ui_prep = True
                self.datum_point_cmb.clear()
                self.datum_coord_cmb.clear()
                self.datum_point_checkbox_cmb.clear()

                self.method_options_gb.resize(*classic_gb_size)
                self.method_options_gb.setTitle('Klasičan metod - opcije')
                self.datum_method_explain_lbl.setText('Izaberite tačku i koordinatu koja će biti fiksirana:')

                self.datum_point_checkbox_cmb.hide()
                self.selected_points_lbl.hide()
                self.datum_coord_lbl.show()
                self.datum_coord_cmb.show()
                self.datum_point_cmb.show()
                self.datum_point_lbl.show()

                # Adding items
                for i, point in enumerate(self.points):
                    self.datum_point_cmb.addItem(point.id)
                    if i != 0:
                        self.datum_coord_cmb.addItem(f'Y-{point.id}')
                        self.datum_coord_cmb.addItem(f'X-{point.id}')

                self.ui_prep = False

            else:
                # UI prep
                self.datum_point_cmb.clear()
                self.datum_coord_cmb.clear()
                self.datum_point_checkbox_cmb.clear()

                self.method_options_gb.resize(classic_gb_size[0], classic_gb_size[1] - 30)
                self.method_options_gb.setTitle('Minimalni trag kofaktorske matrice - opcije')
                self.datum_method_explain_lbl.setText('Izaberite tačke koje će formirati minimalni trag:')
                self.datum_coord_lbl.hide()
                self.datum_coord_cmb.hide()
                self.datum_point_cmb.hide()
                self.datum_point_lbl.hide()
                self.datum_point_checkbox_cmb.show()
                self.selected_points_lbl.show()

                self.method_opt_grid.addWidget(self.datum_point_checkbox_cmb, 0, 0, 1, 1)
                self.method_opt_grid.addWidget(self.selected_points_lbl, 0, 1, 1, 1)
                self.datum_point_checkbox_cmb.show()

                # Adding items
                self.datum_point_checkbox_cmb.addItem('Sve tačke')
                self.datum_point_checkbox_cmb.set_item_checked(0)

                for i, point in enumerate(self.points):
                    self.datum_point_checkbox_cmb.addItem(point.id)
                    self.datum_point_checkbox_cmb.set_item_checked(i+1)  # already reserved item ID 0, 1  for all points

    def _update_datum_coord_cmb(self, selected_point_index):
        """
        Updates self.datum_coord_cmb to exclude the coordinates of the selected point
        :param int selected_point_index: Index in self.points of the currently selected Point from datum_point_cmb
        :return None:
        """
        if self.ui_prep:
            return

        self.datum_coord_cmb.clear()

        selected_point = self.points.pop(selected_point_index)

        for point in self.points:
            self.datum_coord_cmb.addItem(f'Y-{point.id}')
            self.datum_coord_cmb.addItem(f'X-{point.id}')

        self.points.insert(selected_point_index, selected_point)

    def _update_worksheet(self):
        """
        Updates self.PARAMS 'Worksheet' key with value from QComboBox in InputWindow instance
        :return None:
        """
        sheetname = self.window.get_item()
        self.PARAMS.update({'Worksheet': sheetname})
        self.run_btn.setEnabled(True)
        self.window.close()

    def _run_worker(self):
        """
        Moves the Worker instance to a worker_thread and calls Worker's run_adjustment() method providing self.PARAMS
        :return None:
        """
        self._get_user_input()
        self.worker.moveToThread(self.worker_thread)
        self.worker.run_adjustment(self.PARAMS)
        self.worker_thread.start()

    def _exit_worker(self):
        """
        Slot connected to Worker's finished signal. Closing the worker thread when signal emitted.
        :return None:
        """
        self.worker_thread.quit()
        self.pop_up_msg('i', 'Izravnanje uspešno završeno.\nPodaci eksportovani u željene fajlove.')

    def pop_up_msg(self, msg_type, msg):
        """
        Main communication method used for relaying information to the user.
        :param str msg_type: Message type, it can be: 'c': critical, 'i': information, 'w': warning, 'q': question
        :param str msg: Message you want to display
        :return None:
        """
        if msg_type == 'c':
            QMessageBox.critical(self, 'Greška', msg)
        if msg_type == 'i':
            QMessageBox.information(self, 'Info', msg)
        if msg_type == 'w':
            QMessageBox.warning(self, 'Upozorenje', msg)
        if msg_type == 'q':
            QMessageBox.question(self, 'Pitanje', msg)

    def _show_window(self):
        """
        On call creates InputWindow() instance and shows it. Sets self.window_open to True and connects closed signal
        to _on_window_closed method.
        :return None:
        """
        self.window = InputWindow()
        self.window.show()
        self.window_open = True
        self.window.closed.connect(self._on_window_closed)

    def _on_window_closed(self):
        """
        Sets self.window_open to False
        :return None:
        """
        self.window_open = False

    def _exit_window(self):
        """
        Closes running thread and all active windows.
        :return None:
        """
        self.worker_thread.quit()
        if self.window_open:
            self.window.close()
        self.close()


class Worker(QThread):
    """
    Worker class used for handling the actual adjustment portion of the program.
    """
    msg_signal = None

    PARAMS_FULL_NAMES = {
        'inputFile': 'Ulazni Excel fajl',
        'PtsDataStart': 'Početna ćelija tačaka',
        'DisDataStart': 'Početna ćelija dužina',
        'DirDataStart': 'Početna ćelija pravaca',
        'sigmaD': '\u03C3D',
        'sigmaP': '\u03C3P',
        'sigma0': '\u03C30',
        'DisRepeat': 'Broj ponavljanja',
        'DirGyrus': 'Broj girusa',
        'alphaCoef': '\u03B1',
        'betaCoef': '\u03B2',
        'wordOutput': 'Word izveštaj',
        'csvOutput': 'CSV fajl sa koordinata tačaka',
        'excelOutput': 'Prateći Excel fajl',
        'scrOutput': 'CAD Script fajl',
    }

    def __init__(self) -> None:
        super(Worker, self).__init__()

    def _params_no_value_check(self, params) -> Tuple[bool, str, str]:
        """
        Checking values in dictonary for no value (None or empty string).
        :param dict params: Dictonary with user defined parameters from UI
        :returns: True if no value found with two string containing message type (pop_up_msg types) and a message for
        user. False if all dict values are not None or empty.
        """
        flag = False
        msg = ''
        msg_type = 'c'
        error_break = False

        for k, v in params.items():
            # print(f'key: {k} - value: {v}')
            if k == 'previousInputFile' or (k == 'Worksheet' and v is None):
                continue

            if k == 'sigmaD':
                base, per_km = v
                if base == '' or per_km == '':
                    flag = True
                    msg = f'Polje parametra \'{self.PARAMS_FULL_NAMES.get(k)}\' je prazno!'
                    error_break = True

            if k == 'datumMethod' and v is None:
                return True, 'c', 'Niste izabrali metodu određivanja datuma!'

            if k == 'datumCoords' and len(v) == 0:
                return True, 'c', 'Izaberite tačku i koordinatu ili tačke koje će definisati' \
                                  '\ndatum mreže u zavisnosti od izabrane metode.'

            if k == 'excelOutput':
                if params['hasExcel'] and v is None:
                    flag = True
                    msg_type = 'w'
                    msg = f'Nedostaje putanja do opcionog excel fajla.'
                    error_break = True
                else:
                    continue

            if k == 'scrOutput':
                if params['hasScript'] and v is None:
                    flag = True
                    msg_type = 'w'
                    msg = f'Nedostaje putanja do opcionog CAD Script fajla.'
                    error_break = True
                else:
                    continue

            if v in ['', None]:
                if params.get('Worksheet') == '':
                    flag = True
                    msg_type = 'w'
                    msg = f'Postoji više radnih listova u izabranom excel fajlu. Iz padajućeg menija izaberite ' \
                          f'u kom se radnom listu nalaze podaci i pokrenite ponovo.'
                    error_break = True
                else:
                    flag = True
                    msg = f'Polje parametra \'{self.PARAMS_FULL_NAMES.get(k)}\' je prazno!'
                    error_break = True

            if error_break:
                break

        return flag, msg_type, msg

    def _correct_params_value_type(self, params) -> Tuple[bool, str | dict]:
        """
        Checks for correct values and does type casting, value types and other similar conditions.
        :param dict params: Dictornary containing user defined parameters from UI
        :returns: True when error found along with a message. Otherwise, return False with the type casted params dict
         values.
        """
        for k, v in params.items():
            if k in ['PtsDataStart', 'DirDataStart', 'DisDataStart']:
                col_start: str = v.rstrip('0123456789')
                row_start: str = v[len(col_start):]
                if not row_start.isdigit():
                    return True, f'Parametar {self.PARAMS_FULL_NAMES.get(k)} mora biti u formatu: \n' \
                                 f' <slovo kolone><broj reda>'
                elif not col_start.isalpha():
                    return True, f'Parametar {self.PARAMS_FULL_NAMES.get(k)} mora biti u formatu: \n' \
                                 f' <slovo kolone><broj reda>'

            if k == 'sigmaD':
                try:
                    base, per_km = v
                    base = float(base)
                    per_km = float(per_km)
                    params.update({k: (base, per_km)})
                except ValueError:
                    return True, f'Parametar {self.PARAMS_FULL_NAMES.get(k)} mora biti broj!'
                except Exception as e:
                    raise e

            if k in ['sigmaP', 'sigma0']:
                try:
                    v = float(v)
                    params.update({k: v})
                except ValueError:
                    return True, f'Parametar {self.PARAMS_FULL_NAMES.get(k)} mora biti broj!'
                except Exception as e:
                    raise e

            if k in ['DisRepeat', 'DirGyrus']:
                try:
                    v = int(v)
                    params.update({k: v})
                except ValueError as e:
                    if '.' in e.args[0]:
                        return True, f'Parametar {self.PARAMS_FULL_NAMES.get(k)} mora biti ceo broj!'
                    else:
                        return True, f'Parametar {self.PARAMS_FULL_NAMES.get(k)} mora biti broj!'
                except Exception as e:
                    raise e

            if k == 'datumCoords' and params['datumMethod'] == 'classic':
                point, coord = params.get('datumCoords').split(';')
                _, coord_point = coord.split('-')
                if point == coord_point:
                    return True, f"Koordinata je već fiksirana izabranom tačkom: '{point}'. " \
                                 f"\nIzaberite drugu koordinatu ili drugu tačku"

        return False, params

    def run_adjustment(self, params):
        """
        Calls run2d() function from Adjustment.py providing user parameters and self.msg_signal
        Emits finished signal on successful completion.
        :param dict params: Dictornary containing user defined parameters from UI
        :return None:
        """
        global params_correction
        no_value_check = self._params_no_value_check(params)
        if not no_value_check[0]:
            params_correction = self._correct_params_value_type(params)

        if no_value_check[0]:
            self.msg_signal.emit(no_value_check[1], f'{no_value_check[2]}')
            self.quit()
        elif params_correction[0]:
            self.msg_signal.emit('c', f'{params_correction[1]}')
            self.quit()
        else:
            if run2d(params, self.msg_signal):
                self.finished.emit()


class InputWindow(QWidget):
    """
    Classes used for choosing a Worksheet on which the data is in a multi-sheet Excel file.
    """
    windowClosed = QtCore.pyqtSignal()

    def __init__(self):
        super(InputWindow, self).__init__()
        self.verticalLayoutWidget = QWidget(self)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(10, 10, 151, 81))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.worksheet_lbl = QLabel(self.verticalLayoutWidget)
        self.worksheet_lbl.setObjectName(u"worksheet_lbl")
        self.worksheet_lbl.setMinimumSize(QSize(0, 7))
        self.worksheet_lbl.setMaximumSize(QSize(16777215, 20))
        self.verticalLayout.addWidget(self.worksheet_lbl)

        self.combo_box = QComboBox(self.verticalLayoutWidget)
        self.combo_box.setObjectName(u"combo_box")
        self.combo_box.setMinimumSize(QSize(105, 0))

        self.verticalLayout.addWidget(self.combo_box, 0, Qt.AlignHCenter)

        self.button = QPushButton(self.verticalLayoutWidget)
        self.button.setObjectName(u"button")
        self.button.setMinimumSize(QSize(20, 0))
        self.button.setMaximumSize(QSize(50, 50))
        self.verticalLayout.addWidget(self.button, 0, Qt.AlignHCenter)

        self.show()
        self._retranslate_ui()

        QMetaObject.connectSlotsByName(self)

    def _retranslate_ui(self):
        self.setWindowTitle(QCoreApplication.translate("Radni list", u"Radni list", None))
        self.worksheet_lbl.setText(QCoreApplication.translate("Form", u"<html><head/><body><p "
                                                                      u"align=\"center\">Izaberite radni "
                                                                      u"list:</p></body></html>", None))
        self.button.setText(QCoreApplication.translate("Form", u"OK", None))

    def set_cb_items(self, item_list):
        """
        Fills with combo_box with items from the list.
        :param List[str] item_list: List containing all worksheets in the chosen Excel file.
        :return None:
        """
        for item in item_list:
            name = item.title
            self.combo_box.addItem(name)

    def get_item(self):
        """
        :returns str: Returns chosen item from the combo_box.
        """
        return self.combo_box.currentText()

    def closeEvent(self, event):
        self.windowClosed.emit()
        super().closeEvent(event)


class CheckableComboBox(QComboBox):
    """
    QComboBox subclass with checkable items.
    """
    def __init__(self, label):
        super(CheckableComboBox, self).__init__()
        self._changed = False
        self.selected_items = []
        self.selected_points_label: QLabel = label
        self.setPlaceholderText("Izaberite tačke")

        self.view().pressed.connect(self.handle_item_pressed)

    def set_item_checked(self, index, checked=False):
        """
        Sets item of the corresponding index to Qt.Checked/Qt.Unchecked state.
        :param QModelIndex index: Item index in the model.
        :param checked: True if item should be set to Qt.Checked state. Otherwise, sets to Qt.Unchecked state.
        :return None:
        """
        item: QStandardItem = self.model().item(index, self.modelColumn())

        if checked:
            item.setCheckState(Qt.Checked)
        else:
            item.setCheckState(Qt.Unchecked)

    def handle_item_pressed(self, index: QModelIndex):
        """
        Triggers everytime an item is clicked. Sets its Qt.Checked/Qt.Unchecked state based on its checkbox.
        Updates self.selected_items.
        :param QModelIndex index: Item index in the model
        :return None:
        """
        item: QStandardItem = self.model().itemFromIndex(index)

        if item.checkState() == Qt.Checked:
            item.setCheckState(Qt.Unchecked)
        else:
            item.setCheckState(Qt.Checked)

        if index.row() == 0 and item.checkState() == Qt.Checked:  # checks all items if checked
            for i in range(self.count() - 1):
                item: QStandardItem = self.model().itemFromIndex(index.siblingAtRow(i+1))
                item.setCheckState(Qt.Checked)
        elif index.row() == 0 and item.checkState() == Qt.Unchecked:  # unchecks all items if unchecked
            for i in range(self.count() - 1):
                item: QStandardItem = self.model().itemFromIndex(index.siblingAtRow(i+1))
                item.setCheckState(Qt.Unchecked)
        else:
            item: QStandardItem = self.model().itemFromIndex(index.siblingAtRow(0))
            item.setCheckState(Qt.Unchecked)

        self._changed = True
        self.update_selected_items(index)

    def update_selected_items(self, index):
        """
        Updates self.selected items with all checked items.
        :param QModelIndex index:  Item index in the model
        :return None:
        """
        self.selected_items.clear()
        for i in range(self.count()):
            item: QStandardItem = self.model().itemFromIndex(index.siblingAtRow(i))
            if item.checkState() == Qt.Checked:
                self.selected_items.append(item.text())

        if self.itemText(0) in self.selected_items:
            self.selected_items = self.selected_items[0:1]

        self.selected_points_label.setText(' Izabrane tačke: ' + ', '.join(self.selected_items))

    def get_selected_items(self):
        """
        :returns List[str]: Returns selected items in the checkbox
        """
        return self.selected_items

    def hidePopup(self):
        """
        Prevents checkbox window closing after clicking an item.
        :return None:
        """
        if not self._changed:
            super().hidePopup()
        self._changed = False


def main():
    """
    Main program function.
    :return None:
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
