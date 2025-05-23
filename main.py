from treeqsm import treeqsm
from batch_process import batched
import os
from PySide6.QtCore import QObject,QThread,Signal,Qt,QUrl,QProcess
from PySide6.QtWidgets import QWidget,QGridLayout,QVBoxLayout,QLabel,QMainWindow,QPushButton,QApplication,QTextEdit,QToolButton,QComboBox,QHBoxLayout,QSlider,QFileDialog,QMessageBox,QTableWidget,QTableWidgetItem
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QPixmap
from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView 
from tools.define_input import define_input
from Utils.Utils import load_point_cloud
from plotting.point_cloud_plotting import point_cloud_plotting
from plotting.qsm_plotting import qsm_plotting
import numpy as np
import multiprocessing as mp
import sys
# from plotly.graph_objects import Figure, Scatter
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
import time

# os.environ['QT_DEBUG_PLUGINS']='1'
class QSMWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.stacked_widget = QStackedWidget(self)
        self.setWindowTitle("TreeQSM")
        self.setGeometry(100, 100, 700, 300)
        # layout.addWidget(self.stacked_widget)

        # Create a button to start the batch processing
        self.button= QPushButton("Batch Processing (Select Folder)", self)
        self.button.clicked.connect(self.start_batch_processing)
        self.button.setGeometry(50, 200, 250, 50)

        # Create a button to start the single file processing
        self.button2 = QPushButton("Single File Processing (Select File)", self)
        self.button2.clicked.connect(self.start_single_file_processing)
        self.button2.setGeometry(350, 200, 250, 50)

        #slider for intensity threshold
        Label = QLabel("Intensity Threshold:", self)
        Label.setGeometry(50, 10, 200, 30)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setGeometry(50, 50, 200, 30)
        self.slider.setRange(0, 65535 )
        self.slider.setValue(0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(100)
        self.slider.setTickInterval(2000)

        self.threshold_label = QLabel("0", self)
        self.threshold_label.setGeometry(200, 10, 50, 30)
        self.slider.valueChanged.connect(self.update_threshold_label)
        self.slider.setToolTip("Ser the Intensity threshold, this will automically filter out any points with intensity lower than this value. The default value is 0.")
        self.slider.setToolTipDuration(1000)
        
        #slider for number of min PatchDiam to test
        Label2 = QLabel("Number of Min PatchDiam to Test:", self)
        Label2.setGeometry(50, 100, 250, 30)
        self.slider2 = QSlider(Qt.Horizontal, self) 
        self.slider2.setGeometry(50, 150, 200, 30)
        self.slider2.setRange(1, 10)
        self.slider2.setValue(1)
        self.slider2.setSingleStep(1)
        self.slider2.setPageStep(1)
        self.slider2.setTickInterval(2)
        self.slider2.setToolTip("""Set the number of Min PatchDiam to test. The values will set different min cover set 
                                Patch Diameters for the algorithm to test on the variable size cover set step. 
                                The resulting number of passes of the algorithm will be NInit x NMin x Nmax .The default value is 1. """)
        self.slider2.setToolTipDuration(1000)

        self.min_patchdiam_label = QLabel("1", self)
        self.min_patchdiam_label.setGeometry(300, 100, 50, 30)
        self.slider2.valueChanged.connect(self.update_min_patchdiam_label)


        #slider for number of max PatchDiam to test
        Label3 = QLabel("Number of Max PatchDiam to Test:", self)
        Label3.setGeometry(350, 100, 250, 30)
        self.slider3 = QSlider(Qt.Horizontal, self)
        self.slider3.setGeometry(350, 150, 200, 30)
        self.slider3.setRange(1, 10)
        self.slider3.setValue(1)
        self.slider3.setSingleStep(1)
        self.slider3.setPageStep(1)
        self.slider3.setTickInterval(2)
        self.slider3.setToolTip("""Set the number of Max PatchDiam to test. The values will set different max cover set 
                                Patch Diameters for the algorithm to test on the variable size cover set step. 
                                The resulting number of passes of the algorithm will be NInit x NMin x Nmax. The default value is 1.""")
        self.slider3.setToolTipDuration(1000)

        self.max_patchdiam_label = QLabel("1", self)
        self.max_patchdiam_label.setGeometry(600, 100, 50, 30)
        self.slider3.valueChanged.connect(self.update_max_patchdiam_label)


        #slider for number of Initial PatchDiam to test
        Label4 = QLabel("Number of Initial PatchDiam to Test:", self)
        Label4.setGeometry(350, 10, 250, 30)
        self.slider4 = QSlider(Qt.Horizontal, self)
        self.slider4.setGeometry(350, 50, 200, 30)
        self.slider4.setRange(1, 10)
        self.slider4.setValue(1)
        self.slider4.setSingleStep(1)
        self.slider4.setPageStep(1)
        self.slider4.setTickInterval(2)
        self.slider4.setToolTip("""Set the number of Initial PatchDiam to test. The values will set different initial cover set 
        Patch Diameters for the algorithm to test. The resulting number of passes of the algorithm will be NInit x NMin x Nmax.
        The default value is 1.""")
        self.slider4.setToolTipDuration(1000)

        self.init_patchdiam_label = QLabel("1", self)
        self.init_patchdiam_label.setGeometry(600, 10, 50, 30)
        self.slider4.valueChanged.connect(self.update_init_patchdiam_label)


    
    def update_init_patchdiam_label(self, value):
        self.init_patchdiam_label.setText(str(value))
    def update_max_patchdiam_label(self, value):
        self.max_patchdiam_label.setText(str(value))
    def update_min_patchdiam_label(self, value):
        self.min_patchdiam_label.setText(str(value))
    def update_threshold_label(self, value):
        self.threshold_label.setText(str(value))




    def start_batch_processing(self):
            #prompt user for folder path
        inputs = [self.slider.value(), self.slider2.value(), self.slider3.value(), self.slider4.value()]#Intensity threshold, number of min PatchDiam to test, number of max PatchDiam to test, number of Initial PatchDiam to test
        
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        if not folder:
            QMessageBox.warning(self, "No Folder Selected", "Please select a folder containing LAS or LAZ files.")
            return
        self.batch_window = BatchProcessingWindow(self, folder,inputs)
        self.batch_window.show()
        self.hide()
        # self.batch_window = BatchProcessingWindow(self,folder)
        # self.batch_window.setLayout(QVBoxLayout())
        # self.stacked_widget.addWidget(self.batch_window) 

        # self.stacked_widget.setCurrentWidget(self.batch_window)
    
    def start_single_file_processing(self):
        #prompt user for file path
        inputs = [self.slider.value(), self.slider2.value(), self.slider3.value(), self.slider4.value()]#Intensity threshold, number of min PatchDiam to test, number of max PatchDiam to test, number of Initial PatchDiam to test
        file, _ = QFileDialog.getOpenFileName(self, "Select File", "", "LAS Files (*.las *.laz)")
        if not file:
            QMessageBox.warning(self, "No File Selected", "Please select a LAS or LAZ file.")
            return
        self.single_window = SingleFileProcessingWindow(self, file,inputs)
        self.single_window.show()
        self.hide()   

class BatchProcessingWindow(QMainWindow):
    def __init__(self,root,folder,inputs):
        super().__init__()
        self.setWindowTitle("Batch Processing")
        self.setGeometry(100, 100, 1600, 900)  
        self.root = root
        self.folder = folder


        files = os.listdir(folder)
        
        files = [f for f in files if f.endswith('.las') or f.endswith('.laz')]
        table = QTableWidget()
        table.setRowCount(len(files))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["File Name","Completed"])
        table.clicked.connect(self.table_clicked)
        self.file_data = []
        for i, file in enumerate(files):
            
            item_name = QTableWidgetItem(file)

            table.setItem(i, 0, item_name)
            status = QTableWidgetItem("Not Completed")
            self.file_data.append({'file': file, 'status': status})
        self.files = files
        # self.hide()
        self.intensity_threshold = inputs[0]
        self.nPD2Min = inputs[1]
        self.nPD2Max = inputs[2]
        self.nPD1 = inputs[3]
        self.initial_inputs = {'PatchDiam1': self.nPD1,
                        'PatchDiam2Min': self.nPD2Min,
                        'PatchDiam2Max': self.nPD2Max}
        self.ui = QWidget()

        self.ui.setLayout(QGridLayout())
        self.ui.layout().addWidget(table,0,0)
        self.setCentralWidget(self.ui)

        self.ui.layout().setColumnStretch(0,1)
        self.ui.layout().setColumnStretch(1,3)
        # self.info.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        # self.info.setWindowTitle("Inputs")
        self.ui.layout().setSpacing(50)


    

        self.info =QWidget()
        self.info.setLayout(QVBoxLayout())
        self.info.layout()
        self.label2 = QLabel(f"Intensity Threshold: {self.intensity_threshold}")
        # self.label2.setGeometry(50, 30, 200, 30)
        self.label2.setStyleSheet("font-size: 16px;")
        self.info.layout().addWidget(self.label2)
        self.label3 = QLabel(f"Number of Initial PatchDiam to Test: {self.nPD1}")
        # self.label3.setGeometry(50, 50, 200, 30)
        self.label3.setStyleSheet("font-size: 16px;")
        self.info.layout().addWidget(self.label3)
        self.label4 = QLabel(f"Number of Min PatchDiam to Test: {self.nPD2Min}")
        # self.label4.setGeometry(50, 70, 200, 30)
        self.label4.setStyleSheet("font-size: 16px;")
        self.info.layout().addWidget(self.label4)
        self.label5 = QLabel(f"Number of Max PatchDiam to Test: {self.nPD2Max}")
        self.info.layout().addWidget(self.label5)
        


        self.nav_buttons = QWidget()
        self.nav_buttons.setLayout(QHBoxLayout())
        self.ui.layout().addWidget(self.nav_buttons, 2, 1)
        # Create a left arrow button
        self.left_arrow_button = QToolButton()
        self.left_arrow_button.setArrowType(Qt.ArrowType.LeftArrow)
        self.nav_buttons.layout().addWidget(self.left_arrow_button)
        self.left_arrow_button.clicked.connect(self.left_button_clicked)
        self.left_arrow_button.setEnabled(False)

        # Create a right arrow button
        self.right_arrow_button = QToolButton()
        self.right_arrow_button.setArrowType(Qt.ArrowType.RightArrow)
        self.nav_buttons.layout().addWidget(self.right_arrow_button)
        self.right_arrow_button.clicked.connect(self.right_button_clicked)
        self.right_arrow_button.setEnabled(False)

        self.buttons_and_progress = QWidget()
        self.buttons_and_progress.setLayout(QVBoxLayout())
        self.ui.layout().addWidget(self.buttons_and_progress, 1, 0)
        self.buttons_and_progress.layout().addWidget(self.info)


        self.screen_buttons = QWidget()
        self.screen_buttons.setLayout(QVBoxLayout())
        self.buttons_and_progress.layout().addWidget(self.screen_buttons)

        self.point_cloud_button = QPushButton("Show Point Cloud", self)
        self.point_cloud_button.clicked.connect(self.show_point_cloud)
        self.screen_buttons.layout().addWidget(self.point_cloud_button)


        self.tree_data_button = QPushButton("Show Tree Data", self)
        self.tree_data_button.clicked.connect(self.show_tree_data)
        self.screen_buttons.layout().addWidget(self.tree_data_button)
        self.tree_data_button.setEnabled(False)

        self.segment_plot_button = QPushButton("Show Segment Plot", self)
        self.segment_plot_button.clicked.connect(self.show_segment_plot)
        self.screen_buttons.layout().addWidget(self.segment_plot_button)
        self.segment_plot_button.setEnabled(False)

        self.cylinder_plot_button = QPushButton("Show Cylinder Plot", self)
        self.cylinder_plot_button.clicked.connect(self.show_cylinder_plot)
        self.screen_buttons.layout().addWidget(self.cylinder_plot_button)
        self.cylinder_plot_button.setEnabled(False)

        self.combo_boxes = QWidget()
        self.combo_boxes.setLayout(QHBoxLayout())
        self.screen_buttons.layout().addWidget(self.combo_boxes)
        self.npd1_label = QLabel("PatchDiam1:")
        self.npd1_combo = QComboBox()

        self.max_pd_label = QLabel("Max PatchDiam:")
        self.max_pd_combo = QComboBox()
        self.min_pd_label = QLabel("Min PatchDiam:")
        self.min_pd_combo = QComboBox()

        self.npd1_combo.setEnabled(False)
        self.max_pd_combo.setEnabled(False)
        self.min_pd_combo.setEnabled(False)
        self.npd1_combo.setToolTip("Select the PatchDiam1 to display")
        self.max_pd_combo.setToolTip("Select the Max PatchDiam to display")
        self.min_pd_combo.setToolTip("Select the Min PatchDiam to display")
        self.combo_boxes.layout().addWidget(self.npd1_label)
        self.combo_boxes.layout().addWidget(self.npd1_combo)
        self.combo_boxes.layout().addWidget(self.max_pd_label)
        self.combo_boxes.layout().addWidget(self.max_pd_combo)
        self.combo_boxes.layout().addWidget(self.min_pd_label)
        self.combo_boxes.layout().addWidget(self.min_pd_combo)


        self.cloud_web_view = QWebEngineView()
        # self.web_view.setHtml(html)
        self.ui.layout().addWidget(self.cloud_web_view, 0, 1,2,1)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.buttons_and_progress.layout().addWidget(self.text_edit)



        self.button = QPushButton("Start Processing", self)
        self.button.clicked.connect(self.process_files)
        self.ui.layout().addWidget(self.button,2,0)
    
        self.selected_index = 0
        self.num_screens = 1
        self.screen = 0
        self.data_canvas = None
        self.seg_web_view = None
        self.cyl_web_view = None
    


    def table_clicked(self, index):
        # Get the selected row
        self.selected_index = index.row()
        # Get the selected item
        item = self.sender().item(index.row(), 0)
        # Get the text of the selected item 
        file_name = item.text()
        # Get the status of the selected item
        status = self.file_data[index.row()]['status']
        self.append_text(f"Selected file: {file_name}\n")
        if status == "Completed":
            self.tree_data_button.setEnabled(True)
            self.segment_plot_button.setEnabled(True)
            self.cylinder_plot_button.setEnabled(True)
            self.npd1_combo.setEnabled(True)
            self.max_pd_combo.setEnabled(True)
            self.min_pd_combo.setEnabled(True)
            npd1 = self.inputs[self.selected_index]['PatchDiam1']
            self.npd1_combo.clear()
            self.npd1_combo.addItems([str(i) for i in npd1])
            max_pd = self.inputs[self.selected_index]['PatchDiam2Max']
            self.max_pd_combo.clear()
            self.max_pd_combo.addItems([str(i) for i in max_pd])
            min_pd = self.inputs[self.selected_index]['PatchDiam2Min']
            self.min_pd_combo.clear()
            self.min_pd_combo.addItems([str(i) for i in min_pd])
        else:
            self.tree_data_button.setEnabled(False)
            self.segment_plot_button.setEnabled(False)
            self.cylinder_plot_button.setEnabled(False)
            self.npd1_combo.setEnabled(False)
            self.max_pd_combo.setEnabled(False)
            self.min_pd_combo.setEnabled(False)
        self.show_point_cloud()

    def get_selected_inputs(self):
        npd1 = max(self.npd1_combo.currentIndex(),0)
        max_pd = max(self.max_pd_combo.currentIndex(),0)
        min_pd = max(self.min_pd_combo.currentIndex(),0)
        index = int(npd1)*self.nPD2Max*self.nPD2Min + int(max_pd)*self.nPD2Min + int(min_pd)
        return index

    def left_button_clicked(self):
        # Handle left button click
        if self.screen <= 0:

            return
        self.screen -= 1
        if self.screen == 0:
            self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(True)
        self.display_tree_data(self.file_data[self.selected_index]['QSM'][self.get_selected_inputs()]['treedata']['figures'][self.screen])


    def right_button_clicked(self): 
        # Handle right button click
        print(self.screen,self.num_screens)
        if self.screen >= self.num_screens - 1:
            return
        self.screen += 1
        if self.screen == self.num_screens - 1:
            self.right_arrow_button.setEnabled(False)
        self.left_arrow_button.setEnabled(True)
        self.display_tree_data(self.file_data[self.selected_index]['QSM'][self.get_selected_inputs()]['treedata']['figures'][self.screen])

    def display_tree_data(self,figure):
        if self.data_canvas != None:
            self.ui.layout().removeWidget(self.data_canvas)
            self.data_canvas.deleteLater()
        figure.dpi = 100
        self.data_canvas = FigureCanvas(figure)
        self.ui.layout().addWidget(self.data_canvas, 0, 1,2,1)

    def show_tree_data(self):
        self.append_text("Showing Tree Data...\n")
        self.left_arrow_button.setEnabled(False)
        self.screen = 0
        self.num_screens = len(self.file_data[self.selected_index]['QSM'][self.get_selected_inputs()]['treedata']['figures'])
        self.display_tree_data(self.file_data[self.selected_index]['QSM'][self.get_selected_inputs()]['treedata']['figures'][self.screen])
        self.right_arrow_button.setEnabled(True)
    
    def show_point_cloud(self):
        
        self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(False)
        cloud = self.file_data[self.selected_index].get('cloud',None)
        if cloud is None:
            self.append_text("Loading Point Cloud...\n")
            file = os.path.join(self.folder, self.file_data[self.selected_index]['file'])
            cloud = load_point_cloud(file)
            self.file_data[self.selected_index]['cloud'] = cloud
        html = point_cloud_plotting(cloud,subset=True)
        self.cloud_web_view = QWebEngineView()
        self.cloud_web_view.load(QUrl.fromLocalFile(os.getcwd()+"/"+html))
        self.ui.layout().addWidget(self.cloud_web_view, 0, 1,2,1)

    def show_segment_plot(self):
        self.append_text("Showing Segment Plot...\n")
        self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(False)
        index = self.get_selected_inputs()
        qsm = self.file_data[self.selected_index]['QSM'][index]
        cover = qsm['cover']
        segments = qsm['segment']
        html = qsm_plotting(self.file_data[self.selected_index]['cloud'], cover, segments,qsm)
        self.seg_web_view = QWebEngineView()
        self.seg_web_view.load(QUrl.fromLocalFile(os.getcwd()+"/"+html))
        self.ui.layout().addWidget(self.seg_web_view, 0, 1,2,1)

    def show_cylinder_plot(self):
        self.append_text("Showing Cylinder Plot...\n")
        self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(False)
        index = self.get_selected_inputs()

        html = self.file_data[self.selected_index]['plot'][index]
        self.cyl_web_view = QWebEngineView()
        self.cyl_web_view.load(QUrl.fromLocalFile(os.getcwd()+"/"+html))
        self.ui.layout().addWidget(self.cyl_web_view, 0, 1,2,1)

    
        

    def closeEvent(self, event):
        # Handle the close event
        self.root.show()
        event.accept()

    def append_text(self, text):
        self.text_edit.insertPlainText(text)

    def complete_processing(self,package):
        
        index, data, plot = package
        self.file_data[index]['status']="Completed"
        self.file_data[index]['QSM'] = data
        self.file_data[index]['plot'] = plot
        self.tree_data_button.setEnabled(True)
        self.segment_plot_button.setEnabled(True)
        self.cylinder_plot_button.setEnabled(True)
        self.npd1_combo.setEnabled(True)
        self.max_pd_combo.setEnabled(True)
        self.min_pd_combo.setEnabled(True)

        if index == self.selected_index:
            npd1 = self.inputs[self.selected_index]['PatchDiam1']
            self.npd1_combo.clear()
            self.npd1_combo.addItems([str(i) for i in npd1])
            max_pd = self.inputs[self.selected_index]['PatchDiam2Max']
            self.max_pd_combo.clear()
            self.max_pd_combo.addItems([str(i) for i in max_pd])
            min_pd = self.inputs[self.selected_index]['PatchDiam2Min']
            self.min_pd_combo.clear()
            self.min_pd_combo.addItems([str(i) for i in min_pd])

        
        self.append_text(f"Processing Complete for {self.file_data[index]['file']}...\n")

    def add_cloud(self,package):
        index, cloud = package
        self.file_data[index]['cloud'] = cloud
        self.append_text(f"Loaded {self.file_data[index]['file']} with {cloud.shape[0]} points\n")
        
    def set_inputs(self,inputs):
        
        self.inputs = inputs
    def process_files(self):
        self.append_text("Processing file...\n")
        
        task = BatchQSM(self,self.folder,self.files,self.intensity_threshold, self.initial_inputs)
        self.qsm_thread = BackgroundProcess(task)
        task.finished.connect(self.complete_processing)
        task.message.connect(self.append_text)
        task.plot_data.connect(self.add_cloud)
        task.input_list.connect(self.set_inputs)
        self.qsm_thread.start()

class SingleFileProcessingWindow(QMainWindow):
    def __init__(self,root,file,inputs):
        super().__init__()
        self.setWindowTitle("Single File Processing")
        self.setGeometry(100, 100, 1920, 1080)  
        self.root = root
        self.args = inputs
        self.initModel(file,inputs)


        self.ui = QWidget()
        self.ui.setLayout(QGridLayout())
        self.info =QWidget()
        self.info.setLayout(QVBoxLayout())
        self.info.layout()
        self.ui.layout().setColumnStretch(0,1)
        self.ui.layout().setColumnStretch(1,3)
        self.info.setMaximumHeight(200)
        # self.info.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        # self.info.setWindowTitle("Inputs")
        self.info.layout().setSpacing(0)
        self.info.setGeometry(50, 50, 200, 200)
        self.ui.layout().addWidget(self.info,0,0)
        self.ui.layout().setSpacing(100)
        
        #Display inputs in top left corner
        self.label = QLabel("Inputs:")
        # self.label.setGeometry(50, 10, 200, 30)
        self.label.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.info.layout().addWidget(self.label)
        self.label2 = QLabel(f"Intensity Threshold: {self.intensity_threshold}")
        # self.label2.setGeometry(50, 30, 200, 30)
        self.label2.setStyleSheet("font-size: 16px;")
        self.info.layout().addWidget(self.label2)
        self.label3 = QLabel(f"Number of Initial PatchDiam to Test: {self.nPD1}")
        # self.label3.setGeometry(50, 50, 200, 30)
        self.label3.setStyleSheet("font-size: 16px;")
        self.info.layout().addWidget(self.label3)
        self.label4 = QLabel(f"Number of Min PatchDiam to Test: {self.nPD2Min}")
        # self.label4.setGeometry(50, 70, 200, 30)
        self.label4.setStyleSheet("font-size: 16px;")
        self.info.layout().addWidget(self.label4)
        self.label5 = QLabel(f"Number of Max PatchDiam to Test: {self.nPD2Max}")
        # self.label5.setGeometry(50, 90, 200, 30)
        self.label5.setStyleSheet("font-size: 16px;")
        self.info.layout().addWidget(self.label5)
        self.label6 = QLabel(f"File Name: {self.file}")
        # self.label6.setGeometry(50, 110, 200, 30)
        self.label6.setStyleSheet("font-size: 16px;")
        self.info.layout().addWidget(self.label6)
        self.setCentralWidget(self.ui)


        

        self.button = QPushButton("Start Processing", self)
        self.button.clicked.connect(self.process_file)
        self.ui.layout().addWidget(self.button,2,0)

        

        self.nav_buttons = QWidget()
        self.nav_buttons.setLayout(QHBoxLayout())
        self.ui.layout().addWidget(self.nav_buttons, 2, 1)
        # Create a left arrow button
        self.left_arrow_button = QToolButton()
        self.left_arrow_button.setArrowType(Qt.ArrowType.LeftArrow)
        self.nav_buttons.layout().addWidget(self.left_arrow_button)
        self.left_arrow_button.clicked.connect(self.left_button_clicked)
        self.left_arrow_button.setEnabled(False)

        # Create a right arrow button
        self.right_arrow_button = QToolButton()
        self.right_arrow_button.setArrowType(Qt.ArrowType.RightArrow)
        self.nav_buttons.layout().addWidget(self.right_arrow_button)
        self.right_arrow_button.clicked.connect(self.right_button_clicked)
        self.right_arrow_button.setEnabled(False)

        self.buttons_and_progress = QWidget()
        self.buttons_and_progress.setLayout(QVBoxLayout())
        self.ui.layout().addWidget(self.buttons_and_progress, 1, 0)


        self.screen_buttons = QWidget()
        self.screen_buttons.setLayout(QVBoxLayout())
        self.buttons_and_progress.layout().addWidget(self.screen_buttons)
        
        self.point_cloud_button = QPushButton("Show Point Cloud", self)
        self.point_cloud_button.clicked.connect(self.show_point_cloud)
        self.screen_buttons.layout().addWidget(self.point_cloud_button)
        
        
        self.tree_data_button = QPushButton("Show Tree Data", self)
        self.tree_data_button.clicked.connect(self.show_tree_data)
        self.screen_buttons.layout().addWidget(self.tree_data_button)
        self.tree_data_button.setEnabled(False)

        self.segment_plot_button = QPushButton("Show Segment Plot", self)
        self.segment_plot_button.clicked.connect(self.show_segment_plot)
        self.screen_buttons.layout().addWidget(self.segment_plot_button)
        self.segment_plot_button.setEnabled(False)

        self.cylinder_plot_button = QPushButton("Show Cylinder Plot", self)
        self.cylinder_plot_button.clicked.connect(self.show_cylinder_plot)
        self.screen_buttons.layout().addWidget(self.cylinder_plot_button)
        self.cylinder_plot_button.setEnabled(False)

        self.combo_boxes = QWidget()
        self.combo_boxes.setLayout(QHBoxLayout())
        self.screen_buttons.layout().addWidget(self.combo_boxes)
        self.npd1_label = QLabel("PatchDiam1:")
        self.npd1_combo = QComboBox()

        self.max_pd_label = QLabel("Max PatchDiam:")
        self.max_pd_combo = QComboBox()
        self.min_pd_label = QLabel("Min PatchDiam:")
        self.min_pd_combo = QComboBox()
        
        self.npd1_combo.setEnabled(False)
        self.max_pd_combo.setEnabled(False)
        self.min_pd_combo.setEnabled(False)
        self.npd1_combo.setToolTip("Select the PatchDiam1 to display")
        self.max_pd_combo.setToolTip("Select the Max PatchDiam to display")
        self.min_pd_combo.setToolTip("Select the Min PatchDiam to display")
        self.combo_boxes.layout().addWidget(self.npd1_label)
        self.combo_boxes.layout().addWidget(self.npd1_combo)
        self.combo_boxes.layout().addWidget(self.max_pd_label)
        self.combo_boxes.layout().addWidget(self.max_pd_combo)
        self.combo_boxes.layout().addWidget(self.min_pd_label)
        self.combo_boxes.layout().addWidget(self.min_pd_combo)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.buttons_and_progress.layout().addWidget(self.text_edit)

        self.output_text = OutputText(self.text_edit)
        sys.stdout = self.output_text
        #Exit behavior
        


        


        self.show_point_cloud()
        self.plots = None
        self.data = None
        self.data_canvas = None
        self.seg_web_view = None
        self.cyl_web_view = None
        # self.button.setGeometry(300, 500, 200, 50)
        self.screen = 0
        self.num_screens = 1


    def initModel(self,file,inputs):
        self.file = file
        self.intensity_threshold = inputs[0]
        self.nPD2Min = inputs[1]
        self.nPD2Max = inputs[2]
        self.nPD1 = inputs[3]
        self.points = load_point_cloud(self.file)

        # Step 3: Define inputs for TreeQSM
        self.points = self.points - np.mean(self.points,axis = 0)
        self.inputs = define_input(self.file,self.nPD1, self.nPD2Min, self.nPD2Max)[0]
        self.inputs['plot']=0
    
    def show_point_cloud(self):
        self.append_text("Showing Point Cloud...\n")
        html = point_cloud_plotting(self.points,subset=True)

        self.cloud_web_view = QWebEngineView()
        self.cloud_web_view.load(QUrl.fromLocalFile(os.getcwd()+"/"+html))
        # self.web_view.setHtml(html)
        self.ui.layout().addWidget(self.cloud_web_view, 0, 1,2,1)
        self.cloud_web_view.show()
        self.num_screens = 1
        self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(False)

    def left_button_clicked(self):
        # Handle left button click
        if self.screen <= 0:

            return
        self.screen -= 1
        if self.screen == 0:
            self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(True)
        self.display_tree_data(self.tree_data[self.screen])


    def right_button_clicked(self): 
        # Handle right button click
        if self.screen >= self.num_screens - 1:
            return
        self.screen += 1
        if self.screen == self.num_screens - 1:
            self.right_arrow_button.setEnabled(False)
        self.left_arrow_button.setEnabled(True)
        self.display_tree_data(self.tree_data[self.screen])

    def show_tree_data(self):
        self.append_text("Showing Tree Data...\n")
        self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(False)
        index = self.get_selected_index()
        self.tree_data = self.data[index]['treedata']['figures']
        self.screen = 0
        self.num_screens = len(self.tree_data)
        self.display_tree_data(self.tree_data[0])
        self.right_arrow_button.setEnabled(True)


    def display_tree_data(self,figure):
        if self.data_canvas != None:
            self.ui.layout().removeWidget(self.data_canvas)
            self.data_canvas.deleteLater()
        figure.dpi = 100
        self.data_canvas = FigureCanvas(figure)

        self.ui.layout().addWidget(self.data_canvas, 0, 1,2,1)
        self.data_canvas.show()

    def show_segment_plot(self):
        self.append_text("Showing Segment Plot...\n")
        self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(False)
        index = self.get_selected_index()
        qsm = self.data[index]
        cover = qsm['cover']
        segments = qsm['segment']
        html = qsm_plotting(self.points, cover, segments,qsm)
        self.seg_web_view = QWebEngineView()
        self.seg_web_view.load(QUrl.fromLocalFile(os.getcwd()+"/"+html))
        self.ui.layout().addWidget(self.seg_web_view, 0, 1,2,1)


    def show_cylinder_plot(self):   
        self.append_text("Showing Cylinder Plot, this may take a few moments...\n")
        self.left_arrow_button.setEnabled(False)
        self.right_arrow_button.setEnabled(False)
        index = self.get_selected_index()
        self.cyl_web_view = QWebEngineView()
        html = self.cyl_plots[index]
        self.cyl_web_view.load(QUrl.fromLocalFile(os.getcwd()+"/"+html))
        self.ui.layout().addWidget(self.cyl_web_view, 0, 1,2,1)




    def append_text(self, text):
        self.text_edit.insertPlainText(text)

    def process_file(self):
        self.append_text("Processing file. This may take several minutes...\n")
        
        task = SingleQSM(self,self.points,self.inputs)
        self.qsm_thread = BackgroundProcess(task)
        task.finished.connect(self.complete_processing)
        self.qsm_thread.start()


    def get_selected_index(self):
        npd1 = max(self.npd1_combo.currentIndex(),0)
        max_pd = max(self.max_pd_combo.currentIndex(),0)
        min_pd = max(self.min_pd_combo.currentIndex(),0)
        index = int(npd1)*self.nPD2Max*self.nPD2Min + int(max_pd)*self.nPD2Min + int(min_pd)
        return index

    def complete_processing(self,package):
        self.append_text("Processing Complete...\n")

        data,plot = package
        self.cyl_plots =plot
        self.data=data


        self.tree_data_button.setEnabled(True)
        self.segment_plot_button.setEnabled(True)
        self.cylinder_plot_button.setEnabled(True)
        self.npd1_combo.setEnabled(True)
        self.max_pd_combo.setEnabled(True)
        self.min_pd_combo.setEnabled(True)

        npd1=self.inputs['PatchDiam1']
        self.npd1_combo.addItems([str(i) for i in npd1])
        max_pd = self.inputs['PatchDiam2Max']
        self.max_pd_combo.addItems([str(i) for i in max_pd])
        min_pd = self.inputs['PatchDiam2Min']
        self.min_pd_combo.addItems([str(i) for i in min_pd])

        

    def closeEvent(self, event):
        # Handle the close event
        self.root.show()
        event.accept()


class BatchQSM(QObject):
    finished = Signal(tuple)
    plot_data = Signal(tuple)
    message = Signal(str)
    input_list = Signal(list)
    def __init__(self, root, folder,files,threshold,inputs):
        super().__init__()
        self.root = root
        self.folder = folder
        self.files = files
        self.intensity_threshold = threshold
        self.inputs = inputs
        # self.process_file()

    def run(self):
        clouds = []
        for i, file in enumerate(self.files):
            point_cloud = load_point_cloud(os.path.join(self.folder, file), self.intensity_threshold)
            if point_cloud is not None:
                point_cloud = point_cloud - np.mean(point_cloud,axis = 0)
                clouds.append(point_cloud)
                self.plot_data.emit((i,point_cloud))
        inputs = define_input(clouds,self.inputs['PatchDiam1'], self.inputs['PatchDiam2Min'], self.inputs['PatchDiam2Max'])
        self.input_list.emit(inputs)
        for i, input_params in enumerate(inputs):
            input_params['name'] = self.files[i]
            input_params['savemat'] = 0
            input_params['savetxt'] = 1
            input_params['plot'] = 0
        
    # Process each tree
        for i, input_params in enumerate(inputs):
            self.message.emit(f"Processing {input_params['name']}. This may take several minutes...\n")
            try:
                data,plot = treeqsm(clouds[i],input_params,i)
                finished = self.finished.emit((i,data,plot)) 
            except:
                self.message.emit(f"An error occured on file {input_params['name']}. Please try again. Consider checking the console and reporting the bug to us.")  
                
            
            
        self.message.emit("Processing Complete.\n")


class SingleQSM(QObject):
    finished = Signal(tuple)
    def __init__(self, root, points, inputs):
        super().__init__()
        self.root = root
        self.points = points
        self.inputs = inputs
        
        # self.process_file()



    def run(self):
        try:
            mp.set_start_method('spawn')
        except:
            pass
        q = mp.Queue()
        p = mp.Process(target=treeqsm, args=(self.points,self.inputs,0,q))
        p.start()
        
        batch,data,plot = q.get()
        p.join()
        # data,plot = treeqsm(self.points,self.inputs)
        
        finished = self.finished.emit((data,plot))

class BackgroundProcess(QThread):
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        worker.moveToThread(self)
        self.started.connect(worker.run)

    def run(self):
        super().run()
        self.worker = None
             
class OutputSignal(QObject):
    printOccured = Signal(str)

class OutputText(object):
    def __init__(self, text_edit):
        self.text_edit = text_edit
        self.output_signal = OutputSignal()
        self.output_signal.printOccured.connect(self.append_text)
    
    def write(self, text):
        self.output_signal.printOccured.emit(text)
    
    def flush(self):
        pass
    
    def append_text(self, text):
         self.text_edit.insertPlainText(text)


  

if __name__ == "__main__":
    app = QApplication([])
    window = QSMWindow()
    window.show()
    # Start the application

    app.exec()