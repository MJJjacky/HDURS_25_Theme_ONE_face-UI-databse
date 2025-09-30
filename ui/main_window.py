import cv2
import numpy as np
import time
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QTableWidget, QTableWidgetItem,
                             QTabWidget, QGroupBox, QMessageBox, QInputDialog, QDialog,
                             QMenu, QComboBox) # Added QSizePolicy
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
import os # Added for file system operations

# 使用绝对导入
from face_recognition.face_detector import FaceDetector
from face_recognition.face_recognizer import FaceRecognizer
from database.database_manager import DatabaseManager
from serial_communication import SerialCommunication

class TrainingDialog(QDialog):
    """人脸训练对话框"""
    
    def __init__(self, face_detector, face_recognizer, db_manager, user_id, user_name, parent=None):
        super().__init__(parent)
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.db_manager = db_manager
        self.user_id = user_id
        self.user_name = user_name
        self.samples = []
        self.saved_images = []  # 保存的图片路径
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_frame)
        
        # 确保人脸图片目录存在
        self.face_images_dir = "data/faces"
        os.makedirs(self.face_images_dir, exist_ok=True)
        
        self.init_ui()
        self.start_camera()
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle(f"训练用户: {self.user_name}")
        self.setGeometry(200, 200, 800, 600)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 用户信息
        info_label = QLabel(f"正在训练用户: {self.user_name} (ID: {self.user_id})")
        info_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(info_label)
        
        # 摄像头显示
        camera_group = QGroupBox("摄像头")
        camera_layout = QVBoxLayout(camera_group)
        
        self.camera_label = QLabel("摄像头未启动")
        self.camera_label.setMinimumSize(400, 300)
        self.camera_label.setStyleSheet("border: 2px solid #bdc3c7; background-color: #ecf0f1;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(self.camera_label)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        
        self.capture_btn = QPushButton("采集样本")
        self.capture_btn.clicked.connect(self.capture_sample)
        
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        
        self.clear_btn = QPushButton("清空样本")
        self.clear_btn.clicked.connect(self.clear_samples)
        
        button_layout.addWidget(self.capture_btn)
        button_layout.addWidget(self.train_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # 样本信息
        sample_group = QGroupBox("样本信息")
        sample_layout = QVBoxLayout(sample_group)
        
        self.sample_info_label = QLabel("样本数量: 0")
        sample_layout.addWidget(self.sample_info_label)
        
        sample_group.setLayout(sample_layout)
        layout.addWidget(sample_group)
    
    def save_face_image(self, face_image, index):
        """保存人脸图片到文件系统"""
        # 创建用户目录
        user_dir = os.path.join(self.face_images_dir, self.user_name)
        os.makedirs(user_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.user_name}_{timestamp}_{index:03d}.jpg"
        filepath = os.path.join(user_dir, filename)
        
        # 保存图片
        cv2.imwrite(filepath, face_image)
        print(f"保存人脸图片: {filepath}")
        
        return filepath
    
    def start_camera(self):
        """启动摄像头"""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头！")
            return
        
        self.timer.start(30)
    
    def stop_camera(self):
        """停止摄像头"""
        if self.camera:
            self.timer.stop()
            self.camera.release()
            self.camera = None
    
    def update_camera_frame(self):
        """更新摄像头帧"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                # 人脸检测
                faces = self.face_detector.detect_faces(frame)
                
                # 绘制检测框
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face Detected", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 转换帧格式并显示
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                
                # 调整大小以适应标签
                scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio)
                self.camera_label.setPixmap(scaled_pixmap)
    
    def capture_sample(self):
        """采集样本"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                # 人脸检测
                faces = self.face_detector.detect_faces(frame)
                
                if len(faces) > 0:
                    # 获取最大的人脸
                    largest_face = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = largest_face
                    
                    # 提取人脸区域
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # 保存人脸图片
                    saved_path = self.save_face_image(face_roi, len(self.samples))
                    self.saved_images.append(saved_path)
                    
                    # 添加到样本列表
                    self.samples.append(face_roi)
                    
                    # 更新样本信息
                    self.sample_info_label.setText(f"样本数量: {len(self.samples)}")
                    
                    # 如果样本足够，启用训练按钮
                    if len(self.samples) >= 5:
                        self.train_btn.setEnabled(True)
                    
                    QMessageBox.information(self, "成功", f"已采集第 {len(self.samples)} 个样本")
                else:
                    QMessageBox.warning(self, "警告", "未检测到人脸，请调整位置")
    
    def clear_samples(self):
        """清空样本"""
        self.samples.clear()
        self.saved_images.clear()
        self.sample_info_label.setText("样本数量: 0")
        self.train_btn.setEnabled(False)
        QMessageBox.information(self, "成功", "样本已清空")
    
    def start_training(self):
        """开始训练"""
        if len(self.samples) < 5:
            QMessageBox.warning(self, "警告", "样本数量不足，至少需要5个样本")
            return
        
        try:
            # 开始训练
            QMessageBox.information(self, "训练", "开始训练人脸识别模型...")
            
            # 保存训练数据到数据库
            self.save_training_data_to_db()
            
            # 添加训练样本到识别器
            for sample in self.samples:
                self.face_recognizer.add_training_sample(sample, self.user_name)
            
            # 开始训练
            success = self.face_recognizer.train()
            
            if success:
                QMessageBox.information(self, "成功", f"用户 {self.user_name} 的人脸识别模型训练成功！")
                self.accept()  # 关闭对话框
            else:
                QMessageBox.warning(self, "失败", "训练失败，请重试")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"训练失败: {e}")
    
    def save_training_data_to_db(self):
        """保存训练数据到数据库"""
        try:
            # 保存人脸图片路径到数据库
            for image_path in self.saved_images:
                self.db_manager.add_face_image(self.user_id, image_path, self.user_name)
            
            print(f"训练数据已保存到数据库，用户ID: {self.user_id}")
            
        except Exception as e:
            print(f"保存到数据库失败: {e}")
    
    def closeEvent(self, event):
        """关闭事件"""
        self.stop_camera()
        event.accept()

class UserEditDialog(QDialog):
    """用户编辑对话框"""
    
    def __init__(self, user_data, parent=None):
        super().__init__(parent)
        self.user_data = user_data  # (id, name, age, gender, create_time)
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle(f"编辑用户: {self.user_data[1]}")
        self.setGeometry(300, 200, 400, 300)
        self.setModal(True)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 用户信息显示
        info_group = QGroupBox("当前用户信息")
        info_layout = QVBoxLayout()
        
        info_layout.addWidget(QLabel(f"用户ID: {self.user_data[0]}"))
        info_layout.addWidget(QLabel(f"姓名: {self.user_data[1]}"))
        info_layout.addWidget(QLabel(f"年龄: {self.user_data[2]}"))
        info_layout.addWidget(QLabel(f"性别: {self.user_data[3]}"))
        info_layout.addWidget(QLabel(f"创建时间: {self.user_data[4]}"))
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        self.modify_id_btn = QPushButton("修改ID")
        self.modify_id_btn.clicked.connect(self.modify_user_id)
        
        self.modify_info_btn = QPushButton("修改信息")
        self.modify_info_btn.clicked.connect(self.modify_user_info)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.modify_id_btn)
        button_layout.addWidget(self.modify_info_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
    
    def modify_user_id(self):
        """修改用户ID"""
        new_id, ok = QInputDialog.getInt(
            self, "修改用户ID", 
            f"请输入新的用户ID (当前: {self.user_data[0]}):",
            self.user_data[0], 1, 999, 1
        )
        if ok:
            try:
                from database.database_manager import DatabaseManager
                db = DatabaseManager()
                if db.modify_user_id(self.user_data[0], new_id):
                    QMessageBox.information(self, "成功", f"用户ID已从 {self.user_data[0]} 修改为 {new_id}")
                    self.user_data = (new_id, self.user_data[1], self.user_data[2], self.user_data[3], self.user_data[4])
                    self.accept()
                else:
                    QMessageBox.warning(self, "错误", "修改用户ID失败！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"修改失败: {e}")
    
    def modify_user_info(self):
        """修改用户基本信息"""
        # 修改姓名
        new_name, ok = QInputDialog.getText(
            self, "修改姓名", 
            f"请输入新的姓名 (当前: {self.user_data[1]}):",
            text=self.user_data[1]
        )
        if not ok:
            return
        
        # 修改年龄
        new_age, ok = QInputDialog.getInt(
            self, "修改年龄", 
            f"请输入新的年龄 (当前: {self.user_data[2]}):",
            self.user_data[2], 1, 120, 1
        )
        if not ok:
            return
        
        # 修改性别
        new_gender, ok = QInputDialog.getItem(
            self, "修改性别", 
            f"请选择性别 (当前: {self.user_data[3]}):",
            ["男", "女"], 0 if self.user_data[3] == "男" else 1, False
        )
        if not ok:
            return
        
        try:
            from database.database_manager import DatabaseManager
            db = DatabaseManager()
            if db.modify_user_info(self.user_data[0], new_name, new_age, new_gender):
                QMessageBox.information(self, "成功", "用户信息修改成功！")
                self.user_data = (self.user_data[0], new_name, new_age, new_gender, self.user_data[4])
                self.accept()
            else:
                QMessageBox.warning(self, "错误", "修改用户信息失败！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"修改失败: {e}")

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        self.cap = None
    
    def run(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            return
        
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            self.msleep(30)
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 初始化数据库管理器
        self.db_manager = DatabaseManager()
        print("数据库管理器初始化完成")
        
        # 测试数据库连接
        try:
            users = self.db_manager.get_all_users()
            print(f"数据库连接测试成功，当前用户数量: {len(users)}")
        except Exception as e:
            print(f"数据库连接测试失败: {e}")
        
        # 每日12点自动刷新糖量数据
        self.daily_refresh_timer = QTimer()
        self.daily_refresh_timer.timeout.connect(self.check_daily_refresh)
        self.start_daily_refresh_timer()
        
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        
        # 串口通信
        self.serial_comm = SerialCommunication()
        
        # 摄像头线程
        self.camera_thread = None
        self.current_frame = None
        self.is_recognition_active = False
        
        # 添加识别控制变量
        self.last_recognition_time = 0
        self.recognition_interval = 0.5  # 每0.5秒识别一次，减少卡顿
        
        # 当前识别的用户信息
        self.current_user_info = None
        
        self.init_ui()
        
        # 启动串口通信
        self.start_serial_communication()
        
        # 设置串口数据更新回调
        if self.serial_comm:
            self.serial_comm.on_data_updated = self.on_serial_data_updated
    
    def init_ui(self):
        self.setWindowTitle("人脸识别健康管理系统")
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # 顶部信息区域（识别结果和用户信息）
        info_panel = self.create_info_panel()
        main_layout.addWidget(info_panel)
        
        # 中间控制面板（功能按键）
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # 底部显示区域（摄像头和标签页）
        display_panel = self.create_display_panel()
        main_layout.addWidget(display_panel)
        
        # 调试信息：检查界面初始化状态
        print("界面初始化完成")
        if hasattr(self, 'tab_widget'):
            print(f"tab_widget已初始化，标签页数量: {self.tab_widget.count()}")
            for i in range(self.tab_widget.count()):
                print(f"标签页 {i}: {self.tab_widget.tabText(i)}")
        else:
            print("tab_widget未初始化")
    
    def create_control_panel(self):
        panel = QWidget()
        layout = QHBoxLayout(panel)  # 改为水平布局
        
        # 摄像头控制
        camera_group = QGroupBox("摄像头控制")
        camera_layout = QVBoxLayout(camera_group)
        
        self.start_camera_btn = QPushButton("启动摄像头")
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.start_camera_btn.setMaximumHeight(30)  # 限制按钮高度
        camera_layout.addWidget(self.start_camera_btn)
        
        self.stop_camera_btn = QPushButton("停止摄像头")
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.stop_camera_btn.setEnabled(False)
        self.stop_camera_btn.setMaximumHeight(30)
        camera_layout.addWidget(self.stop_camera_btn)
        
        layout.addWidget(camera_group)
        
        # 人脸识别控制
        recognition_group = QGroupBox("人脸识别")
        recognition_layout = QVBoxLayout(recognition_group)
        
        self.start_recognition_btn = QPushButton("开始识别")
        self.start_recognition_btn.clicked.connect(self.start_recognition)
        self.start_recognition_btn.setMaximumHeight(30)
        recognition_layout.addWidget(self.start_recognition_btn)
        
        self.stop_recognition_btn = QPushButton("停止识别")
        self.stop_recognition_btn.clicked.connect(self.stop_recognition)
        self.stop_recognition_btn.setEnabled(False)
        self.stop_recognition_btn.setMaximumHeight(30)
        recognition_layout.addWidget(self.stop_recognition_btn)
        
        recognition_group.setLayout(recognition_layout)
        layout.addWidget(recognition_group)
        
        # 用户管理
        user_group = QGroupBox("用户管理")
        user_layout = QVBoxLayout(user_group)
        
        self.add_user_btn = QPushButton("添加用户")
        self.add_user_btn.clicked.connect(self.add_user)
        self.add_user_btn.setMaximumHeight(30)
        user_layout.addWidget(self.add_user_btn)
        
        self.view_users_btn = QPushButton("查看用户")
        self.view_users_btn.clicked.connect(self.view_users)
        self.view_users_btn.setMaximumHeight(30)
        user_layout.addWidget(self.view_users_btn)
        
        user_group.setLayout(user_layout)
        layout.addWidget(user_group)
        
        # 人脸训练
        training_group = QGroupBox("人脸训练")
        training_layout = QVBoxLayout(training_group)
        
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self.start_face_training)
        self.train_btn.setMaximumHeight(30)
        training_layout.addWidget(self.train_btn)
        
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        # 健康记录
        health_group = QGroupBox("健康记录")
        health_layout = QVBoxLayout(health_group)
        
        self.add_health_record_btn = QPushButton("添加健康记录")
        self.add_health_record_btn.clicked.connect(self.add_health_record)
        self.add_health_record_btn.setMaximumHeight(30)
        health_layout.addWidget(self.add_health_record_btn)
        
        # 一键归零今日摄入糖量
        self.reset_sugar_btn = QPushButton("归零今日糖量")
        self.reset_sugar_btn.clicked.connect(self.reset_today_sugar)
        self.reset_sugar_btn.setStyleSheet("background-color: #ff6b6b; color: white; font-weight: bold;")
        self.reset_sugar_btn.setMaximumHeight(30)
        health_layout.addWidget(self.reset_sugar_btn)
        
        health_group.setLayout(health_layout)
        layout.addWidget(health_group)
        
        # 系统状态
        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("系统就绪")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        # 串口配置
        serial_config_layout = QHBoxLayout()
        serial_config_layout.addWidget(QLabel("串口端口:"))
        
        self.serial_port_combo = QComboBox()
        self.serial_port_combo.addItem("/dev/ttyCH341USB0")
        self.serial_port_combo.addItem("/dev/ttyUSB0")
        self.serial_port_combo.addItem("/dev/ttyACM0")
        self.serial_port_combo.setEditable(True)
        self.serial_port_combo.setCurrentText("/dev/ttyCH341USB0")
        self.serial_port_combo.setMaximumHeight(25)
        
        self.refresh_ports_btn = QPushButton("刷新端口")
        self.refresh_ports_btn.clicked.connect(self.refresh_serial_ports)
        self.refresh_ports_btn.setMaximumHeight(25)
        
        serial_config_layout.addWidget(self.serial_port_combo)
        serial_config_layout.addWidget(self.refresh_ports_btn)
        status_layout.addLayout(serial_config_layout)
        
        # 健康信息更新按钮
        self.refresh_health_btn = QPushButton("刷新健康信息")
        self.refresh_health_btn.clicked.connect(self.refresh_health_info)
        self.refresh_health_btn.setMaximumHeight(30)
        status_layout.addWidget(self.refresh_health_btn)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        return panel
    
    def create_info_panel(self):
        """创建顶部信息面板（识别结果和用户信息）"""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        
        # 实际增加糖量
        sugar_added_group = QGroupBox("实际增加糖量")
        sugar_added_layout = QVBoxLayout(sugar_added_group)
        
        self.sugar_added_label = QLabel("等待识别...")
        self.sugar_added_label.setAlignment(Qt.AlignCenter)
        self.sugar_added_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 8px; background-color: #e3f2fd; border: 1px solid #2196f3;")
        self.sugar_added_label.setMinimumHeight(30)
        sugar_added_layout.addWidget(self.sugar_added_label)
        
        sugar_added_group.setLayout(sugar_added_layout)
        layout.addWidget(sugar_added_group)
        
        # 用户信息
        user_info_group = QGroupBox("用户信息")
        user_info_layout = QVBoxLayout(user_info_group)
        
        self.user_info_label = QLabel("用户信息: 等待识别...")
        self.user_info_label.setAlignment(Qt.AlignCenter)
        self.user_info_label.setStyleSheet("font-size: 14px; padding: 6px; background-color: #f0f0f0; border: 1px solid #ccc;")
        self.user_info_label.setMinimumHeight(30)
        user_info_layout.addWidget(self.user_info_label)
        
        user_info_group.setLayout(user_info_layout)
        layout.addWidget(user_info_group)
        
        # 健康信息
        health_info_group = QGroupBox("健康信息")
        health_info_layout = QVBoxLayout(health_info_group)
        
        self.health_info_label = QLabel("健康信息: 等待识别...")
        self.health_info_label.setAlignment(Qt.AlignCenter)
        self.health_info_label.setStyleSheet("font-size: 14px; padding: 6px; background-color: #e8f5e8; border: 1px solid #4caf50;")
        self.health_info_label.setMinimumHeight(30)
        health_info_layout.addWidget(self.health_info_label)
        
        health_info_group.setLayout(health_info_layout)
        layout.addWidget(health_info_group)
        
        # 串口状态
        serial_status_group = QGroupBox("串口状态")
        serial_status_layout = QVBoxLayout(serial_status_group)
        
        self.serial_status_label = QLabel("串口状态: 未连接")
        self.serial_status_label.setAlignment(Qt.AlignCenter)
        self.serial_status_label.setStyleSheet("color: orange; font-weight: bold; padding: 6px; background-color: #fff3e0; border: 1px solid #ff9800;")
        self.serial_status_label.setMinimumHeight(30)
        serial_status_layout.addWidget(self.serial_status_label)
        
        serial_status_group.setLayout(serial_status_layout)
        layout.addWidget(serial_status_group)
        
        return panel
    
    def refresh_serial_ports(self):
        """刷新串口端口列表"""
        try:
            # 获取可用串口
            available_ports = self.serial_comm.get_available_ports()
            
            # 清空并重新填充串口列表
            self.serial_port_combo.clear()
            
            if available_ports:
                for port in available_ports:
                    self.serial_port_combo.addItem(port)
                print(f"找到 {len(available_ports)} 个可用串口: {available_ports}")
                
                # 自动选择第一个可用串口
                if self.serial_port_combo.count() > 0:
                    self.serial_port_combo.setCurrentIndex(0)
                    print(f"自动选择串口: {self.serial_port_combo.currentText()}")
            else:
                print("没有找到可用的串口")
                self.serial_port_combo.addItem("无可用串口")
                
        except Exception as e:
            print(f"刷新串口列表失败: {e}")
            QMessageBox.warning(self, "警告", f"刷新串口列表失败: {e}")
    
    def create_display_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标签页
        tab_widget = QTabWidget()
        self.tab_widget = tab_widget  # 保存为实例变量
        
        # 摄像头视图
        camera_tab = QWidget()
        camera_layout = QVBoxLayout(camera_tab)
        
        self.camera_label = QLabel("摄像头未启动")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("border: 2px solid gray; background-color: black; color: white;")
        self.camera_label.setMinimumSize(480, 360)
        camera_layout.addWidget(self.camera_label)
        
        # 摄像头视图区域（移除重复的标签，因为已经在顶部信息面板中显示）
        camera_layout.addStretch()
        
        tab_widget.addTab(camera_tab, "摄像头视图")
        
        # 用户列表
        users_tab = QWidget()
        users_layout = QVBoxLayout(users_tab)
        
        self.users_table = QTableWidget()
        self.users_table.setColumnCount(5)
        self.users_table.setHorizontalHeaderLabels(["ID", "姓名", "年龄", "性别", "创建时间"])
        
        # 启用右键菜单
        self.users_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.users_table.customContextMenuRequested.connect(self.show_users_context_menu)
        
        # 双击编辑
        self.users_table.cellDoubleClicked.connect(self.edit_user_cell)
        
        # 选择变化时启用/禁用按钮
        self.users_table.itemSelectionChanged.connect(self.on_user_selection_changed)
        
        users_layout.addWidget(self.users_table)
        
        # 用户管理按钮
        users_btn_layout = QHBoxLayout()
        
        self.edit_user_btn = QPushButton("编辑用户")
        self.edit_user_btn.clicked.connect(self.edit_selected_user)
        self.edit_user_btn.setEnabled(False)
        
        self.delete_user_btn = QPushButton("删除用户")
        self.delete_user_btn.clicked.connect(self.delete_selected_user)
        self.delete_user_btn.setEnabled(False)
        
        # 归零选中用户糖量按钮
        self.reset_selected_sugar_btn = QPushButton("归零糖量")
        self.reset_selected_sugar_btn.clicked.connect(self.reset_selected_user_sugar)
        self.reset_selected_sugar_btn.setEnabled(False)
        self.reset_selected_sugar_btn.setStyleSheet("background-color: #ff6b6b; color: white; font-weight: bold;")
        
        users_btn_layout.addWidget(self.edit_user_btn)
        users_btn_layout.addWidget(self.delete_user_btn)
        users_btn_layout.addWidget(self.reset_selected_sugar_btn)
        users_btn_layout.addStretch()
        
        users_layout.addLayout(users_btn_layout)
        
        tab_widget.addTab(users_tab, "用户列表")
        
        layout.addWidget(tab_widget)
        return panel
    
    def start_camera(self):
        if self.camera_thread is None or not self.camera_thread.isRunning():
            self.camera_thread = CameraThread(0)
            self.camera_thread.frame_ready.connect(self.on_frame_ready)
            self.camera_thread.start()
            
            self.start_camera_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(True)
            self.start_recognition_btn.setEnabled(True)
            self.status_label.setText("摄像头已启动")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
    
    def stop_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None
            
            self.start_camera_btn.setEnabled(True)
            self.stop_camera_btn.setEnabled(False)
            self.start_recognition_btn.setEnabled(False)
            self.stop_recognition_btn.setEnabled(False)
            self.status_label.setText("摄像头已停止")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            
            self.camera_label.setText("摄像头未启动")
            self.camera_label.setStyleSheet("border: 2px solid gray; background-color: black; color: white;")
    
    def on_frame_ready(self, frame):
        self.current_frame = frame
        
        if self.is_recognition_active:
            self.process_frame_for_recognition(frame)
        else:
            self.display_frame(frame)
    
    def display_frame(self, frame):
        if frame is None:
            return
        
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.camera_label.setPixmap(scaled_pixmap)
    
    def process_frame_for_recognition(self, frame):
        if frame is None:
            return
        
        current_time = time.time()
        
        # 控制识别频率，减少卡顿
        if current_time - self.last_recognition_time < self.recognition_interval:
            # 如果距离上次识别时间太短，只显示画面，不进行识别
            self.display_frame(frame)
            return
        
        # 更新识别时间
        self.last_recognition_time = current_time
        
        # 缩小图像尺寸，提高检测速度
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        # 人脸检测
        faces = self.face_detector.detect_faces(small_frame)
        
        if len(faces) > 0:
            # 将检测结果转换回原始尺寸
            faces = [(int(x*2), int(y*2), int(w*2), int(h*2)) for (x, y, w, h) in faces]
            
            # 获取最大的人脸
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # 提取人脸区域，增加边界确保完整
            margin = int(min(w, h) * 0.1)  # 10%的边界
            y1 = max(0, y - margin)
            y2 = min(frame.shape[0], y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(frame.shape[1], x + w + margin)
            
            face_roi = frame[y1:y2, x1:x2]
            
            # 进行人脸识别
            name, confidence = self.face_recognizer.recognize_face(face_roi)
            
            # 调试信息
            print(f"检测到人脸: {x}, {y}, {w}, {h}")
            print(f"识别结果: name={name}, confidence={confidence}")
            
            if name and name != "Unknown":
                # 已知人脸 - 绿色框
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                        self.sugar_added_label.setText(f"识别到: {name}")
        self.sugar_added_label.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")
                
                # 获取用户信息
                self.current_user_info = self.db_manager.get_user_by_name(name)
                if self.current_user_info:
                    # 设置当前用户到串口通信模块
                    self.serial_comm.set_current_user(self.current_user_info[0], self.current_user_info[1])
                    
                    self.user_info_label.setText(f"用户信息: {self.current_user_info[1]} (ID: {self.current_user_info[0]})")
                    self.user_info_label.setStyleSheet("color: green; font-weight: bold;")
                    
                    # 获取健康记录 - 每次都重新获取最新数据
                    try:
                        print(f"=== 开始获取用户 {self.current_user_info[1]} 的健康信息 ===")
                        
                        # 强制刷新数据库连接，获取最新数据
                        health_records = self.db_manager.get_health_records(self.current_user_info[0])
                        print(f"获取到 {len(health_records)} 条健康记录")
                        
                        if health_records:
                            latest_record = health_records[-1]
                            # 确保显示的是最新的糖量数据
                            current_sugar = latest_record[3]
                            current_limit = latest_record[4]
                            
                            print(f"最新记录: ID={latest_record[0]}, 用户ID={latest_record[1]}, 日期={latest_record[2]}, 糖量={latest_record[3]}, 限制={latest_record[4]}")
                            print(f"显示数据: 糖量={current_sugar:.2f}g, 限制={current_limit:.2f}g")
                            
                            self.health_info_label.setText(f"健康信息: 今日糖分摄入: {current_sugar:.2f}g, 今日糖分限制: {current_limit:.2f}g")
                            self.health_info_label.setStyleSheet("color: green; font-weight: bold;")
                            
                            print(f"✅ 界面已更新: 用户 {self.current_user_info[1]} 糖量 {current_sugar:.2f}g, 限制 {current_limit:.2f}g")
                        else:
                            print("❌ 没有找到健康记录")
                            self.health_info_label.setText("健康信息: 无健康记录")
                            self.health_info_label.setStyleSheet("color: orange; font-weight: bold;")
                    except Exception as e:
                        print(f"❌ 获取健康记录失败: {e}")
                        self.health_info_label.setText("健康信息: 获取失败")
                        self.health_info_label.setStyleSheet("color: red; font-weight: bold;")
                else:
                    self.user_info_label.setText("用户信息: 未知用户")
                    self.user_info_label.setStyleSheet("color: orange; font-weight: bold;")
                    self.health_info_label.setText("健康信息: 未知用户")
                    self.health_info_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                # 未知人脸 - 红色框
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.sugar_added_label.setText("未识别到已知人脸")
        self.sugar_added_label.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
                
                # 清除当前用户信息
                self.current_user_info = None
                self.serial_comm.clear_current_user()
                
                self.user_info_label.setText("用户信息: 未知用户")
                self.user_info_label.setStyleSheet("color: orange; font-weight: bold;")
                self.health_info_label.setText("健康信息: 未知用户")
                self.health_info_label.setStyleSheet("color: orange; font-weight: bold;")
            
            self.display_frame(frame)
        else:
            self.display_frame(frame)
                    self.sugar_added_label.setText("未检测到人脸")
        self.sugar_added_label.setStyleSheet("color: orange; font-size: 16px; font-weight: bold;")
            
            # 清除当前用户信息
            self.current_user_info = None
            self.serial_comm.clear_current_user()
            
            self.user_info_label.setText("用户信息: 未检测到人脸")
            self.user_info_label.setStyleSheet("color: orange; font-weight: bold;")
            self.health_info_label.setText("健康信息: 未检测到人脸")
            self.health_info_label.setStyleSheet("color: orange; font-weight: bold;")
    def start_recognition(self):
        self.is_recognition_active = True
        self.start_recognition_btn.setEnabled(False)
        self.stop_recognition_btn.setEnabled(True)
        self.status_label.setText("人脸识别已启动")
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
    
    def stop_recognition(self):
        """停止人脸识别"""
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None
            
            # 清除识别结果和用户信息
            self.clear_recognition_results()
            
            self.status_label.setText("人脸识别已停止")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            print("人脸识别已停止")
    
    def clear_recognition_results(self):
        """清除识别结果和用户信息"""
        # 清除摄像头显示
        self.camera_label.setText("摄像头未启动")
        self.camera_label.setStyleSheet("border: 2px solid gray; background-color: black; color: white;")
        
        # 清除识别结果
        self.sugar_added_label.setText("等待识别...")
        self.sugar_added_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 8px;")
        
        # 清除用户信息
        self.user_info_label.setText("用户信息: 等待识别...")
        self.user_info_label.setStyleSheet("font-size: 14px; padding: 8px; background-color: #f0f0f0; border: 1px solid #ccc;")
        
        # 清除健康信息
        self.health_info_label.setText("健康信息: 等待识别...")
        self.health_info_label.setStyleSheet("font-size: 14px; padding: 8px; background-color: #e8f5e8; border: 1px solid #4caf50;")
        
        # 清除当前用户信息
        self.current_user_info = None
        
        # 清除串口通信中的当前用户
        if hasattr(self, 'serial_comm') and self.serial_comm:
            self.serial_comm.clear_current_user()
        
        print("识别结果已清除")
    
    def on_serial_data_updated(self, user_id, user_name, actual_sugar):
        """串口数据更新后的回调函数"""
        print(f"=== 串口数据已更新，刷新用户 {user_name} 的显示，实际增加糖量: {actual_sugar:.1f}g ===")
        
        # 更新实际增加糖量标签
        self.sugar_added_label.setText(f"实际增加糖量: {actual_sugar:.1f}g")
        self.sugar_added_label.setStyleSheet("color: blue; font-size: 16px; font-weight: bold;")
        
        # 如果当前显示的是这个用户，刷新健康信息
        if self.current_user_info and self.current_user_info[0] == user_id:
            try:
                print(f"当前显示用户匹配，开始刷新界面")
                
                # 强制刷新数据库连接，获取最新数据
                health_records = self.db_manager.get_health_records(user_id)
                print(f"获取到 {len(health_records)} 条健康记录")
                
                if health_records:
                    latest_record = health_records[-1]
                    current_sugar = latest_record[3]
                    current_limit = latest_record[4]
                    
                    print(f"最新记录: ID={latest_record[0]}, 用户ID={latest_record[1]}, 日期={latest_record[2]}, 糖量={latest_record[3]}, 限制={latest_record[4]}")
                    print(f"显示数据: 糖量={current_sugar:.2f}g, 限制={current_limit:.2f}g")
                    
                    # 更新界面显示
                    self.health_info_label.setText(f"健康信息: 今日糖分摄入: {current_sugar:.2f}g, 今日糖分限制: {current_limit:.2f}g")
                    self.health_info_label.setStyleSheet("color: green; font-weight: bold;")
                    
                    # 同时更新用户信息标签
                    self.user_info_label.setText(f"用户信息: {user_name} (ID: {user_id})")
                    self.user_info_label.setStyleSheet("color: green; font-weight: bold;")
                    
                    print(f"✅ 界面已更新: 用户 {user_name} 糖量 {current_sugar:.2f}g, 限制 {current_limit:.2f}g")
                    
                else:
                    print(f"❌ 用户 {user_name} 没有健康记录")
                    
            except Exception as e:
                print(f"❌ 刷新界面显示失败: {e}")
        else:
            print(f"当前显示的不是用户 {user_name}，不更新界面")
            if self.current_user_info:
                print(f"当前显示用户: ID={self.current_user_info[0]}, 姓名={self.current_user_info[1]}")
            else:
                print("当前没有显示用户")
    
    def start_face_training(self):
        """开始人脸训练"""
        try:
            # 检查是否有用户
            users = self.db_manager.get_all_users()
            if not users:
                QMessageBox.warning(self, "警告", "请先添加用户！")
                return
            
            # 选择用户进行训练
            user_names = [user[1] for user in users]
            user_name, ok = QInputDialog.getItem(self, "选择用户", "请选择要训练的用户:", user_names, 0, False)
            if not ok:
                return
            
            # 获取用户ID
            user_id = None
            for user in users:
                if user[1] == user_name:
                    user_id = user[0]
                    break
            
            if user_id is None:
                return
            
            # 创建训练对话框
            training_dialog = TrainingDialog(self.face_detector, self.face_recognizer, self.db_manager, user_id, user_name, self)
            if training_dialog.exec_() == QDialog.Accepted:
                # 训练成功后，重新加载识别器
                self.face_recognizer.load_model()
                QMessageBox.information(self, "成功", f"用户 {user_name} 训练完成！现在可以进行人脸识别了。")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动训练失败: {e}")
    
    def add_user(self):
        name, ok = QInputDialog.getText(self, "添加用户", "请输入姓名:")
        if not ok or not name:
            return
        
        age, ok = QInputDialog.getInt(self, "添加用户", "请输入年龄:", 25, 1, 120)
        if not ok:
            return
        
        gender, ok = QInputDialog.getItem(self, "添加用户", "请选择性别:", ["男", "女", "未知"], 0, False)
        if not ok:
            return
        
        user_id = self.db_manager.add_user(name, age, gender)
        
        if user_id:
            QMessageBox.information(self, "成功", f"用户 {name} 添加成功！")
            self.refresh_users_table()
        else:
            QMessageBox.warning(self, "错误", "添加用户失败！")
    
    def view_users(self):
        """查看用户列表"""
        try:
            # 检查tab_widget是否存在
            if not hasattr(self, 'tab_widget') or not self.tab_widget:
                print("tab_widget未初始化")
                QMessageBox.warning(self, "错误", "界面未正确初始化")
                return
            
            # 获取用户列表标签页的索引
            users_tab_index = -1
            for i in range(self.tab_widget.count()):
                if self.tab_widget.tabText(i) == "用户列表":
                    users_tab_index = i
                    break
            
            if users_tab_index == -1:
                print("未找到用户列表标签页")
                QMessageBox.warning(self, "错误", "未找到用户列表标签页")
                return
            
            # 切换到用户列表标签页
            self.tab_widget.setCurrentIndex(users_tab_index)
            print(f"切换到用户列表标签页 (索引: {users_tab_index})")
            
            # 刷新用户表格
            self.refresh_users_table()
            
            # 显示用户数量
            users = self.db_manager.get_all_users()
            QMessageBox.information(self, "用户信息", f"当前共有 {len(users)} 个用户")
            
        except Exception as e:
            print(f"查看用户失败: {e}")
            QMessageBox.warning(self, "错误", f"查看用户失败: {e}")
    
    def on_user_selection_changed(self):
        """用户选择变化时的处理"""
        selected_rows = self.users_table.selectionModel().selectedRows()
        has_selection = len(selected_rows) > 0
        
        # 启用/禁用按钮
        self.edit_user_btn.setEnabled(has_selection)
        self.delete_user_btn.setEnabled(has_selection)
        self.reset_selected_sugar_btn.setEnabled(has_selection)
    
    def show_users_context_menu(self, position):
        """显示用户表格右键菜单"""
        menu = QMenu()
        
        edit_action = menu.addAction("编辑用户")
        delete_action = menu.addAction("删除用户")
        reset_sugar_action = menu.addAction("重置糖量")
        
        action = menu.exec_(self.users_table.mapToGlobal(position))
        
        if action == edit_action:
            self.edit_selected_user()
        elif action == delete_action:
            self.delete_selected_user()
        elif action == reset_sugar_action:
            self.reset_selected_user_sugar()
    
    def edit_selected_user(self):
        """编辑选中的用户"""
        selected_rows = self.users_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "警告", "请先选择要编辑的用户！")
            return
        
        row = selected_rows[0].row()
        user_id = int(self.users_table.item(row, 0).text())
        user_name = self.users_table.item(row, 1).text()
        user_age = int(self.users_table.item(row, 2).text())
        user_gender = self.users_table.item(row, 3).text()
        
        # 获取创建时间
        create_time = ""
        if self.users_table.item(row, 4):
            create_time = self.users_table.item(row, 4).text()
        
        user_data = (user_id, user_name, user_age, user_gender, create_time)
        
        # 打开编辑对话框
        dialog = UserEditDialog(user_data, self)
        if dialog.exec_() == QDialog.Accepted:
            # 刷新用户数据
            self.load_users_data()
    
    def edit_user_cell(self, row, column):
        """双击编辑用户表格单元格"""
        if column == 0:  # ID列不允许编辑
            return
        
        item = self.users_table.item(row, column)
        if not item:
            return
        
        current_value = item.text()
        
        if column == 1:  # 姓名列
            new_value, ok = QInputDialog.getText(
                self, "编辑姓名", 
                f"请输入新的姓名 (当前: {current_value}):",
                text=current_value
            )
        elif column == 2:  # 年龄列
            new_value, ok = QInputDialog.getInt(
                self, "编辑年龄",
                f"请输入新的年龄 (当前: {current_value}):",
                int(current_value), 1, 120, 1
            )
            if ok:
                new_value = str(new_value)
        elif column == 3:  # 性别列
            new_value, ok = QInputDialog.getItem(
                self, "编辑性别",
                f"请选择性别 (当前: {current_value}):",
                ["男", "女"], 0 if current_value == "男" else 1, False
            )
        else:
            return
        
        if ok and new_value != current_value:
            try:
                user_id = int(self.users_table.item(row, 0).text())
                
                if column == 1:  # 姓名
                    if self.db_manager.modify_user_info(user_id, new_value, None, None):
                        item.setText(new_value)
                        print(f"用户 {user_id} 姓名已更新为: {new_value}")
                    else:
                        QMessageBox.warning(self, "错误", "修改姓名失败！")
                elif column == 2:  # 年龄
                    if self.db_manager.modify_user_info(user_id, None, int(new_value), None):
                        item.setText(new_value)
                        print(f"用户 {user_id} 年龄已更新为: {new_value}")
                    else:
                        QMessageBox.warning(self, "错误", "修改年龄失败！")
                elif column == 3:  # 性别
                    if self.db_manager.modify_user_info(user_id, None, None, new_value):
                        item.setText(new_value)
                        print(f"用户 {user_id} 性别已更新为: {new_value}")
                    else:
                        QMessageBox.warning(self, "错误", "修改性别失败！")
                        
            except Exception as e:
                QMessageBox.critical(self, "错误", f"修改失败: {e}")
    
    def delete_selected_user(self):
        """删除选中的用户"""
        selected_rows = self.users_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "警告", "请先选择要删除的用户！")
            return
        
        row = selected_rows[0].row()
        user_id = int(self.users_table.item(row, 0).text())
        user_name = self.users_table.item(row, 1).text()
        
        reply = QMessageBox.question(
            self, "确认删除", 
            f"确定要删除用户 '{user_name}' (ID: {user_id}) 吗？\n此操作不可恢复！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                if self.db_manager.delete_user(user_id):
                    QMessageBox.information(self, "成功", f"用户 '{user_name}' 已删除！")
                    self.load_users_data()  # 刷新表格
                else:
                    QMessageBox.warning(self, "错误", "删除用户失败！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除失败: {e}")
    
    def refresh_users_table(self):
        """刷新用户表格"""
        try:
            print("开始刷新用户表格...")
            
            # 获取所有用户
            users = self.db_manager.get_all_users()
            print(f"获取到 {len(users)} 个用户")
            
            if not users:
                print("没有找到用户数据")
                self.users_table.setRowCount(0)
                return
            
            # 设置表格行数
            self.users_table.setRowCount(len(users))
            
            # 填充表格数据
            for row, user in enumerate(users):
                print(f"填充用户 {row}: {user}")
                for col in range(min(len(user), 5)):
                    item = QTableWidgetItem(str(user[col]) if user[col] is not None else "")
                    self.users_table.setItem(row, col, item)
            
            print("用户表格刷新完成")
            
        except Exception as e:
            print(f"刷新用户表格失败: {e}")
            QMessageBox.warning(self, "错误", f"刷新用户表格失败: {e}")
    
    def add_health_record(self):
        users = self.db_manager.get_all_users()
        if not users:
            QMessageBox.warning(self, "警告", "请先添加用户！")
            return
        
        user_names = [user[1] for user in users]
        user_name, ok = QInputDialog.getItem(self, "添加健康记录", "请选择用户:", user_names, 0, False)
        if not ok:
            return
        
        user_id = None
        for user in users:
            if user[1] == user_name:
                user_id = user[0]
                break
        
        if user_id is None:
            return
        
        sugar_intake, ok = QInputDialog.getDouble(self, "添加健康记录", "请输入今日糖分摄入量(g):", 0.0, 0.0, 1000.0, 1)
        if not ok:
            return
        
        sugar_limit, ok = QInputDialog.getDouble(self, "添加健康记录", "请输入糖分摄入上限(g):", 50.0, 10.0, 1000.0, 1)
        if not ok:
            return
        
        today = datetime.now().strftime("%Y-%m-%d")
        self.db_manager.add_health_record(user_id, today, sugar_intake, sugar_limit)
        
        QMessageBox.information(self, "成功", f"健康记录添加成功！")
    
    def reset_today_sugar(self):
        """一键归零今日糖量"""
        if self.current_user_info:
            user_id = self.current_user_info[0]
            try:
                print(f"=== 开始归零用户 {self.current_user_info[1]} 的今日糖量 ===")
                
                # 获取今日健康记录的ID
                health_record_id = self.db_manager.get_user_health_today_id(user_id)
                if health_record_id:
                    print(f"找到今日健康记录ID: {health_record_id}")
                    
                    # 更新糖分摄入量为0
                    if self.db_manager.update_health_record_sugar(health_record_id, 0.0):
                        QMessageBox.information(self, "成功", f"用户 {self.current_user_info[1]} 今日糖分摄入已归零！")
                        
                        # 刷新健康信息显示
                        self.refresh_health_info()
                        
                        # 发送更新后的用户信息到串口
                        self.serial_comm.send_user_info()
                        
                        print(f"✅ 用户 {self.current_user_info[1]} 今日糖量已归零")
                    else:
                        QMessageBox.warning(self, "错误", "归零糖量失败！")
                else:
                    print("未找到今日健康记录，尝试创建新记录")
                    # 如果没有今日记录，创建一个
                    today = datetime.now().strftime("%Y-%m-%d")
                    self.db_manager.add_health_record(user_id, today, 0.0, 50.0)
                    QMessageBox.information(self, "成功", f"用户 {self.current_user_info[1]} 今日糖分摄入已归零！")
                    
                    # 刷新健康信息显示
                    self.refresh_health_info()
                    
                    # 发送更新后的用户信息到串口
                    self.serial_comm.send_user_info()
                    
                    print(f"✅ 用户 {self.current_user_info[1]} 今日糖量已归零（新建记录）")
            except Exception as e:
                print(f"❌ 归零糖量失败: {e}")
                QMessageBox.critical(self, "错误", f"归零糖量失败: {e}")
        else:
            QMessageBox.warning(self, "警告", "请先识别用户！")
    
    def reset_selected_user_sugar(self):
        """重置选中用户的今日糖量"""
        selected_rows = self.users_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "警告", "请先选择一个用户！")
            return
        
        row = selected_rows[0].row()
        user_id = int(self.users_table.item(row, 0).text())
        user_name = self.users_table.item(row, 1).text()
        
        reply = QMessageBox.question(
            self, "确认重置", 
            f"确定要重置用户 '{user_name}' 的今日糖量吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # 获取今日健康记录的ID
                health_record_id = self.db_manager.get_user_health_today_id(user_id)
                if health_record_id:
                    # 更新糖分摄入量为0
                    self.db_manager.update_health_record_sugar(health_record_id, 0.0)
                    QMessageBox.information(self, "成功", f"用户 '{user_name}' 今日糖分摄入已归零！")
                    # 刷新健康信息显示
                    self.refresh_health_info()
                    # 发送更新后的用户信息到串口
                    self.serial_comm.send_user_info()
                else:
                    QMessageBox.warning(self, "警告", "未找到今日健康记录，无法归零糖量。")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"归零糖量失败: {e}")
    
    def start_serial_communication(self):
        """启动串口通信"""
        try:
            if self.serial_comm.start():
                self.status_label.setText("串口通信已启动")
                self.status_label.setStyleSheet("color: green; font-weight: bold;")
                self.serial_status_label.setText("串口状态: 已连接")
                self.serial_status_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.status_label.setText("串口通信启动失败")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                self.serial_status_label.setText("串口状态: 未连接")
                self.serial_status_label.setStyleSheet("color: red; font-weight: bold;")
        except Exception as e:
            print(f"启动串口通信失败: {e}")
            self.status_label.setText("串口通信启动失败")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.serial_status_label.setText("串口状态: 未连接")
            self.serial_status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def refresh_health_info(self):
        """刷新健康信息显示"""
        if self.current_user_info:
            try:
                print(f"=== 手动刷新用户 {self.current_user_info[1]} 的健康信息 ===")
                
                # 强制刷新数据库连接，获取最新数据
                health_records = self.db_manager.get_health_records(self.current_user_info[0])
                print(f"获取到 {len(health_records)} 条健康记录")
                
                if health_records:
                    latest_record = health_records[-1]
                    current_sugar = latest_record[3]
                    current_limit = latest_record[4]
                    
                    print(f"最新记录: ID={latest_record[0]}, 用户ID={latest_record[1]}, 日期={latest_record[2]}, 糖量={latest_record[3]}, 限制={latest_record[4]}")
                    print(f"显示数据: 糖量={current_sugar:.2f}g, 限制={current_limit:.2f}g")
                    
                    self.health_info_label.setText(f"健康信息: 今日糖分摄入: {current_sugar:.2f}g, 今日糖分限制: {current_limit:.2f}g")
                    self.health_info_label.setStyleSheet("color: green; font-weight: bold;")
                    
                    print(f"✅ 手动刷新完成: 用户 {self.current_user_info[1]} 糖量 {current_sugar:.2f}g, 限制 {current_limit:.2f}g")
                    
                    # 发送更新后的用户信息到串口
                    self.serial_comm.send_user_info()
                else:
                    print("❌ 没有找到健康记录")
                    self.health_info_label.setText("健康信息: 无健康记录")
                    self.health_info_label.setStyleSheet("color: orange; font-weight: bold;")
            except Exception as e:
                print(f"❌ 刷新健康信息失败: {e}")
                self.health_info_label.setText("健康信息: 刷新失败")
                self.health_info_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            print("❌ 当前没有识别用户，无法刷新健康信息")
            self.health_info_label.setText("健康信息: 请先识别用户")
            self.health_info_label.setStyleSheet("color: orange; font-weight: bold;")
    
    def start_daily_refresh_timer(self):
        """启动每日12点自动刷新定时器"""
        now = datetime.now()
        # 设置为每天的12:00:00
        next_refresh_time = datetime(now.year, now.month, now.day, 12, 0, 0)
        if now > next_refresh_time:
            next_refresh_time += timedelta(days=1)
        
        # 计算到下次刷新的秒数
        seconds_until_refresh = int((next_refresh_time - now).total_seconds())
        
        # 设置定时器
        self.daily_refresh_timer.setInterval(seconds_until_refresh * 1000)  # 转换为毫秒
        self.daily_refresh_timer.start()
        
        print(f"每日糖量数据刷新定时器已设置，下次刷新在: {next_refresh_time} (还有 {seconds_until_refresh} 秒)")
    
    def check_daily_refresh(self):
        """每日12点自动刷新糖量数据"""
        print("执行每日糖量数据刷新...")
        try:
            users = self.db_manager.get_all_users()
            for user in users:
                user_id = user[0]
                # 获取今日健康记录
                health_record = self.db_manager.get_user_health_today(user_id)
                if health_record:
                    # 重置今日糖分摄入量为0
                    cursor = self.db_manager.conn.cursor()
                    cursor.execute(
                        "UPDATE health_records SET sugar_intake = 0.0 WHERE id = ?",
                        (health_record[0],)
                    )
                    self.db_manager.conn.commit()
                    print(f"用户 {user[1]} (ID: {user_id}) 的糖分摄入量已重置为0")
            
            # 重新设置下次刷新时间（24小时后）
            self.daily_refresh_timer.setInterval(24 * 60 * 60 * 1000)  # 24小时
            print("每日糖量数据刷新完成，下次刷新在24小时后")
            
        except Exception as e:
            print(f"每日糖量数据刷新失败: {e}")
    
    def closeEvent(self, event):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
        
        # 停止串口通信
        if self.serial_comm:
            self.serial_comm.stop()
        
        event.accept()