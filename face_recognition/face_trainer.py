import cv2
import numpy as np
import os
from typing import List, Tuple
import random
import shutil
from datetime import datetime

class FaceTrainer:
    """统一的人脸训练器，整合Qt界面和命令行训练"""
    
    def __init__(self, face_detector, face_recognizer, db_manager=None):
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.db_manager = db_manager
        self.training_data = {}  # {person_name: [face_images]}
        self.face_images_dir = "data/faces"  # 人脸图片保存目录
        
        # 确保目录存在
        os.makedirs(self.face_images_dir, exist_ok=True)
    
    def preprocess_face(self, face_image):
        """预处理人脸图像"""
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # 直方图均衡化
        gray = cv2.equalizeHist(gray)
        
        # 调整大小
        gray = cv2.resize(gray, (150, 150))
        
        return gray
    
    def save_face_image(self, face_image, person_name, index):
        """保存人脸图片到文件系统"""
        # 创建用户目录
        user_dir = os.path.join(self.face_images_dir, person_name)
        os.makedirs(user_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{person_name}_{timestamp}_{index:03d}.jpg"
        filepath = os.path.join(user_dir, filename)
        
        # 保存图片
        cv2.imwrite(filepath, face_image)
        print(f"保存人脸图片: {filepath}")
        
        return filepath
    
    def augment_face(self, face_image):
        """数据增强：生成多个变体"""
        augmented = []
        
        # 原始图像
        augmented.append(self.preprocess_face(face_image))
        
        # 轻微旋转
        for angle in [-5, 5]:
            height, width = face_image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(face_image, rotation_matrix, (width, height))
            augmented.append(self.preprocess_face(rotated))
        
        # 亮度调整
        for alpha in [0.8, 1.2]:
            adjusted = cv2.convertScaleAbs(face_image, alpha=alpha, beta=0)
            augmented.append(self.preprocess_face(adjusted))
        
        # 对比度调整
        for beta in [-30, 30]:
            adjusted = cv2.convertScaleAbs(face_image, alpha=1.0, beta=beta)
            augmented.append(self.preprocess_face(adjusted))
        
        return augmented
    
    def collect_training_samples(self, camera, person_name, num_samples=20):
        """收集训练样本，包含数据增强和图片保存"""
        samples = []
        saved_images = []  # 保存的图片路径
        print(f"开始收集 {person_name} 的训练样本...")
        
        for i in range(num_samples):
            ret, frame = camera.read()
            if not ret:
                continue
            
            # 检测人脸
            faces = self.face_detector.detect_faces(frame)
            
            if len(faces) > 0:
                # 获取最大人脸
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # 提取人脸区域
                face_roi = frame[y:y+h, x:x+w]
                
                # 保存原始人脸图片
                saved_path = self.save_face_image(face_roi, person_name, i)
                saved_images.append(saved_path)
                
                # 数据增强
                augmented_faces = self.augment_face(face_roi)
                samples.extend(augmented_faces)
                
                print(f"样本 {i+1}/{num_samples} - 生成 {len(augmented_faces)} 个增强样本")
            
            # 等待一下
            cv2.waitKey(100)
        
        print(f"总共收集到 {len(samples)} 个训练样本，保存了 {len(saved_images)} 张图片")
        
        # 保存到数据库（如果有数据库管理器）
        if self.db_manager:
            self.save_training_data_to_db(person_name, saved_images, len(samples))
        
        return samples, saved_images
    
    def collect_from_directory(self, dir_path, person_name):
        """从目录收集训练样本（基于用户代码）"""
        faces = []
        saved_images = []
        print(f"从目录收集 {person_name} 的训练样本: {dir_path}")
        
        if not os.path.exists(dir_path):
            print(f"目录不存在: {dir_path}")
            return faces, saved_images
        
        for i, file in enumerate(os.listdir(dir_path)):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                # 读取图像
                img = cv2.imread(file_path)
                if img is None:
                    continue
                
                # 检测人脸
                face, rect = self.face_detect_demo(img)
                if face is not None:
                    # 保存检测到的人脸图片
                    saved_path = self.save_face_image(face, person_name, i)
                    saved_images.append(saved_path)
                    
                    # 预处理人脸
                    processed_face = self.preprocess_face(face)
                    faces.append(processed_face)
                    print(f"成功处理: {file}")
        
        print(f"从目录收集到 {len(faces)} 个样本，保存了 {len(saved_images)} 张图片")
        
        # 保存到数据库
        if self.db_manager:
            self.save_training_data_to_db(person_name, saved_images, len(faces))
        
        return faces, saved_images
    
    def save_training_data_to_db(self, person_name, image_paths, sample_count):
        """保存训练数据到数据库"""
        try:
            if not self.db_manager:
                print("数据库管理器未初始化，跳过数据库保存")
                return
            
            # 检查用户是否存在，不存在则创建
            users = self.db_manager.get_all_users()
            user_id = None
            
            for user in users:
                if user[1] == person_name:
                    user_id = user[0]
                    break
            
            if user_id is None:
                # 创建新用户
                user_id = self.db_manager.add_user(person_name, 25, "未知")
                print(f"创建新用户: {person_name}, ID: {user_id}")
            
            # 保存人脸图片路径到数据库
            for image_path in image_paths:
                self.db_manager.add_face_image(user_id, image_path, person_name)
            
            print(f"训练数据已保存到数据库，用户ID: {user_id}")
            
        except Exception as e:
            print(f"保存到数据库失败: {e}")
    
    def face_detect_demo(self, image):
        """人脸检测函数（基于用户代码）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier("data/models/haarcascade_frontalface_default.xml")
        faces = face_detector.detectMultiScale(gray, 1.2, 6)
        
        # 如果未检测到面部，则返回None
        if len(faces) == 0:
            return None, None
        
        # 获取最大的人脸
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        (x, y, w, h) = largest_face
        
        # 返回图像的脸部部分
        return gray[y:y+h, x:x+w], largest_face
    
    def add_training_data(self, person_name, face_images):
        """添加训练数据"""
        if person_name not in self.training_data:
            self.training_data[person_name] = []
        
        self.training_data[person_name].extend(face_images)
        print(f"为 {person_name} 添加了 {len(face_images)} 个训练样本")
    
    def train_all(self):
        """训练所有收集的数据"""
        if not self.training_data:
            print("没有训练数据")
            return False
        
        print("开始训练所有数据...")
        
        # 清空之前的训练数据
        self.face_recognizer.clear_training_data()
        
        # 添加所有训练样本
        for person_name, faces in self.training_data.items():
            print(f"添加 {person_name} 的 {len(faces)} 个样本")
            for face in faces:
                self.face_recognizer.add_training_sample(face, person_name)
        
        # 开始训练
        success = self.face_recognizer.train()
        
        if success:
            print("所有数据训练完成！")
            # 清空训练数据
            self.training_data.clear()
        else:
            print("训练失败！")
        
        return success
    
    def train_person(self, person_name, samples):
        """训练特定人员的人脸识别模型"""
        print(f"开始训练 {person_name} 的识别模型...")
        
        # 清空之前的训练数据
        self.face_recognizer.clear_training_data()
        
        # 添加所有样本
        for sample in samples:
            self.face_recognizer.add_training_sample(sample, person_name)
        
        # 训练模型
        success = self.face_recognizer.train()
        
        if success:
            print(f"{person_name} 训练完成！")
        else:
            print(f"{person_name} 训练失败！")
        
        return success
    
    def get_training_status(self):
        """获取训练状态"""
        status = {}
        for person_name, faces in self.training_data.items():
            status[person_name] = len(faces)
        return status
    
    def get_saved_faces(self, person_name):
        """获取已保存的人脸图片路径"""
        user_dir = os.path.join(self.face_images_dir, person_name)
        if not os.path.exists(user_dir):
            return []
        
        image_files = []
        for file in os.listdir(user_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(user_dir, file))
        
        return image_files
