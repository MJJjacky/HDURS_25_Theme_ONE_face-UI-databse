#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的人脸训练脚本
整合Qt界面和命令行训练，确保训练数据能正确保存
"""

import cv2
import os
import sys
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from face_recognition.face_detector import FaceDetector
from face_recognition.face_recognizer import FaceRecognizer
from database.database_manager import DatabaseManager

class UnifiedFaceTrainer:
    """统一的人脸训练器"""
    
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.db_manager = DatabaseManager()
        self.face_images_dir = "data/faces"
        
        # 确保目录存在
        os.makedirs(self.face_images_dir, exist_ok=True)
    
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
    
    def collect_from_directory(self, dir_path, person_name):
        """从目录收集训练样本（基于用户代码）"""
        faces = []
        saved_images = []
        print(f"从目录收集 {person_name} 的训练样本: {dir_path}")
        
        if not os.path.exists(dir_path):
            print(f"目录不存在: {dir_path}")
            return faces, saved_images
        
        # 创建用户目录
        user_dir = os.path.join(self.face_images_dir, person_name)
        os.makedirs(user_dir, exist_ok=True)
        
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
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{person_name}_{timestamp}_{i:03d}.jpg"
                    save_path = os.path.join(user_dir, filename)
                    cv2.imwrite(save_path, face)
                    saved_images.append(save_path)
                    
                    # 预处理人脸（调整大小）
                    processed_face = cv2.resize(face, (150, 150))
                    faces.append(processed_face)
                    print(f"成功处理: {file} -> {filename}")
        
        print(f"从目录收集到 {len(faces)} 个样本，保存了 {len(saved_images)} 张图片")
        return faces, saved_images
    
    def save_to_database(self, person_name, image_paths, age=25, gender="未知"):
        """保存用户信息和人脸图片到数据库"""
        try:
            # 检查用户是否存在
            users = self.db_manager.get_all_users()
            user_id = None
            
            for user in users:
                if user[1] == person_name:
                    user_id = user[0]
                    break
            
            if user_id is None:
                # 创建新用户
                user_id = self.db_manager.add_user(person_name, age, gender)
                print(f"创建新用户: {person_name}, ID: {user_id}")
            else:
                print(f"用户已存在: {person_name}, ID: {user_id}")
            
            # 保存人脸图片路径到数据库
            for image_path in image_paths:
                self.db_manager.add_face_image(user_id, image_path, person_name)
            
            print(f"训练数据已保存到数据库，用户ID: {user_id}")
            return user_id
            
        except Exception as e:
            print(f"保存到数据库失败: {e}")
            return None
    
    def train_from_directories(self, training_data):
        """从多个目录训练模型"""
        """
        training_data = [
            {"dir": "path/to/yangmi", "name": "杨幂", "age": 30, "gender": "女"},
            {"dir": "path/to/liuyifei", "name": "刘亦菲", "age": 35, "gender": "女"}
        ]
        """
        all_faces = []
        all_labels = []
        all_names = []
        
        print("开始收集训练数据...")
        
        for data in training_data:
            dir_path = data["dir"]
            person_name = data["name"]
            age = data.get("age", 25)
            gender = data.get("gender", "未知")
            
            # 收集人脸样本
            faces, saved_images = self.collect_from_directory(dir_path, person_name)
            
            if len(faces) > 0:
                # 保存到数据库
                user_id = self.save_to_database(person_name, saved_images, age, gender)
                
                # 添加到训练数据
                all_faces.extend(faces)
                all_labels.extend([person_name] * len(faces))
                all_names.append(person_name)
                
                print(f"{person_name}: {len(faces)} 个样本")
            else:
                print(f"警告: {person_name} 没有收集到有效样本")
        
        if len(all_faces) == 0:
            print("没有收集到任何训练样本！")
            return False
        
        print(f"\n总共收集到 {len(all_faces)} 个样本，{len(all_names)} 个用户")
        
        # 创建标签映射
        unique_names = list(set(all_labels))
        name_to_id = {name: i for i, name in enumerate(unique_names)}
        
        # 转换标签为数字ID
        numeric_labels = [name_to_id[name] for name in all_labels]
        
        # 转换为numpy数组
        images = np.array(all_faces)
        labels = np.array(numeric_labels)
        
        print(f"开始训练模型...")
        print(f"图像数量: {len(images)}, 标签数量: {len(labels)}")
        print(f"标签映射: {name_to_id}")
        
        # 训练模型
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(images, labels)
            
            # 保存模型
            model_path = "data/models/face_recognizer.yml"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            recognizer.write(model_path)
            
            # 保存标签映射
            import pickle
            label_map_path = model_path.replace('.yml', '_labels.pkl')
            label_data = {
                'name_to_id': name_to_id,
                'id_to_name': {i: name for name, i in name_to_id.items()}
            }
            with open(label_map_path, 'wb') as f:
                pickle.dump(label_data, f)
            
            print(f"✅ 训练完成！")
            print(f"模型已保存: {model_path}")
            print(f"标签映射已保存: {label_map_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            return False
    
    def train_from_camera(self, person_name, num_samples=20):
        """从摄像头训练模型"""
        print(f"开始从摄像头收集 {person_name} 的训练样本...")
        
        # 启动摄像头
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("无法打开摄像头")
            return False
        
        samples = []
        saved_images = []
        
        try:
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
                    
                    # 保存人脸图片
                    saved_path = self.save_face_image(face_roi, person_name, i)
                    saved_images.append(saved_path)
                    
                    # 预处理人脸
                    processed_face = cv2.resize(face_roi, (150, 150))
                    samples.append(processed_face)
                    
                    print(f"样本 {i+1}/{num_samples} 已采集")
                
                # 等待一下
                cv2.waitKey(100)
            
            camera.release()
            
            if len(samples) > 0:
                # 保存到数据库
                user_id = self.save_to_database(person_name, saved_images)
                
                # 训练模型
                return self.train_single_person(person_name, samples)
            else:
                print("没有采集到有效样本")
                return False
                
        except Exception as e:
            print(f"摄像头训练失败: {e}")
            camera.release()
            return False
    
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
    
    def train_single_person(self, person_name, samples):
        """训练单个人员的模型"""
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

def main():
    print("=== 统一的人脸识别模型训练程序 ===")
    print("支持从目录和摄像头训练，确保数据持久化")
    
    trainer = UnifiedFaceTrainer()
    
    print("\n请选择训练方式:")
    print("1. 从目录训练（推荐，基于用户代码逻辑）")
    print("2. 从摄像头训练")
    print("3. 查看现有用户")
    
    choice = input("请选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 从目录训练
        training_data = []
        
        while True:
            dir_path = input("请输入用户图片目录路径: ").strip()
            if not dir_path:
                break
                
            person_name = input("请输入用户姓名: ").strip()
            if not person_name:
                break
                
            age = int(input("请输入用户年龄 (默认25): ").strip() or "25")
            gender = input("请输入用户性别 (男/女/未知): ").strip() or "未知"
            
            training_data.append({
                "dir": dir_path,
                "name": person_name,
                "age": age,
                "gender": gender
            })
            
            more = input("是否添加更多用户? (y/n): ").strip().lower()
            if more != 'y':
                break
        
        if training_data:
            success = trainer.train_from_directories(training_data)
            if success:
                print("\n🎉 训练成功！现在可以运行主程序进行人脸识别了")
            else:
                print("\n❌ 训练失败，请检查错误信息")
        else:
            print("没有输入训练数据")
    
    elif choice == "2":
        # 从摄像头训练
        person_name = input("请输入用户姓名: ").strip()
        if person_name:
            num_samples = int(input("请输入样本数量 (默认20): ").strip() or "20")
            success = trainer.train_from_camera(person_name, num_samples)
            if success:
                print("\n🎉 训练成功！现在可以运行主程序进行人脸识别了")
            else:
                print("\n❌ 训练失败，请检查错误信息")
        else:
            print("姓名不能为空")
    
    elif choice == "3":
        # 查看现有用户
        users = trainer.db_manager.get_all_users()
        if users:
            print("\n现有用户:")
            for user in users:
                print(f"ID: {user[0]}, 姓名: {user[1]}, 年龄: {user[2]}, 性别: {user[3]}")
                
                # 获取用户的人脸图片
                face_images = trainer.db_manager.get_user_face_images(user[0])
                print(f"  人脸图片数量: {len(face_images)}")
        else:
            print("没有现有用户")
    
    else:
        print("无效选择")

if __name__ == "__main__":
    main()