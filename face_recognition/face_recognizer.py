import os
import cv2
import numpy as np
import pickle
from typing import List, Tuple, Optional

class FaceRecognizer:
    """使用OpenCV LBPH的人脸识别器，基于用户代码优化"""
    
    def __init__(self, model_path=None, tolerance=100):
        self.tolerance = tolerance  # 置信度阈值，LBPH中置信度越低越好，所以设置较高阈值
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.known_face_names = []
        self.model_path = model_path or "data/models/face_recognizer.yml"
        self.name_to_id = {}  # 姓名到ID的映射
        self.id_to_name = {}  # ID到姓名的映射
        
        # 尝试加载已有模型
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            if os.path.exists(self.model_path):
                self.recognizer.read(self.model_path)
                print(f"成功加载模型: {self.model_path}")
                
                # 尝试加载标签映射文件
                label_map_path = self.model_path.replace('.yml', '_labels.pkl')
                if os.path.exists(label_map_path):
                    with open(label_map_path, 'rb') as f:
                        label_data = pickle.load(f)
                        self.name_to_id = label_data.get('name_to_id', {})
                        self.id_to_name = label_data.get('id_to_name', {})
                        self.known_face_names = list(self.name_to_id.keys())
                        print(f"加载标签映射: {self.known_face_names}")
                
                return True
            else:
                print(f"模型文件不存在: {self.model_path}")
                return False
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
    
    def save_model(self):
        """保存训练好的模型和标签映射"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # 保存模型
            self.recognizer.write(self.model_path)
            print(f"模型已保存: {self.model_path}")
            
            # 保存标签映射
            label_map_path = self.model_path.replace('.yml', '_labels.pkl')
            label_data = {
                'name_to_id': self.name_to_id,
                'id_to_name': self.id_to_name
            }
            with open(label_map_path, 'wb') as f:
                pickle.dump(label_data, f)
            print(f"标签映射已保存: {label_map_path}")
            
            return True
        except Exception as e:
            print(f"保存模型失败: {e}")
            return False
    
    def add_training_sample(self, face_image, person_name):
        """添加训练样本"""
        try:
            # 转换为灰度图像
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # 图像预处理：直方图均衡化提高对比度
            gray = cv2.equalizeHist(gray)
            
            # 调整图像大小为标准尺寸
            gray = cv2.resize(gray, (150, 150))
            
            # 添加到训练数据
            if not hasattr(self, 'training_images'):
                self.training_images = []
                self.training_labels = []
            
            self.training_images.append(gray)
            self.training_labels.append(person_name)
            
            print(f"成功添加 {person_name} 的训练样本")
            return True
        except Exception as e:
            print(f"添加训练样本失败: {e}")
            return False
    
    def train(self):
        """训练模型"""
        try:
            if not hasattr(self, 'training_images') or len(self.training_images) < 2:
                print("训练样本不足")
                return False
            
            # 创建标签映射
            unique_names = list(set(self.training_labels))
            self.name_to_id = {name: i for i, name in enumerate(unique_names)}
            self.id_to_name = {i: name for name, i in self.name_to_id.items()}
            
            # 转换标签为数字ID
            numeric_labels = [self.name_to_id[name] for name in self.training_labels]
            
            # 转换为numpy数组
            images = np.array(self.training_images)
            labels = np.array(numeric_labels)
            
            print(f"开始训练，图像数量: {len(images)}, 标签数量: {len(labels)}")
            print(f"标签映射: {self.name_to_id}")
            
            # 训练模型
            self.recognizer.train(images, labels)
            
            # 保存模型
            self.save_model()
            
            # 更新已知人脸信息
            self.known_face_names = unique_names
            
            print(f"训练完成，共 {len(unique_names)} 个用户")
            return True
            
        except Exception as e:
            print(f"训练失败: {e}")
            return False
    
    def recognize_face(self, face_image):
        """识别人脸"""
        try:
            # 转换为灰度图像
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # 图像预处理：直方图均衡化提高对比度
            gray = cv2.equalizeHist(gray)
            
            # 调整图像大小
            gray = cv2.resize(gray, (150, 150))
            
            # 进行预测
            label_id, confidence = self.recognizer.predict(gray)
            
            # 调试信息
            print(f"识别调试: label_id={label_id}, confidence={confidence}")
            print(f"标签映射: {self.id_to_name}")
            print(f"已知人脸: {self.known_face_names}")
            
            # 获取对应的姓名
            if label_id in self.id_to_name:
                name = self.id_to_name[label_id]
                print(f"找到匹配: label_id={label_id} -> name={name}")
            else:
                name = "Unknown"
                print(f"未找到匹配: label_id={label_id} 不在 {list(self.id_to_name.keys())}")
            
            # 检查置信度 - LBPH的置信度越低越好
            if confidence > self.tolerance:
                print(f"置信度 {confidence} 超过阈值 {self.tolerance}，标记为Unknown")
                name = "Unknown"
            
            print(f"最终识别结果: name={name}, confidence={confidence}")
            
            return name, confidence
            
        except Exception as e:
            print(f"人脸识别失败: {e}")
            return "Unknown", 999.0
    
    def get_known_faces(self):
        """获取已知人脸列表"""
        return self.known_face_names.copy()
    
    def clear_training_data(self):
        """清空训练数据"""
        if hasattr(self, 'training_images'):
            self.training_images.clear()
            self.training_labels.clear()
        print("训练数据已清空")