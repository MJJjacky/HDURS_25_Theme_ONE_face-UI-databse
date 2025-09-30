import cv2
import numpy as np
import os

class FaceDetector:
    """人脸检测器，使用OpenCV的Haar级联分类器"""
    
    def __init__(self, cascade_path=None):
        if cascade_path is None:
            # 使用OpenCV内置的人脸检测模型
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError(f"无法加载人脸检测模型: {cascade_path}")
    
    def detect_faces(self, image, scale_factor=1.05, min_neighbors=6, min_size=(50, 50)):
        """
        检测图像中的人脸
        
        Args:
            image: 输入图像
            scale_factor: 图像缩放因子（更小的值提高精度但降低速度）
            min_neighbors: 最小邻居数（更高的值减少误检）
            min_size: 最小人脸尺寸
            
        Returns:
            faces: 检测到的人脸矩形框列表 [(x, y, w, h), ...]
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 图像预处理：直方图均衡化提高检测效果
        gray = cv2.equalizeHist(gray)
        
        # 人脸检测
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        return faces
    
    def extract_largest_face(self, image):
        """提取最大的人脸区域"""
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            return None, None
        
        # 获取最大的人脸
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # 提取人脸区域
        face_roi = image[y:y+h, x:x+w]
        
        return face_roi, largest_face