#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试人脸识别模型是否正确加载
"""

import os
import sys
import pickle

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from face_recognition.face_recognizer import FaceRecognizer

def test_model_loading():
    """测试模型加载"""
    print("=== 测试人脸识别模型加载 ===")
    
    # 创建识别器
    recognizer = FaceRecognizer()
    
    print(f"\n1. 模型路径: {recognizer.model_path}")
    print(f"2. 模型文件存在: {os.path.exists(recognizer.model_path)}")
    
    # 检查标签映射文件
    label_map_path = recognizer.model_path.replace('.yml', '_labels.pkl')
    print(f"3. 标签映射文件存在: {os.path.exists(label_map_path)}")
    
    if os.path.exists(label_map_path):
        try:
            with open(label_map_path, 'rb') as f:
                label_data = pickle.load(f)
                print(f"4. 标签映射内容:")
                print(f"   name_to_id: {label_data.get('name_to_id', {})}")
                print(f"   id_to_name: {label_data.get('id_to_name', {})}")
        except Exception as e:
            print(f"4. 读取标签映射失败: {e}")
    
    print(f"\n5. 识别器状态:")
    print(f"   tolerance: {recognizer.tolerance}")
    print(f"   known_face_names: {recognizer.known_face_names}")
    print(f"   name_to_id: {recognizer.name_to_id}")
    print(f"   id_to_name: {recognizer.id_to_name}")
    
    # 检查训练数据
    if hasattr(recognizer, 'training_images'):
        print(f"   training_images: {len(recognizer.training_images)} 个")
        print(f"   training_labels: {recognizer.training_labels}")
    else:
        print(f"   training_images: 未初始化")
    
    return recognizer

def test_simple_recognition(recognizer):
    """测试简单识别"""
    print("\n=== 测试简单识别 ===")
    
    # 创建一个简单的测试图像（随机噪声）
    import numpy as np
    test_image = np.random.randint(0, 255, (150, 150), dtype=np.uint8)
    
    try:
        name, confidence = recognizer.recognize_face(test_image)
        print(f"测试识别结果: name={name}, confidence={confidence}")
    except Exception as e:
        print(f"识别测试失败: {e}")

if __name__ == "__main__":
    recognizer = test_model_loading()
    test_simple_recognition(recognizer)
