#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统测试脚本
验证修复的问题和功能
"""

import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        import time
        print("✅ time模块导入成功")
    except ImportError as e:
        print(f"❌ time模块导入失败: {e}")
        return False
    
    try:
        from ui.main_window import MainWindow
        print("✅ MainWindow导入成功")
    except ImportError as e:
        print(f"❌ MainWindow导入失败: {e}")
        return False
    
    try:
        from face_recognition.face_detector import FaceDetector
        print("✅ FaceDetector导入成功")
    except ImportError as e:
        print(f"❌ FaceDetector导入失败: {e}")
        return False
    
    try:
        from face_recognition.face_recognizer import FaceRecognizer
        print("✅ FaceRecognizer导入成功")
    except ImportError as e:
        print(f"❌ FaceRecognizer导入失败: {e}")
        return False
    
    try:
        from face_recognition.face_trainer import FaceTrainer
        print("✅ FaceTrainer导入成功")
    except ImportError as e:
        print(f"❌ FaceTrainer导入失败: {e}")
        return False
    
    return True

def test_components():
    """测试组件初始化"""
    print("\n🔍 测试组件初始化...")
    
    try:
        from face_recognition.face_detector import FaceDetector
        detector = FaceDetector()
        print("✅ FaceDetector初始化成功")
    except Exception as e:
        print(f"❌ FaceDetector初始化失败: {e}")
        return False
    
    try:
        from face_recognition.face_recognizer import FaceRecognizer
        recognizer = FaceRecognizer()
        print("✅ FaceRecognizer初始化成功")
    except Exception as e:
        print(f"❌ FaceRecognizer初始化失败: {e}")
        return False
    
    try:
        from face_recognition.face_trainer import FaceTrainer
        trainer = FaceTrainer(detector, recognizer)
        print("✅ FaceTrainer初始化成功")
    except Exception as e:
        print(f"❌ FaceTrainer初始化失败: {e}")
        return False
    
    return True

def test_config():
    """测试配置文件"""
    print("\n🔍 测试配置文件...")
    
    config_path = "config/config.yaml"
    if os.path.exists(config_path):
        print(f"✅ 配置文件存在: {config_path}")
        
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("✅ 配置文件格式正确")
            print(f"   人脸检测参数: {config.get('face_detection', {})}")
            print(f"   人脸识别参数: {config.get('face_recognition', {})}")
        except Exception as e:
            print(f"❌ 配置文件读取失败: {e}")
            return False
    else:
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    return True

def test_models():
    """测试模型文件"""
    print("\n🔍 测试模型文件...")
    
    model_files = [
        "data/models/haarcascade_frontalface_default.xml",
        "data/models/face_recognizer.yml"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ 模型文件存在: {model_file}")
        else:
            print(f"⚠️  模型文件不存在: {model_file}")
    
    return True

def main():
    """主测试函数"""
    print("🚀 人脸识别系统测试开始")
    print("=" * 50)
    
    tests = [
        ("模块导入测试", test_imports),
        ("组件初始化测试", test_components),
        ("配置文件测试", test_config),
        ("模型文件测试", test_models)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} 通过")
        else:
            print(f"❌ {test_name} 失败")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统修复成功！")
        print("\n💡 使用建议:")
        print("1. 运行 'python train_faces.py' 训练新用户")
        print("2. 运行 'python main.py' 启动主程序")
        print("3. 确保摄像头正常工作")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
