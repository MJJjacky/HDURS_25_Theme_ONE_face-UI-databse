#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA检测脚本 - 检查CUDA、cuDNN和GPU状态
"""

import sys
import os

def check_python_version():
    """检查Python版本"""
    print("=== Python版本检查 ===")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print()

def check_cuda_python():
    """检查CUDA Python包"""
    print("=== CUDA Python包检查 ===")
    
    # 检查torch
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA可用")
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   cuDNN版本: {torch.backends.cudnn.version()}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("❌ PyTorch CUDA不可用")
    except ImportError:
        print("❌ PyTorch未安装")
    
    print()
    
    # 检查tensorflow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow版本: {tf.__version__}")
        if tf.config.list_physical_devices('GPU'):
            print(f"✅ TensorFlow GPU可用")
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                print(f"   GPU: {gpu}")
        else:
            print("❌ TensorFlow GPU不可用")
    except ImportError:
        print("❌ TensorFlow未安装")
    
    print()

def check_dlib_cuda():
    """检查dlib CUDA支持"""
    print("=== dlib CUDA支持检查 ===")
    
    try:
        import dlib
        print(f"✅ dlib版本: {dlib.__version__}")
        
        # 检查dlib是否支持CUDA
        if hasattr(dlib, 'cuda_get_num_devices'):
            try:
                num_devices = dlib.cuda_get_num_devices()
                print(f"✅ dlib CUDA支持: {num_devices} 个GPU设备")
                
                for i in range(num_devices):
                    try:
                        device_name = dlib.cuda_get_device_name(i)
                        print(f"   GPU {i}: {device_name}")
                    except:
                        print(f"   GPU {i}: 无法获取名称")
                        
            except Exception as e:
                print(f"❌ dlib CUDA检测失败: {e}")
        else:
            print("❌ dlib不支持CUDA")
            
    except ImportError:
        print("❌ dlib未安装")
    
    print()

def check_system_cuda():
    """检查系统CUDA"""
    print("=== 系统CUDA检查 ===")
    
    # 检查CUDA_HOME环境变量
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        print(f"✅ CUDA_HOME: {cuda_home}")
    else:
        print("❌ CUDA_HOME未设置")
    
    # 检查PATH中的CUDA
    cuda_in_path = False
    for path in os.environ.get('PATH', '').split(':'):
        if 'cuda' in path.lower():
            cuda_in_path = True
            print(f"✅ CUDA在PATH中: {path}")
            break
    
    if not cuda_in_path:
        print("❌ CUDA不在PATH中")
    
    # 检查nvcc命令
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvcc命令可用")
            # 提取CUDA版本
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"   系统CUDA版本: {line.strip()}")
                    break
        else:
            print("❌ nvcc命令不可用")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvcc命令未找到")
    
    print()

def check_gpu_info():
    """检查GPU信息"""
    print("=== GPU硬件信息检查 ===")
    
    try:
        import subprocess
        
        # 检查nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ nvidia-smi可用")
                # 显示GPU信息
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'NVIDIA' in line and 'GPU' in line:
                        print(f"   {line.strip()}")
            else:
                print("❌ nvidia-smi不可用")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ nvidia-smi未找到")
        
        # 检查lspci (Linux)
        try:
            result = subprocess.run(['lspci'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_lines = [line for line in result.stdout.split('\n') 
                           if 'nvidia' in line.lower() or 'gpu' in line.lower()]
                if gpu_lines:
                    print("✅ 发现GPU设备:")
                    for line in gpu_lines[:3]:  # 只显示前3个
                        print(f"   {line.strip()}")
                else:
                    print("❌ 未发现GPU设备")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ lspci命令不可用")
            
    except Exception as e:
        print(f"❌ GPU信息检查失败: {e}")
    
    print()

def check_dlib_models():
    """检查dlib模型文件"""
    print("=== dlib模型文件检查 ===")
    
    model_files = [
        "data/models/shape_predictor_68_face_landmarks.dat",
        "data/models/dlib_face_recognition_resnet_model_v1.dat"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"✅ {model_file} - {size:.1f} MB")
        else:
            print(f"❌ {model_file} - 缺失")
    
    print()

def main():
    """主函数"""
    print("=== CUDA环境检测工具 ===")
    print("此工具将检查您的CUDA环境配置")
    print()
    
    check_python_version()
    check_cuda_python()
    check_dlib_cuda()
    check_system_cuda()
    check_gpu_info()
    check_dlib_models()
    
    print("=== 检测完成 ===")
    print()
    print("建议:")
    print("1. 如果dlib CUDA支持正常，可以移除CPU强制模式")
    print("2. 如果CUDA有问题，建议使用CPU模式")
    print("3. 检查Jetson Orin Nano的CUDA驱动是否正确安装")

if __name__ == "__main__":
    main()
