#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载dlib模型文件脚本
"""

import os
import urllib.request
import zipfile
import tarfile

def download_file(url, filename):
    """下载文件"""
    print(f"正在下载 {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ {filename} 下载完成")
        return True
    except Exception as e:
        print(f"❌ {filename} 下载失败: {e}")
        return False

def extract_archive(archive_path, extract_to):
    """解压文件"""
    print(f"正在解压 {archive_path}...")
    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith('.tar.bz2'):
            with tarfile.open(archive_path, 'r:bz2') as tar_ref:
                tar_ref.extractall(extract_to)
        print(f"✅ 解压完成")
        return True
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return False

def main():
    """主函数"""
    print("=== dlib模型文件下载工具 ===")
    print()
    
    # 创建模型目录
    models_dir = "data/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # 模型文件URL
    models = {
        "shape_predictor_68_face_landmarks.dat": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        "dlib_face_recognition_resnet_model_v1.dat": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    }
    
    print("需要下载的模型文件:")
    for name, url in models.items():
        print(f"  - {name}")
    print()
    
    # 检查是否已存在
    existing_models = []
    for name in models.keys():
        if os.path.exists(os.path.join(models_dir, name)):
            existing_models.append(name)
            print(f"✅ {name} 已存在")
    
    if existing_models:
        print(f"\n发现 {len(existing_models)} 个已存在的模型文件")
        choice = input("是否重新下载? (y/N): ").strip().lower()
        if choice != 'y':
            print("跳过下载")
            return
    
    print("\n开始下载模型文件...")
    
    # 下载模型文件
    for name, url in models.items():
        if os.path.exists(os.path.join(models_dir, name)):
            print(f"跳过 {name} (已存在)")
            continue
            
        # 下载压缩文件
        archive_name = url.split('/')[-1]
        archive_path = os.path.join(models_dir, archive_name)
        
        if download_file(url, archive_path):
            # 解压文件
            if extract_archive(archive_path, models_dir):
                # 删除压缩文件
                os.remove(archive_path)
                print(f"✅ {name} 处理完成")
            else:
                print(f"❌ {name} 解压失败")
        else:
            print(f"❌ {name} 下载失败")
    
    print("\n=== 下载完成 ===")
    print("模型文件位置:", models_dir)
    
    # 检查最终结果
    print("\n最终状态:")
    for name in models.keys():
        model_path = os.path.join(models_dir, name)
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"✅ {name} - {size:.1f} MB")
        else:
            print(f"❌ {name} - 缺失")

if __name__ == "__main__":
    main()
