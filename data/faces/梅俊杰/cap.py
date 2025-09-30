#!/usr/bin/env python3
# -*- coding: utf-8 -*-：
import cv2
import os

def capture_image():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误：无法访问摄像头")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("错误：无法获取图像")
        return
    
    count = 1
    while os.path.exists(f"{count}.jpg"):
        count += 1
    
    filename = f"{count}.jpg"
    cv2.imwrite(filename, frame)
    print(f"照片已保存为: {filename}")

if __name__ == "__main__":
    capture_image()