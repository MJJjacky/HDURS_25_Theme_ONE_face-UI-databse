#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块
"""

import yaml
import os
from typing import Dict, Any

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                return self.get_default_config()
        else:
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'camera': {
                'device_id': 0,
                'width': 640,
                'height': 480,
                'fps': 30
            },
            'face_detection': {
                'model_path': 'haarcascade_frontalface_default.xml',
                'min_face_size': 30,
                'scale_factor': 1.1,
                'min_neighbors': 5
            },
            'face_recognition': {
                'tolerance': 0.6,
                'model_path': 'data/models/face_encoding.pkl',
                'face_size': 150
            },
            'database': {
                'path': 'database/face_recognition.db',
                'backup_path': 'database/backup/'
            },
            'training': {
                'samples_per_person': 10,
                'face_size': 150,
                'encoding_method': 'dlib'
            },
            'ui': {
                'window_width': 1200,
                'window_height': 800,
                'theme': 'light'
            }
        }
    
    def get(self, key: str, default=None):
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self):
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            print(f"保存配置文件失败: {e}")
            return False
    
    def reload_config(self):
        """重新加载配置"""
        self.config = self.load_config()

# 创建全局配置实例
config = ConfigManager()