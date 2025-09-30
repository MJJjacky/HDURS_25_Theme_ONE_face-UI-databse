from dataclasses import dataclass
from datetime import date
from typing import Optional

@dataclass
class User:
    """用户模型"""
    id: Optional[int]
    name: str
    age: int
    gender: str
    face_encoding_id: Optional[str]
    created_at: Optional[str]
    
    def __post_init__(self):
        if self.id is None:
            self.id = 0
        if self.face_encoding_id is None:
            self.face_encoding_id = ""
        if self.created_at is None:
            self.created_at = ""

@dataclass
class HealthRecord:
    """健康记录模型"""
    id: Optional[int]
    user_id: int
    date: str
    sugar_intake: float
    sugar_limit: float
    notes: Optional[str]
    
    def __post_init__(self):
        if self.id is None:
            self.id = 0
        if self.notes is None:
            self.notes = ""
        if self.sugar_limit == 0:
            self.sugar_limit = 50.0

@dataclass
class FaceEncoding:
    """人脸编码模型"""
    id: str
    user_id: int
    encoding_data: str
