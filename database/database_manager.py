import sqlite3
import os
import json
from datetime import datetime
import numpy as np

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path="database/face_recognition.db"):
        self.db_path = db_path
        self.ensure_db_directory()
        self.init_database()
    
    def get_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)
    
    def ensure_db_directory(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 用户表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER,
                    gender TEXT,
                    face_encoding_id TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 人脸编码表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_encodings (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    encoding_data TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # 人脸图片表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    image_path TEXT NOT NULL,
                    person_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # 饮品糖量表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drinks (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    sugar_content REAL NOT NULL,
                    description TEXT
                )
            ''')
            
            # 健康记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    date DATE NOT NULL,
                    sugar_intake REAL DEFAULT 0.0,
                    sugar_limit REAL DEFAULT 50.0,
                    notes TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # 初始化饮品表
            cursor.execute('''
                INSERT OR IGNORE INTO drinks (id, name, sugar_content, description) VALUES 
                (1, '大麦茶', 0.5, '大麦茶'),
                (2, '枸杞水', 3, '枸杞水'),
                (3, '黑枸杞', 5, '黑枸杞'),
                (4, '咖啡', 10, '美式咖啡')
            ''')
            
            conn.commit()
    
    def add_user(self, name, age, gender="未知"):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (name, age, gender)
                VALUES (?, ?, ?)
            ''', (name, age, gender))
            return cursor.lastrowid
    
    def add_face_encoding(self, user_id, face_encoding):
        encoding_id = f"face_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        encoding_data = json.dumps(face_encoding.tolist())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO face_encodings (id, user_id, encoding_data)
                VALUES (?, ?, ?)
            ''', (encoding_id, user_id, encoding_data))
            
            cursor.execute('''
                UPDATE users SET face_encoding_id = ? WHERE id = ?
            ''', (encoding_id, user_id))
            
            conn.commit()
            return encoding_id
    
    def get_all_users(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users ORDER BY created_at DESC')
            return cursor.fetchall()
    
    def get_all_face_encodings(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT user_id, encoding_data FROM face_encodings')
            rows = cursor.fetchall()
            
            encodings = []
            for row in rows:
                user_id = row[0]
                encoding_list = json.loads(row[1])
                encoding = np.array(encoding_list)
                encodings.append((user_id, encoding))
            return encodings
    
    def add_health_record(self, user_id, date, sugar_intake, sugar_limit=50.0):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO health_records (user_id, date, sugar_intake, sugar_limit)
                VALUES (?, ?, ?, ?)
            ''', (user_id, date, sugar_intake, sugar_limit))
            conn.commit()
    
    def get_health_records(self, user_id, date=None):
        """获取健康记录"""
        try:
            print(f"=== 数据库查询: 获取用户 {user_id} 的健康记录 ===")
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if date:
                    print(f"查询条件: 用户ID={user_id}, 日期={date}")
                    cursor.execute('''
                        SELECT * FROM health_records 
                        WHERE user_id = ? AND date = ?
                    ''', (user_id, date))
                else:
                    # 默认获取今天的记录
                    today = datetime.now().strftime("%Y-%m-%d")
                    print(f"查询条件: 用户ID={user_id}, 今天日期={today}")
                    cursor.execute('''
                        SELECT * FROM health_records 
                        WHERE user_id = ? AND date = ? ORDER BY id DESC
                    ''', (user_id, today))
                
                records = cursor.fetchall()
                print(f"查询结果: 获取到 {len(records)} 条记录")
                
                for i, record in enumerate(records):
                    print(f"  记录 {i}: ID={record[0]}, 用户ID={record[1]}, 日期={record[2]}, 糖量={record[3]}, 限制={record[4]}")
                
                return records
                
        except Exception as e:
            print(f"❌ 获取健康记录失败: {e}")
            return []
            
    def delete_user(self, user_id):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 首先删除相关的人脸编码
                cursor.execute('DELETE FROM face_encodings WHERE user_id = ?', (user_id,))
                
                # 删除健康记录
                cursor.execute('DELETE FROM health_records WHERE user_id = ?', (user_id,))
                
                # 最后删除用户
                cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
                
                conn.commit()
                return True
        except Exception as e:
            print(f"删除用户失败: {e}")
            return False

    def add_face_image(self, user_id, image_path, person_name):
        """添加人脸图片路径到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 检查是否已存在该图片路径
                cursor.execute('''
                    SELECT id FROM face_images WHERE image_path = ? AND user_id = ?
                ''', (image_path, user_id))
                
                if cursor.fetchone() is None:
                    # 插入新的人脸图片记录
                    cursor.execute('''
                        INSERT INTO face_images (user_id, image_path, person_name, created_at)
                        VALUES (?, ?, ?, ?)
                    ''', (user_id, image_path, person_name, datetime.now()))
                    
                    conn.commit()
                    print(f"人脸图片路径已保存到数据库: {image_path}")
                    return True
                else:
                    print(f"人脸图片路径已存在: {image_path}")
                    return False
                    
        except Exception as e:
            print(f"保存人脸图片路径失败: {e}")
            return False
    
    def get_user_face_images(self, user_id):
        """获取用户的人脸图片路径"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT image_path, created_at FROM face_images 
                    WHERE user_id = ? ORDER BY created_at DESC
                ''', (user_id,))
                return cursor.fetchall()
        except Exception as e:
            print(f"获取用户人脸图片失败: {e}")
            return []
    
    def get_user_by_name(self, name):
        """根据姓名获取用户信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM users WHERE name = ?', (name,))
                return cursor.fetchone()
        except Exception as e:
            print(f"获取用户信息失败: {e}")
            return None
    
    def get_user_health_today(self, user_id):
        """获取用户今日健康记录"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM health_records 
                    WHERE user_id = ? AND date = ?
                ''', (user_id, today))
                record = cursor.fetchone()
                
                if record:
                    return record
                else:
                    # 如果没有今日记录，创建一个默认记录
                    cursor.execute('''
                        INSERT INTO health_records (user_id, date, sugar_intake, sugar_limit)
                        VALUES (?, ?, 0.0, 50.0)
                    ''', (user_id, today))
                    conn.commit()
                    
                    # 返回新创建的记录
                    cursor.execute('''
                        SELECT * FROM health_records 
                        WHERE user_id = ? AND date = ?
                    ''', (user_id, today))
                    return cursor.fetchone()
                    
        except Exception as e:
            print(f"获取用户今日健康记录失败: {e}")
            return None
    
    def get_user_health_today_id(self, user_id):
        """获取用户今日健康记录的ID"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id FROM health_records 
                    WHERE user_id = ? AND date = ?
                ''', (user_id, today))
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            print(f"获取用户今日健康记录ID失败: {e}")
            return None
    
    def update_health_record_sugar(self, record_id, new_sugar_intake):
        """更新健康记录的糖分摄入量"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE health_records 
                    SET sugar_intake = ? 
                    WHERE id = ?
                ''', (new_sugar_intake, record_id))
                conn.commit()
                print(f"✅ 成功更新健康记录 {record_id} 的糖分摄入量为 {new_sugar_intake}g")
                return True
        except Exception as e:
            print(f"❌ 更新健康记录糖分摄入量失败: {e}")
            return False
    
    def add_drink_consumption(self, user_id, drink_id):
        """添加饮品消费，更新用户糖量摄入"""
        try:
            # 饮品含糖量映射 (g)
            drink_sugar = {
                1: 0.5,  # 大麦茶: 0.5g
                2: 3,    # 枸杞水: 3g
                3: 5,    # 黑枸杞: 5g
                4: 10    # 咖啡: 10g
            }
            
            if drink_id not in drink_sugar:
                print(f"❌ 无效的饮品ID: {drink_id}")
                return False
            
            # 基础糖量
            base_sugar = drink_sugar[drink_id]
            
            # 添加±3g随机波动，模拟实际差异
            import random
            sugar_variation = random.uniform(-3, 3)
            actual_sugar = max(0, base_sugar + sugar_variation)  # 确保糖量不为负数
            
            print(f"饮品ID {drink_id} 基础糖量: {base_sugar}g, 波动: {sugar_variation:+.1f}g, 实际糖量: {actual_sugar:.1f}g")
            
            # 获取今日健康记录
            health_record = self.get_user_health_today(user_id)
            if health_record:
                current_sugar = health_record[3]
                sugar_limit = health_record[4]  # 获取糖量限制
                new_sugar = current_sugar + actual_sugar
                
                # 更新今日糖量摄入
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE health_records SET sugar_intake = ? WHERE id = ?",
                        (new_sugar, health_record[0])
                    )
                    conn.commit()
                
                print(f"✅ 用户 {user_id} 今日糖量摄入: {current_sugar:.1f}g + {actual_sugar:.1f}g = {new_sugar:.1f}g")
                print(f"当前糖量: {new_sugar:.1f}g, 限制: {sugar_limit:.1f}g")
                
                # 检查是否超过限制
                if new_sugar > sugar_limit:
                    print(f"⚠️ 警告: 用户 {user_id} 糖量摄入已超过限制! 当前: {new_sugar:.1f}g, 限制: {sugar_limit:.1f}g")
                    return ("WARNING", actual_sugar)  # 返回警告标识和实际糖量
                else:
                    return ("SUCCESS", actual_sugar)  # 返回成功标识和实际糖量
            else:
                print(f"❌ 无法获取用户 {user_id} 的健康记录")
                return False
                
        except Exception as e:
            print(f"❌ 添加饮品消费失败: {e}")
            return False
    
    def get_drinks(self):
        """获取所有饮品信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM drinks ORDER BY id')
                return cursor.fetchall()
        except Exception as e:
            print(f"获取饮品信息失败: {e}")
            return []

    def modify_user_id(self, old_id, new_id):
        """修改用户ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 检查新ID是否已存在
                cursor.execute("SELECT id FROM users WHERE id = ?", (new_id,))
                if cursor.fetchone():
                    raise Exception(f"用户ID {new_id} 已存在")
                
                # 开始事务
                conn.execute("BEGIN TRANSACTION")
                
                try:
                    # 更新users表
                    cursor.execute("UPDATE users SET id = ? WHERE id = ?", (new_id, old_id))
                    
                    # 更新health_records表
                    cursor.execute("UPDATE health_records SET user_id = ? WHERE user_id = ?", (new_id, old_id))
                    
                    # 更新face_images表
                    cursor.execute("UPDATE face_images SET user_id = ? WHERE user_id = ?", (new_id, old_id))
                    
                    # 提交事务
                    conn.commit()
                    print(f"✅ 成功修改用户ID: {old_id} -> {new_id}")
                    return True
                    
                except Exception as e:
                    # 回滚事务
                    conn.rollback()
                    raise e
                    
        except Exception as e:
            print(f"❌ 修改用户ID失败: {e}")
            return False
    
    def modify_user_info(self, user_id, new_name, new_age, new_gender):
        """修改用户基本信息"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE users SET name = ?, age = ?, gender = ? WHERE id = ?",
                    (new_name, new_age, new_gender, user_id)
                )
                conn.commit()
                print(f"✅ 成功修改用户 {user_id} 的信息")
                return True
        except Exception as e:
            print(f"❌ 修改用户信息失败: {e}")
            return False
    
    def delete_user(self, user_id):
        """删除用户"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 开始事务
                conn.execute("BEGIN TRANSACTION")
                
                try:
                    # 删除相关数据
                    cursor.execute("DELETE FROM health_records WHERE user_id = ?", (user_id,))
                    cursor.execute("DELETE FROM face_images WHERE user_id = ?", (user_id,))
                    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                    
                    # 提交事务
                    conn.commit()
                    print(f"✅ 成功删除用户 {user_id}")
                    return True
                    
                except Exception as e:
                    # 回滚事务
                    conn.rollback()
                    raise e
                    
        except Exception as e:
            print(f"❌ 删除用户失败: {e}")
            return False
