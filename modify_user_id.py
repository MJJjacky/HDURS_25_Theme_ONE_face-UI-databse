#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改用户ID脚本
"""

from database.database_manager import DatabaseManager

def modify_user_id():
    """修改用户ID"""
    db = DatabaseManager()
    
    # 显示当前用户
    print("=== 当前数据库中的用户 ===")
    users = db.get_all_users()
    for user in users:
        print(f"ID: {user[0]}, 姓名: {user[1]}, 年龄: {user[2]}")
    
    print("\n=== 修改用户ID ===")
    
    # 输入要修改的用户姓名
    user_name = input("请输入要修改的用户姓名 (默认: 梅俊杰): ").strip()
    if not user_name:
        user_name = "梅俊杰"
    
    # 输入新的用户ID
    try:
        new_id = int(input("请输入新的用户ID (默认: 1): ").strip() or "1")
    except ValueError:
        print("❌ 用户ID必须是数字")
        return
    
    # 检查用户是否存在
    user = db.get_user_by_name(user_name)
    if not user:
        print(f"❌ 用户 '{user_name}' 不存在")
        return
    
    old_id = user[0]
    print(f"\n当前用户: {user_name}")
    print(f"当前ID: {old_id}")
    print(f"新ID: {new_id}")
    
    # 确认修改
    confirm = input("\n确认修改? (y/N): ").strip().lower()
    if confirm != 'y':
        print("取消修改")
        return
    
    try:
        # 修改用户ID
        db.modify_user_id(old_id, new_id)
        print(f"✅ 成功将用户 '{user_name}' 的ID从 {old_id} 修改为 {new_id}")
        
        # 显示修改后的结果
        print("\n=== 修改后的用户列表 ===")
        users = db.get_all_users()
        for user in users:
            print(f"ID: {user[0]}, 姓名: {user[1]}, 年龄: {user[2]}")
            
    except Exception as e:
        print(f"❌ 修改失败: {e}")

if __name__ == "__main__":
    modify_user_id()
