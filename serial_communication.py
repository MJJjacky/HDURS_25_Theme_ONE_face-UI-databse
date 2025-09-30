#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
串口通信模块
用于接收饮品选择和用户ID，更新用户糖量摄入
"""

import serial
import threading
import time
from database.database_manager import DatabaseManager

class SerialCommunication:
    """串口通信类"""
    
    def __init__(self, port="/dev/ttyCH341USB0", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_port = None
        self.is_running = False
        self.listener_thread = None
        
        # 数据库管理器
        self.db_manager = DatabaseManager()
        
        # 当前识别的用户
        self.current_user_id = None
        self.current_user_name = None
        
        # 饮品映射
        self.drinks = {
            1: "大麦茶",
            2: "枸杞水", 
            3: "黑枸杞",
            4: "咖啡"
        }
        
        # 回调函数，用于通知Qt界面数据更新
        self.on_data_updated = None
        
        # 上次发送的数据，用于避免重复发送
        self.last_sent_data = None
    

    
    def set_current_user(self, user_id, user_name):
        """设置当前识别的用户"""
        self.current_user_id = user_id
        self.current_user_name = user_name
        print(f"串口通信: 设置当前用户 - ID: {user_id}, 姓名: {user_name}")
        
        # 发送用户信息到串口
        self.send_user_info()
    
    def clear_current_user(self):
        """清除当前用户信息"""
        print(f"串口通信: 清除当前用户 - ID: {self.current_user_id}, 姓名: {self.current_user_name}")
        self.current_user_id = None
        self.current_user_name = None
        self.last_sent_data = None
    
    def send_user_info(self):
        """发送当前用户信息到串口"""
        try:
            if self.current_user_id is not None:
                # 获取用户健康记录
                health_record = self.db_manager.get_user_health_today(self.current_user_id)
                if health_record:
                    sugar_intake = health_record[3]
                    sugar_limit = health_record[4]
                    
                    # 检查是否超过限制
                    if sugar_intake > sugar_limit:
                        # 糖量超过限制，只发送警告
                        warning_msg = f"{self.current_user_id},999"
                        self.send_data(warning_msg)
                        print(f"警告: 用户糖量摄入已超过限制! 只发送警告信息: {warning_msg}")
                        # 更新上次发送的数据为警告数据
                        self.last_sent_data = ("WARNING", self.current_user_id)
                    else:
                        # 糖量正常，发送用户信息
                        current_data = (self.current_user_id, round(sugar_intake), round(sugar_limit))
                        if self.last_sent_data != current_data:
                            message = f"{self.current_user_id},{round(sugar_intake):02d},{round(sugar_limit):02d}"
                            self.send_data(message)
                            print(f"数据有变化，发送用户信息到串口: {message}")
                            
                            # 更新上次发送的数据
                            self.last_sent_data = current_data
                        else:
                            print(f"数据无变化，不发送串口信息: {current_data}")
                else:
                    print("无法获取用户健康记录")
            else:
                print("没有当前用户，无法发送用户信息")
        except Exception as e:
            print(f"发送用户信息失败: {e}")
    
    def start(self):
        """启动串口通信"""
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            self.is_running = True
            
            # 启动监听线程
            self.listener_thread = threading.Thread(target=self._listen_serial)
            self.listener_thread.daemon = True
            self.listener_thread.start()
            
            print(f"串口通信已启动: {self.port}")
            return True
            
        except Exception as e:
            print(f"启动串口通信失败: {e}")
            return False
    
    def stop(self):
        """停止串口通信"""
        self.is_running = False
        
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.serial_port = None
        
        if self.listener_thread:
            self.listener_thread.join(timeout=1)
        
        print("串口通信已停止")
    
    def _listen_serial(self):
        """串口监听线程"""
        while self.is_running:
            try:
                if self.serial_port and self.serial_port.in_waiting > 0:
                    # 读取数据
                    data = self.serial_port.readline().decode('utf-8').strip()
                    
                    if data:
                        self._process_serial_data(data)
                        
            except Exception as e:
                print(f"串口读取错误: {e}")
                time.sleep(0.1)
            
            time.sleep(0.01)  # 避免CPU占用过高
    
    def _process_serial_data(self, data):
        """处理串口数据"""
        try:
            print(f"收到串口数据: {data}")
            
            # 只接收饮品ID（单个数字）
            try:
                drink_id = int(data)
                
                # 验证饮品ID
                if drink_id in self.drinks:
                    drink_name = self.drinks[drink_id]
                    
                    # 检查是否有当前识别用户
                    if self.current_user_id is not None:
                        print(f"用户 {self.current_user_name} 选择了 {drink_name}")
                        
                        # 更新数据库
                        result = self.db_manager.add_drink_consumption(self.current_user_id, drink_id)
                        print(f"数据库返回结果: {result} (类型: {type(result)})")
                        
                        if isinstance(result, tuple) and (result[0] == "SUCCESS" or result[0] == "WARNING"):
                            actual_sugar = result[1]  # 获取实际增加的糖量
                            print(f"成功更新用户 {self.current_user_name} 的糖量摄入")
                            
                            # 获取更新后的健康记录
                            health_record = self.db_manager.get_user_health_today(self.current_user_id)
                            if health_record:
                                sugar_intake = health_record[3]
                                sugar_limit = health_record[4]
                                print(f"当前糖量摄入: {sugar_intake}g / {sugar_limit}g")
                                
                                # 发送更新后的用户信息到串口 (纯数字，糖量四舍五入为两位数，英文逗号)
                                current_data = (self.current_user_id, round(sugar_intake), round(sugar_limit))
                                
                                # 检查是否超过限制
                                if sugar_intake > sugar_limit:
                                    # 糖量超过限制，只发送警告
                                    warning_msg = f"{self.current_user_id},999"
                                    self.send_data(warning_msg)
                                    print(f"警告: 用户糖量摄入已超过限制! 只发送警告信息: {warning_msg}")
                                    # 更新上次发送的数据为警告数据
                                    self.last_sent_data = ("WARNING", self.current_user_id)
                                else:
                                    # 糖量正常，发送用户信息
                                    if self.last_sent_data != current_data:
                                        message = f"{self.current_user_id},{round(sugar_intake):02d},{round(sugar_limit):02d}"
                                        self.send_data(message)
                                        print(f"数据有变化，发送更新信息到串口: {message}")
                                        
                                        # 更新上次发送的数据
                                        self.last_sent_data = current_data
                                    else:
                                        print(f"数据无变化，不发送串口信息: {current_data}")
                                
                                # 通知Qt界面刷新显示，传递实际增加的糖量
                                if self.on_data_updated:
                                    self.on_data_updated(self.current_user_id, self.current_user_name, actual_sugar)
                                
                                # 移除重复的警告检测，因为上面已经处理了
                                print(f"饮品消费处理完成，结果: {result}")
                        else:
                            print(f"更新用户 {self.current_user_name} 糖量摄入失败，返回结果: {result}")
                    else:
                        print("没有识别到用户，无法添加饮品消费")
                        # 发送错误信息到串口 (纯数字: 888表示错误)
                        self.send_data("888,0")
                else:
                    print(f"无效的饮品ID: {drink_id}")
                    # 发送错误信息到串口 (纯数字: 777表示无效饮品)
                    self.send_data(f"777,{drink_id}")
                    
            except ValueError:
                print(f"数据格式错误: {data}，应为单个数字")
                # 发送错误信息到串口 (纯数字: 666表示格式错误)
                self.send_data("666,0")
                
        except Exception as e:
            print(f"处理串口数据失败: {e}")
            # 发送错误信息到串口 (纯数字: 555表示处理失败)
            self.send_data("555,0")
    
    def send_data(self, data):
        """发送数据到串口"""
        try:
            if self.serial_port and self.serial_port.is_open:
                message = f"{data}\n".encode('utf-8')
                self.serial_port.write(message)
                print(f"发送数据: {data}")
                return True
        except Exception as e:
            print(f"发送数据失败: {e}")
        return False
    
    def get_status(self):
        """获取串口状态"""
        if self.serial_port and self.serial_port.is_open:
            return f"已连接 - {self.port}"
        else:
            return "未连接"
    
    def get_available_ports(self):
        """获取可用串口列表"""
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    
    def get_current_port(self):
        """获取当前串口端口"""
        return self.port

def main():
    """测试串口通信"""
    print("=== 串口通信测试 ===")
    
    # 获取可用串口
    comm = SerialCommunication()
    available_ports = comm.get_available_ports()
    
    if not available_ports:
        print("没有找到可用的串口")
        return
    
    print(f"可用串口: {available_ports}")
    
    # 选择串口
    port = available_ports[0] if available_ports else '/dev/ttyCH341USB0'
    print(f"使用串口: {port}")
    
    # 启动串口通信
    if comm.start():
        print("串口通信启动成功")
        
        try:
            # 模拟设置用户
            comm.set_current_user(1, "测试用户")
            
            # 等待用户输入
            print("请输入饮品ID (1-4) 或 'quit' 退出:")
            while True:
                user_input = input().strip()
                if user_input.lower() == 'quit':
                    break
                
                try:
                    drink_id = int(user_input)
                    if drink_id in [1, 2, 3, 4]:
                        comm._process_serial_data(str(drink_id))
                    else:
                        print("无效的饮品ID，请输入1-4")
                except ValueError:
                    print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            comm.stop()
    else:
        print("串口通信启动失败")

if __name__ == "__main__":
    main()
