import time

from ipywidgets import Controller, Button
from pynput.keyboard import Controller as key_cl
from pynput.mouse import Controller,Button
import time


def keyborad_input(string):

    '''
    :params striing: hello
    :return: None
    '''
    keyboard = key_cl()   #开始控制键盘
    keyboard.type(string)  # 键盘输入string

def mouse_click(): #  点击发送信息
    mouse = Controller()      # 开始控制鼠标
    mouse.press(Button.left)     # 按住鼠标左键
    mouse.release(Button.left)    # 放开鼠标左键



def main(number, string):
    time.sleep(5)   #  暂停五秒
    for i in range(number):
        keyborad_input(string + str(i))
        mouse_click()
        time.sleep(0.1)



if __name__ == "__main__":
    number = input("请输入发送次数， 回车继续： ")
    string = input("请输入发送内容， 回车继续： ")
    main(int(number), string)
    print("发送完成！")
    input("回车退出")
