# -*- coding:utf-8 -*-
import sys
sys.path.append('C:\\Users\\Brainlab\\Desktop\\real_time')
import serial
import time
import sys


class TodoDriver(object):
    def __init__(self, port='COM5', baudrate=115200, timeout=1):
        self.com = serial.Serial(
            # window
            port=port, # window
            # # Linux
            # port='/dev/ttyUSB0',
            baudrate=baudrate,
            timeout=timeout)

        """
        	# 1. get params
        	# values are roughly obtained by doing few experiments
        	# => 1) go   (b'47 100\n')
        	# => 2) wait (run_time)   default: run_time = 0.5
        	# => 3) stop (b'47 41\n')
        	# examples,
        	# # order = b'47 100\n'   #'Go/Back Right/Left'
        	# # G/B : 0 <= back  < 47 <=  go  <= 99
        	# # R/L : 0 <= right < 41 <= left <= 99
        """
    def control_cmd(self, data):
        # Left
        if data == 2:
            self.todo_driver_run(com=self.com, x=-600, y=1200, go_time=2)
            print('Left')
        # Right
        elif data == 3:
            self.todo_driver_run(com=self.com, x=-600, y=-1200, go_time=1.5)
            print('Right')
        # Forawrd
        elif data == 0:
            self.todo_driver_run(com=self.com, x=600, y=0, go_time=3)
            print('Forward')
        # Backward
        elif data == 1:
            self.todo_driver_run(com=self.com, x=-600, y=0, go_time=3)
            print('Backward')


    @staticmethod
    def todo_driver_run(com, x=int, y=int, go_time=int):
        x += 2047
        y += 2047
        # # go cmd
        if len(sys.argv) >= 3:  # get params from terminal
            # x, y
            go_cmd = sys.argv[1] + ' ' + sys.argv[2] + '\n'
            # time
            if len(sys.argv) >= 4:
                go_time = float(sys.argv[3])
        else:  # get params from function.
            go_cmd = 'x,' + str(x) + ',y,' + str(y) + '\n'  # 건대 1차수정

        # stop cmd
        x_stop = '2047'
        y_stop = '2047'
        stop_cmd = 'x,' + x_stop + ',y,' + y_stop + '\n'  # 건대 1차수정

        #######################
        # 2. encode
        #######################
        go_cmd = go_cmd.encode()  # encode
        stop_cmd = stop_cmd.encode()

        #######################
        # 3. send to esp32
        #######################
        # go
        com.write(go_cmd)
        # print("go_cmd: ", go_cmd)

        # run time
        # go_time = 0.5
        time.sleep(go_time)

        # stop
        com.write(stop_cmd)
        # print("stop_cmd:", stop_cmd)

        # 1차수정
        # 함수안에 com.close()를 실행하면 BLE연결이 끊어졌다가 다시 연결
        # com.close()
        return go_cmd


if __name__ == "__main__":
    # 전/후/좌/우 : 0/1/2/3
    con_list = [0, 3, 0, 2, 0]

    time.sleep(2)
    todo = TodoDriver()

    while todo:
        print('input: ')
        # i = print(rf'input: {int(input())}')
        i = int(input())
        todo.control_cmd(data=i)
        time.sleep(1)
        if i is 9:
            break

    # for i in con_list:
        
    #     todo.control_cmd(data=i)
    #     time.sleep(5)