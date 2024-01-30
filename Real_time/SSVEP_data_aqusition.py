# -*- coding:utf -8 -*-
import os
import datetime as dt
import socket
import time
import winsound
import pandas as pd
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

# Subject number
sub_num = '05'

# HoloLens 2 HOST and PORT number
# HOST = '172.20.10.2'
HOST = '192.168.0.101'
PORT = 619

# data path
data_path = r'A:\BrainLab_RealTimeSSVEP\DataBase\Raw'

# Make dir
try:
    if not os.path.exists(rf'{data_path}\DataBase\Raw\sub_{sub_num}'):
        os.makedirs(rf'{data_path}\DataBase\Raw\sub_{sub_num}')
except OSError:
    print('Error: Creating directory.')

date = dt.datetime.now()
date = date.strftime("%Y-%m-%d-%H%M")
PATH = rf'{data_path}\DataBase\Raw\sub_{sub_num}\sub_{sub_num}_{date}.csv'
print(PATH)

# EEG 기기 설정
params = BrainFlowInputParams()
params.serial_port = "COM3"  # 포트 번호를 "장치 관리자"에서 확인
board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)

# EEG 측정 시작
board.prepare_session()
board.config_board('!@#$%^&*QWERTYUI')
board.start_stream()
time.sleep(5)  # EEG 신호 안정화 시간

print("\nWaiting for Connection...")
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()
client_socket, addr = server_socket.accept()
print('\nConnected by', addr)

while True:
    try:
        data = client_socket.recv(1024).decode()
        trigger = int(data)
        if trigger:
            board.insert_marker(trigger)
            print(f"Trigger inserted: {trigger}")
            if trigger == 999:
                break
    except:
        continue

DataFilter.write_file(board.get_board_data(), PATH, 'a')
winsound.Beep(frequency=440, duration=100)

# 데이터 DataFrame 으로 변환
restore_data = DataFilter.read_file(PATH)
restore_df = pd.DataFrame(np.transpose(restore_data))
restore_df.to_csv(PATH, index=False)

# EEG 측정 종료
board.stop_stream()
board.release_session()
print("Data are normally saved")

# 측정 시간 확인
df = pd.read_csv(PATH)

stim_start = df[df['31'] == 11.0].index[0]
stim_end = len(df)

ssvep_time = (stim_end-stim_start)/(125*60)
s_min = int(ssvep_time)
s_sec = (ssvep_time-s_min)*60
print("설정한 시간: 5 min 24 sec")
print(rf"실제 측정된 시간: {s_min} min {s_sec: .0f} sec")