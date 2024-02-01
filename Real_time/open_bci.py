# -*- coding:utf -8 -*-
import sys
sys.path.append('C:\\Users\\Brainlab\\Desktop\\real_time')
import socket
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from classification import real_time_classification
from todo import TodoDriver
from config.data_config import EpochConfig


class real_time_open_bci(object):
    def __init__(self, config: EpochConfig, ar_port=619, ar_host='172.20.10.2', ar_send_port=620, ar_send_host='172.20.10.14', eeg_port="COM4", serial_num='0x04', config_board='wertyui'):
        self.ar_port = ar_port
        self.ar_host = ar_host
        self.ar_send_port = ar_send_port
        self.ar_send_host = ar_send_host
        self.eeg_port = eeg_port
        self.serial_num = serial_num
        self.config_board = config_board
        self.model = real_time_classification(config=config)
        self.todo = TodoDriver()
        self.mi_dict = {0: 'Left hand', 1: 'Right hand', 2: 'Both hands', 3: 'Both feet', 4: 'Tongue'}
        self.ssvep_dict = {0: '5.45Hz', 1: '6.67Hz', 2: '7.5Hz', 3: '8.57Hz', 4: '12Hz'}

        """
        port number : "장치관리자"에서 확인

        serial number : GUI -> cyton -> serial -> manual -> 포트 선택 후 auto-scan -> system status -> 16진법으로 넣어주기

        config_board : openbci -> docs 확인 - > https://docs.openbci.com/Cyton/CytonSDK/
        """

    def real_time_analysis(self):

        # AR device connect
        print("\nWaiting for Connection...")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.ar_host, self.ar_port))
        server_socket.listen()
        client_socket, addr = server_socket.accept()
        print('\nConnected by', addr)

        # eeg device connect
        params = BrainFlowInputParams()
        params.serial_port = self.eeg_port
        params.serial_number = self.serial_num

        board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)

        # EEG 측정 시작
        board.prepare_session()
        board.config_board(self.config_board)

        # BUFFER에 데이터 쌓임
        board.start_stream()
        # EEG/ECG 신호 안정화 시간
        time.sleep(5)
        print('Start session')

        # while True:
        #     try:
        #         trigger = int(client_socket.recv(128).decode())
        #         if trigger==111:
        #             sender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #             sender.connect((self.ar_send_host, self.ar_send_port))
        #             k = 0
        #             while k == 5000:
        #                 k+=1
        #             sender.send("11".encode())
        #             print("결과 전송")
        #             sender.close()
        #     except:
        #         continue

        while True:
            try:
                trigger = int(client_socket.recv(128).decode())
                # MI classification
                if trigger:

                    ######################## WHEELCHAIR CONTROL ########################
                    # Wheelchair control (MI) --> 4 class : 전/후/좌/우
                    if trigger == 111:
                        label_dict = {0: "11", 1: "12", 2: "13", 3: "14"}
                        print('\nMI task start (Wheelchair control)')
                        out = self.mi_real_time(board=board, class_num=4, label_dict=label_dict) ##### class 정보 추가

                    # Wheelchair control (SSVEP) --> 4 class : 전/후/좌/우
                    if trigger == 222:
                        label_dict = {0: "21", 1: "22", 2: "23", 3: "24"}
                        # 21: 5.45Hz, 22: 6.67Hz, 23: 8.57Hz, 24: 12Hz
                        print('\nSSVEP task start (Wheelchair control)')
                        out = self.ssvep_real_time(board=board, class_num=4, label_dict=label_dict) ##### class 정보 추가

                    # 예측 값이 맞으면 wheelchair 구동
                    if trigger == 1234:
                        # control TODO_drive
                        self.todo.control_cmd(data=int(out))

                    ######################## ROBOT ARM CONTROL ########################
                    # Robot Arm control (MI)
                    if trigger == 333:
                        # 3 class : 가져오기/치우기/취소
                        label_dict = {0: "1-31", 1: "1-32", 2: "1-33"}
                        print('\nMI task start (Robot Arm control, 3 class)')
                        out = self.mi_real_time(board=board, class_num=3, label_dict=label_dict)  ##### class 정보 추가

                    if trigger == 444:
                        # 1 class : 되돌려놓기
                        label_dict = {0: "2-31", 1: "2-32"}
                        print('\nMI task start (Robot Arm control, 2 class)')
                        out = self.mi_real_time(board=board, class_num=2, label_dict=label_dict)  ##### class 정보 추가

                    if trigger == 555:
                        # 4 class: 왼쪽/오른쪽/밀기/당기기
                        label_dict = {0: "3-31", 1: "3-32", 2: "3-33", 3: "3-34"}
                        print('\nMI task start (Robot Arm control, 4 class)')
                        out = self.mi_real_time(board=board, class_num=4, label_dict=label_dict)  ##### class 정보 추가

                    # ===============로봇팔 SSVEP===============
                    if trigger == 666:
                        # 3 class: 가져오기/치우기/취소
                        label_dict = {0: "1-31", 1: "1-32", 2: "1-33"}
                        print('\nSSVEP task start (Robot Arm control, 3 class)')
                        out = self.ssvep_real_time(board=board, class_num=3, label_dict=label_dict) ##### class 정보 추가

                    if trigger == 777:
                        # 2 class: 되돌려놓기
                        label_dict = {0: "2-31", 1: "2-32"}
                        print('\nSSVEP task start (Robot Arm control, 2 class)')
                        out = self.ssvep_real_time(board=board, class_num=2, label_dict=label_dict) ##### class 정보 추가

                    if trigger == 888:
                        # 4 class : 왼쪽/오른쪽/밀기/당기기
                        label_dict = {0: "3-31", 1: "3-32", 2: "3-33", 3: "3-34"}
                        print('\nSSVEP task start (Robot Arm control, 4 class)')
                        out = self.ssvep_real_time(board=board, class_num=4, label_dict=label_dict) ##### class 정보 추가

                    if trigger == 999:
                        board.stop_stream()
                        print('\nEnd session')
                        break
            except:
                continue

    def mi_real_time(self, board, class_num, label_dict):
        # drop dummy data
        board.get_board_data()

        # get mi data
        print('\nEEG data acquisition ...')
        time.sleep(4.5)
        # data = board.get_board_data(501)
        data = board.get_board_data()

        # preprocessing & classification
        print('\nEEG data preprocessing')
        # data = data[1:10, :] / 1e6  
        data = data[1:10, :501] / 1e6  
        # print(data.shape) # --> (9, 501)

        # classification
        out = self.model.mi_model(data=data, class_num=class_num)  ################ class 정보 추가하기

        print(rf"MI output: {label_dict[out]}")
        output = label_dict[out]
        sender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sender.connect((self.ar_send_host, self.ar_send_port))
        sender.send(output.encode())
        print("결과 전송")
        sender.close()

        return out

    def ssvep_real_time(self, board, class_num, label_dict):
        # drop dummy data
        board.get_board_data()

        # get mi data
        print('\nEEG data acquisition ...')
        time.sleep(5.5)
        # data = board.get_board_data(626)
        data = board.get_board_data()

        # preprocessing & classification
        print('\nEEG data preprocessing')
        # data = data[1:10, :] / 1e6 
        data = data[1:10, :626] / 1e6 
        # print(data.shape) # --> (9, 626)

        # classification
        out = self.model.ssvep_model(data=data, class_num=class_num)  ################ class 정보 추가하기
        print(rf"SSVEP output: {label_dict[out]}")
        output = label_dict[out]
        sender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sender.connect((self.ar_send_host, self.ar_send_port))
        sender.send(output.encode())
        print("결과 전송")
        sender.close()
        return out


if __name__ == "__main__":
    config = EpochConfig(sub_name='jung_woo')
    real_time_open_bci(config=config).real_time_analysis()
