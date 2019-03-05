from djitellopy import Tello
import cv2
import zbar
import numpy as np
import time
import math
last_qrcode = ""
list_of_qrcode = dict()
i = 1
FPS = 25
e_gain = 1
for_back_velocity = 0
left_right_velocity = 0
up_down_velocity = 0
yaw_velocity = 0
speed = 50
send_rc_control = True
qr_forward_sleep = 3
rotate_sleep = 4
now_front = "N"
mission_number = 1

cv2.namedWindow('Tello')


def nothing(x):
    pass


def stop():
    global for_back_velocity
    for_back_velocity = 0
    global left_right_velocity
    left_right_velocity = 0
    global up_down_velocity
    up_down_velocity = 0
    global yaw_velocity
    yaw_velocity = 0
    rc_update()


cv2.createTrackbar('L_tresh', 'Tello', 50, 255, nothing)
cv2.createTrackbar('H_Yellow', 'Tello', 30, 179, nothing)
cv2.createTrackbar('S_Yellow', 'Tello', 0, 255, nothing)
cv2.createTrackbar('V_Yellow', 'Tello', 0, 255, nothing)


###############################################

def rc_update():
    if send_rc_control:
        print(left_right_velocity)
        tello.send_rc_control(left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity)
    else:
        print("not send control")


###################################################################3


def att_control(att, check):
    while True:
        try:

            if check:

                global list_of_qrcode
                global i
                global mission_number
                # if mission_number == 2:
                #     value_qr = qr_scanner2()
                # else:

                value_qr = qr_scanner()
                if (value_qr[0]):
                    print("*******************************************")
                    print("*******************************************")
                    print("*******************************************")
                    print("*******************************************")
                    print("*******************************************")
                    print("*************&&&&&&&&&&&&&&&&**************")
                    print("*******************************************")
                    print("*******************************************")
                    print("*******************************************")
                    print("*******************************************")
                    print("*******************************************")
                    print(value_qr[1])

            high = tello.get_distance_tof()
            h_valu = int(high.replace("mm", ""))
            print(h_valu)
            print(att)
            global for_back_velocity
            global left_right_velocity
            global yaw_velocity
            for_back_velocity = 0
            left_right_velocity = 0
            yaw_velocity = 0
            global up_down_velocity
            print(up_down_velocity)
            if h_valu < att - 100:
                up_down_velocity = 50

            elif h_valu > att + 100:
                up_down_velocity = -50
            else:
                up_down_velocity = int((att - h_valu) / 3)

            if h_valu > att - 50 and h_valu < att + 50:
                up_down_velocity = 0
                rc_update()
                return True

            rc_update()



        except:
            pass


#########################color##############################

def color(frame, h, s, v):
    hsv = cv2.cvtColor(frame[50:200, 0:640], cv2.COLOR_BGR2HSV)

    lower = np.array([h - 10, s, v], dtype=np.uint8)
    upper = np.array([h + 10, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_and(hsv, hsv, mask=mask)

    graymask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    x, graymask = cv2.threshold(graymask, 127, 255, cv2.THRESH_BINARY)
    graymask = cv2.medianBlur(graymask, 3)

    kernel = np.ones((10, 10), np.uint8)
    graymask = cv2.dilate(graymask, kernel, iterations=1)

    graymask = cv2.morphologyEx(graymask, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(graymask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # filter size 1
    hand = []
    for i in contours:
        if cv2.contourArea(i) > 2000:
            hand.append(i)

    # filter find
    chull = []
    for contour in hand:
        hull = cv2.convexHull(contour)
        chull.append(hull)
    xtotal = 0
    ytotal = 0
    flg = 0

    if len(hand) > 0 and len(chull) > 0:
        hull_points = []
        for item in chull[0]:
            point = [item[0][0], item[0][1]]
            tx = item[0][0]
            ty = item[0][1]
            xtotal = xtotal + tx
            ytotal = ytotal + ty
            hull_points.append(point)
            flg = flg + 1
        try:
            for item in chull[1]:
                tx = item[0][0]
                ty = item[0][1]
                xtotal = xtotal + tx
                ytotal = ytotal + ty
                flg = flg + 1
        except:
            cv2.drawContours(mask, chull, -1, (255, 0, 255), 3)

            cv2.imshow("color", mask)
            return False

        xfinal = int(xtotal / flg)
        yfinal = int(ytotal / flg)

        cv2.circle(mask, (xfinal, yfinal), 10, (0, 0, 255), -1)

        cv2.drawContours(mask, chull, -1, (255, 0, 255), 3)
        cv2.imshow("color", mask)
        return True
    cv2.imshow("color", mask)
    return False


################################################################
def line_err_detect(input_frame, line_err_detect_threshold):
    line_frame = cv2.resize(input_frame, (64, 48))
    line_frame = cv2.cvtColor(line_frame, cv2.COLOR_BGR2HSV)
    lower = np.array([143, 0, 0], dtype=np.uint8)
    upper = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(line_frame, lower, upper)

    mask = cv2.bitwise_not(line_frame, line_frame, mask=mask)
    hsv = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    trash, line_frame = cv2.threshold(hsv, 45, 1, cv2.THRESH_BINARY_INV)

    # line_frame = cv2.cvtColor(line_frame, cv2.COLOR_BGR2GRAY)
    # trash, line_frame = cv2.threshold(line_frame, line_err_detect_threshold, 1, cv2.THRESH_BINARY_INV)
    cv2.imshow('Line_err', line_frame * 200)
    cv2.waitKey(10)

    # line_frame*200 for imshow
    line_err_detect_point = line_frame[47, 0:64] + line_frame[46, 0:64] + line_frame[45, 0:64] + line_frame[44,
                                                                                                 0:64] + line_frame[43,
                                                                                                         0:64]
    line_err_detect_point = line_err_detect_point / 5
    line_err_detect_err = 0
    line_err_detect_point_sum = 0
    for i in range(1, 64):
        line_err_detect_err = line_err_detect_err + (line_err_detect_point[i] * (i - 32))
        line_err_detect_point_sum = line_err_detect_point_sum + line_err_detect_point[i]
    line_err_detect_err = line_err_detect_err / (line_err_detect_point_sum + 0.001)

    line_err_detect_point_2 = line_frame[30, 0:64] + line_frame[31, 0:64] + line_frame[32, 0:64] + line_frame[33,
                                                                                                   0:64] + line_frame[
                                                                                                           34,
                                                                                                           0:64]
    line_err_detect_point_2 = line_err_detect_point_2 / 5
    line_err_detect_err_2 = 0
    line_err_detect_point_sum_2 = 0
    for i in range(1, 64):
        line_err_detect_err_2 = line_err_detect_err_2 + (line_err_detect_point_2[i] * (i - 32))
        line_err_detect_point_sum_2 = line_err_detect_point_sum_2 + line_err_detect_point_2[i]
    line_err_detect_err_2 = line_err_detect_err_2 / (line_err_detect_point_sum_2 + 0.001)

    line_err_detect_err = line_err_detect_err + line_err_detect_err_2
    line_err_detect_err = int(line_err_detect_err / 2)
    print('line err = ' + str(line_err_detect_err))
    return line_err_detect_err


##########################################################
def cut_line_detector(input_frame, line_err_detect_threshold):
    line_frame = cv2.resize(input_frame, (64, 48))
    line_frame = cv2.cvtColor(line_frame, cv2.COLOR_BGR2GRAY)
    trash, line_frame = cv2.threshold(line_frame, line_err_detect_threshold, 1, cv2.THRESH_BINARY_INV)
    cv2.imshow('Line_err', line_frame * 200)
    cv2.waitKey(10)

    # line_frame*200 for imshow
    line_err_detect_point = line_frame[47, 0:64] + line_frame[46, 0:64] + line_frame[45, 0:64] + line_frame[44,
                                                                                                 0:64] + line_frame[43,
                                                                                                         0:64]

    line_err_detect_point = line_err_detect_point / 5
    line = sum(line_err_detect_point)

    cut_line_detect_point = line_frame[10, 0:64] + line_frame[11, 0:64] + line_frame[12, 0:64] + line_frame[13,
                                                                                                 0:64] + line_frame[14,
                                                                                                         0:64]

    cut_line = sum(cut_line_detect_point)
    print("cut_line=", cut_line)
    print("line_err=", line_err_detect_point)

    if int(line) > 10 and int(cut_line) < 3:
        return True

    return False


############################################################
def line_follow(errore):
    left_right_velocit = int((errore * e_gain))
    print("velosity" + str(left_right_velocit))
    if errore > 0:
        for_back_velocity = int((32 - errore) * e_gain)
    elif errore < 0:
        for_back_velocity = int((errore + 32) * e_gain)

    else:
        for_back_velocity = int(32 * e_gain)

    # rc_update()

    if errore < 5 and errore > -5:
        return for_back_velocity, left_right_velocit
    else:
        return for_back_velocity, left_right_velocit


################################################################


def myFunc(e):
    a, b = e
    return b


def qr_scanner():
    global frame_cap
    frame = cv2.resize(frame_cap.frame, (640, 480))
    cv2.imshow('Tello', frame)
    qr_ret = False
    qr_space = 0
    qr_data = 0
    qr_Xpos = 0
    qr_Ypos = 0
    hasan = zbar.Scanner()
    qr_farame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    qr_results = hasan.scan(qr_farame)
    for result in qr_results:
        qr_ret = True
        for i in result.position:
            a_point, b_point = i
            cv2.circle(qr_farame, (a_point, b_point), 4, (0, 255, 255), -1)
        result.position.sort(key=myFunc)
        a1, b1 = result.position[0]
        a2, b2 = result.position[1]
        a3, b3 = result.position[2]
        a4, b4 = result.position[3]
        dist1 = math.sqrt((b2 - b1) ** 2 + (a2 - a1) ** 2)
        dist2 = math.sqrt((b4 - b3) ** 2 + (a4 - a3) ** 2)
        if dist1 > dist2:
            qr_space = 2
            # wall
        else:
            qr_space = 1
            # grund
        qr_Xpos = (a1 + a2 + a3 + a4) / 4
        qr_Ypos = (b1 + b2 + b3 + b4) / 4
        qr_data = result.data.decode()
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))
        print('qr detected ' + str(qr_data) + str(qr_space))





    return qr_ret, qr_data, qr_space, qr_Xpos, qr_Ypos


def qr_scanner2():
    global frame_cap
    frame = cv2.resize(frame_cap.frame, (640, 480))
    cv2.imshow('Tello', frame)
    qr_ret = False
    qr_space = 0
    qr_data = 0
    qr_Xpos = 0
    qr_Ypos = 0
    hasan = zbar.Scanner()
    qr_farame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    qr_results = hasan.scan(qr_farame)
    for result in qr_results:
        qr_ret = True
        for i in result.position:
            a_point, b_point = i
            cv2.circle(qr_farame, (a_point, b_point), 4, (0, 255, 255), -1)
        result.position.sort(key=myFunc)
        a1, b1 = result.position[0]
        a2, b2 = result.position[1]
        a3, b3 = result.position[2]
        a4, b4 = result.position[3]
        dist1 = math.sqrt((b2 - b1) ** 2 + (a2 - a1) ** 2)
        dist2 = math.sqrt((b4 - b3) ** 2 + (a4 - a3) ** 2)
        if dist1 > dist2:
            qr_space = 2
            # wall
        else:
            qr_space = 1
            # grund
        qr_Xpos = (a1 + a2 + a3 + a4) / 4
        qr_Ypos = (b1 + b2 + b3 + b4) / 4
        qr_data = result.data.decode()
        print('qr detected ' + str(qr_data) + str(qr_space))
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    return qr_ret, qr_data, qr_space, qr_Xpos, qr_Ypos


####################################################################


###########################################################333
def qr_operator(qr_space, qr_Xpos, qr_Ypos):
    if qr_Ypos > 240 and qr_space == 1:
        qr_distanse = int(((480 - qr_Ypos) / 4) + 0)
        move_value = tello.move_forward(30)
        time.sleep(qr_forward_sleep)
        print("top of qrcode")

        return True
    return False


##################################################
def jahat2daraje(jahat):
    if jahat == "N":
        return 0
    if jahat == "E":
        return 90
    if jahat == "S":
        return 180
    if jahat == "W":
        return (-90)


##################################################################
def rotate_on_qr(koja, now_front):
    print("koja=" + koja)
    print("now_front=" + now_front)
    now_deg = jahat2daraje(now_front)
    sec_deg = jahat2daraje(koja)
    rot_flag = now_deg - sec_deg
    if rot_flag == 0:
        print("stright")
    if rot_flag == 90 or rot_flag == -270:
        tello.rotate_counter_clockwise(90)
        time.sleep(rotate_sleep)
    if rot_flag == 180 or rot_flag == -180:
        tello.rotate_counter_clockwise(180)
        time.sleep(rotate_sleep * 2)
    if rot_flag == -90 or rot_flag == 270:
        tello.rotate_clockwise(90)
        time.sleep(rotate_sleep)
    now_front = koja
    global mission_number
    global last_qrcode
    print(last_qrcode)
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&&&&&&")

    
    att_control(700, False)
    return now_front


##########################################################3
def do_mission_1(frame, line_tresh, frame_cap, qr_data):
    # while not cut_line_detector(frame, line_tresh):
    #     print("Start Follow Line Function!!!!!!!!!!!!!!")
    #     error_line = line_err_detect(frame, line_tresh)
    #     print("error_line" + str(error_line))
    #     value_of_line = line_follow(error_line)
    #     # value_of_line = line_follow(line_err_detect(frame,line_tresh))
    #     global for_back_velocity
    #     for_back_velocity = int(value_of_line[0])
    #     print("for_back_velocity" + str(for_back_velocity))
    #     global left_right_velocity
    #     left_right_velocity = int(value_of_line[1])
    #     print("left_right_velocit" + str(left_right_velocity))
    #     rc_update()
    #     frame = cv2.resize(frame_cap.frame, (640, 480))
    #     print("in mission frame")
    #     cv2.imshow('Tello', frame)
    #     line_tresh = cv2.getTrackbarPos('L_tresh', 'Tello')
    global last_qrcode
    last_qrcode = qr_data
    # tello.move_forward(90)
    # time.sleep(qr_forward_sleep)
    print("mission 1 on it")
    tello.rotate_clockwise(180)
    time.sleep(rotate_sleep * 2)
    global mission_number
    mission_number += 1
    global now_front
    print("current_front in mission", now_front)
    if now_front == "N":
        now_front = "S"

    elif now_front == "S":
        now_front = "N"

    elif now_front == "E":
        now_front = "W"

    elif now_front == "W":
        now_front = "E"

    # tello.land()
    # tello.move_forward(90)
    # time.sleep(qr_forward_sleep)
    print("after mission jahat=", last_qrcode[(mission_number - 1) * 2])
    print("after mission front", now_front)
    new_front = rotate_on_qr(last_qrcode[(mission_number - 1) * 2], now_front)
    now_front = new_front
    print("after rotate in last check bar= ", new_front)
    print("nowwwwwwww", now_front)
    # tello.land()
    # time.sleep(3)
    return True


##########################################################3
def do_mission_2(frame, line_tresh, frame_cap, qr_data):
    # while not cut_line_detector(frame, line_tresh):
    #     print("Start Follow Line Function!!!!!!!!!!!!!!")
    #     error_line = line_err_detect(frame, line_tresh)
    #     print("error_line" + str(error_line))
    #     value_of_line = line_follow(error_line)
    #     # value_of_line = line_follow(line_err_detect(frame,line_tresh))
    #     global for_back_velocity
    #     for_back_velocity = int(value_of_line[0])
    #     print("for_back_velocity" + str(for_back_velocity))
    #     global left_right_velocity
    #     left_right_velocity = int(value_of_line[1])
    #     print("left_right_velocit" + str(left_right_velocity))
    #     rc_update()
    #     frame = cv2.resize(frame_cap.frame, (640, 480))
    #     print("in mission frame")
    #     cv2.imshow('Tello', frame)
    #     line_tresh = cv2.getTrackbarPos('L_tresh', 'Tello')
    global last_qrcode
    last_qrcode = qr_data
    # tello.move_forward(90)
    # time.sleep(qr_forward_sleep)
    print("mission 2 on it")
    att_control(3300, True)
    tello.move_right(60)
    time.sleep(rotate_sleep * 2)
    att_control(700, True)
    tello.move_left(60)
    time.sleep(rotate_sleep * 2)
    att_control(700, False)
    tello.rotate_clockwise(180)
    time.sleep(rotate_sleep * 2)
    global mission_number
    mission_number += 1
    global now_front
    print("current_front in mission", now_front)
    if now_front == "N":
        now_front = "S"

    elif now_front == "S":
        now_front = "N"

    elif now_front == "E":
        now_front = "W"

    elif now_front == "W":
        now_front = "E"

    # tello.land()
    # tello.move_forward(90)
    # time.sleep(qr_forward_sleep)
    print("after mission jahat=", last_qrcode[(mission_number - 1) * 2])
    print("after mission front", now_front)
    new_front = rotate_on_qr(last_qrcode[(mission_number - 1) * 2], now_front)
    now_front = new_front
    print("after rotate in last check bar= ", new_front)
    print("nowwwwwwww", now_front)
    # tello.land()
    # time.sleep(3)
    return True

##########################################################3

def do_mission_3(frame, line_tresh, frame_cap, qr_data):
    # while not cut_line_detector(frame, line_tresh):
    #     print("Start Follow Line Function!!!!!!!!!!!!!!")
    #     error_line = line_err_detect(frame, line_tresh)
    #     print("error_line" + str(error_line))
    #     value_of_line = line_follow(error_line)
    #     # value_of_line = line_follow(line_err_detect(frame,line_tresh))
    #     global for_back_velocity
    #     for_back_velocity = int(value_of_line[0])
    #     print("for_back_velocity" + str(for_back_velocity))
    #     global left_right_velocity
    #     left_right_velocity = int(value_of_line[1])
    #     print("left_right_velocit" + str(left_right_velocity))
    #     rc_update()
    #     frame = cv2.resize(frame_cap.frame, (640, 480))
    #     print("in mission frame")
    #     cv2.imshow('Tello', frame)
    #     line_tresh = cv2.getTrackbarPos('L_tresh', 'Tello')
    global last_qrcode
    last_qrcode = qr_data
    # tello.move_forward(90)
    # time.sleep(qr_forward_sleep)
    print("mission 3 on it")
    att_control(2700, True)
    # tello.move_right(80)
    #
    # time.sleep(rotate_sleep * 2)
    #
    # tello.rotate_counter_clockwise(90)
    # time.sleep(rotate_sleep * 2)
    # tello.move_right(40)
    #
    # time.sleep(rotate_sleep * 2)
    #
    # att_control(1700, True)
    # tello.move_left(40)
    #
    # time.sleep(rotate_sleep * 2)
    #
    # tello.move_forward(160)
    # time.sleep(rotate_sleep * 2)
    #
    # tello.rotate_counter_clockwise(180)
    # time.sleep(rotate_sleep * 2)
    # tello.move_left(40)
    #
    # time.sleep(rotate_sleep * 2)
    # att_control(2700, True)
    # att_control(700, True)
    # tello.move_right(40)
    #
    # time.sleep(rotate_sleep * 2)
    #
    # tello.move_forward(80)
    # time.sleep(rotate_sleep * 2)
    #
    # tello.rotate_counter_clockwise(90)
    # time.sleep(rotate_sleep * 2)
    #

    att_control(700, True)
    tello.rotate_clockwise(180)
    time.sleep(rotate_sleep * 2)
    global mission_number
    mission_number += 1
    global now_front
    print("current_front in mission", now_front)
    if now_front == "N":
        now_front = "S"

    elif now_front == "S":
        now_front = "N"

    elif now_front == "E":
        now_front = "W"

    elif now_front == "W":
        now_front = "E"

    # tello.land()
    # tello.move_forward(90)
    # time.sleep(qr_forward_sleep)
    print("after mission jahat=", last_qrcode[(mission_number - 1) * 2])
    print("after mission front", now_front)
    new_front = rotate_on_qr(last_qrcode[(mission_number - 1) * 2], now_front)
    now_front = new_front
    print("after rotate in last check bar= ", new_front)
    print("nowwwwwwww", now_front)
    # tello.land()
    # time.sleep(3)
    return True


# def do_mission_3():
#     return True
# while not cut_line_detector(frame, line_tresh):
#     print("Start Follow Line Function!!!!!!!!!!!!!!")
#     error_line = line_err_detect(frame, line_tresh)
#     print("error_line" + str(error_line))
#     value_of_line = line_follow(error_line)
#     # value_of_line = line_follow(line_err_detect(frame,line_tresh))
#     global for_back_velocity
#     for_back_velocity = int(value_of_line[0])
#     print("for_back_velocity" + str(for_back_velocity))
#     global left_right_velocity
#     left_right_velocity = int(value_of_line[1])
#     print("left_right_velocit" + str(left_right_velocity))
#     rc_update()
#     frame = cv2.resize(frame_cap.frame, (640, 480))
#     print("in mission frame")
#     cv2.imshow('Tello', frame)
#     line_tresh = cv2.getTrackbarPos('L_tresh', 'Tello')


##########################################################3
def do_mission_4(frame, line_tresh, frame_cap, qr_data):
    # while not cut_line_detector(frame, line_tresh):
    #     print("Start Follow Line Function!!!!!!!!!!!!!!")
    #     error_line = line_err_detect(frame, line_tresh)
    #     print("error_line" + str(error_line))
    #     value_of_line = line_follow(error_line)
    #     # value_of_line = line_follow(line_err_detect(frame,line_tresh))
    #     global for_back_velocity
    #     for_back_velocity = int(value_of_line[0])
    #     print("for_back_velocity" + str(for_back_velocity))
    #     global left_right_velocity
    #     left_right_velocity = int(value_of_line[1])
    #     print("left_right_velocit" + str(left_right_velocity))
    #     rc_update()
    #     frame = cv2.resize(frame_cap.frame, (640, 480))
    #     print("in mission frame")
    #     cv2.imshow('Tello', frame)
    #     line_tresh = cv2.getTrackbarPos('L_tresh', 'Tello')
    global last_qrcode
    last_qrcode = qr_data
    # tello.move_forward(90)
    # time.sleep(qr_forward_sleep)

    # tello.land()
    tello.move_forward(90)
    time.sleep(qr_forward_sleep)
    tello.land()
    time.sleep(5)
    return True


####################################################################
def mission_operation(qr_data, frame, line_tresh, frame_cap):
    print("current_mission=" + str(mission_number))
    global now_front
    print("mission_number=" + str(qr_data[8]))
    if not int(qr_data[8]) == mission_number:
        now = rotate_on_qr(qr_data[(mission_number - 1) * 2], now_front)
        now_front = now
        return now
    else:
        print("misiion not equal")
        new_front = rotate_on_qr(qr_data[(mission_number - 1) * 2], now_front)
        now_front = new_front
        if mission_number == 1:
            do_mission_1(frame, line_tresh, frame_cap, qr_data)
            print("new mission" + str(mission_number))
            print("***********************")
            print("***********************")
            print("***********************")
            print("***********************")
            print("***********************")
            print("***********************")
            print("***********************")
            print("***********************")
            print("***********************")
            print("***********************")
            print("***********************")
            print("***********************")
            print("***********************")
            print("***********************")
            print("***********************")
            print("***********************")
            print("***********************")
            print("now_front==========", now_front)
            return now_front
        elif mission_number == 2:
            do_mission_2(frame, line_tresh, frame_cap, qr_data)

        elif mission_number == 3:
            do_mission_3(frame, line_tresh, frame_cap, qr_data)

        elif mission_number == 4:
            print("in mission 4")
            do_mission_4(frame, line_tresh, frame_cap, qr_data)

        return now_front


####################################################################

tello = Tello()

if not tello.connect():
    print("Tello not connected")

if not tello.set_speed(speed):
    print("Not set speed to lowest possible")

    # In case streaming is on. This happens when we quit this program without the escape key.
if not tello.streamoff():
    print("Could not stop video stream")

if not tello.streamon():
    print("Could not start video stream")

frame_cap = tello.get_frame_read()
###############################################################################
tello.takeoff()
time.sleep(3)
print(att_control(1000, False))
# print(att_control(2000, False))

# tello.go_xyz_speed(50, 50, 50, 50, 50, 50, 20)

# tello.rotate_counter_clockwise(90)
# time.sleep(rotate_sleep)
# tello.move_back(100)
# time.sleep(5)
# tello.move_right(100)
# time.sleep(4)
#
#
#
# tello.rotate_counter_clockwise(90)
# time.sleep(rotate_sleep)
# tello.move_back(100)
# time.sleep(5)
# tello.move_right(100)
# time.sleep(4)
#
#
#
# tello.rotate_counter_clockwise(90)
# time.sleep(rotate_sleep)
# tello.move_back(100)
# time.sleep(5)
# tello.move_right(100)
# time.sleep(4)
#
#
# tello.rotate_counter_clockwise(90)
# time.sleep(rotate_sleep)
# tello.move_back(100)
# time.sleep(5)
# tello.move_right(100)
# time.sleep(4)
# # print(att_control(2400, False))
# tello.land()
# time.sleep(4)

while True:
    print("first_step in while!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    frame = cv2.resize(frame_cap.frame, (640, 480))
    print("00000000000000000000000000000000000000000000")
    cv2.imshow('Tello', frame)
    line_tresh = cv2.getTrackbarPos('L_tresh', 'Tello')

    print(cv2.getTrackbarPos('tresh', 'Tello'))

    qr_scanner_value = qr_scanner()
    if (qr_scanner_value[0]):
        last_qrcode = qr_scanner_value[1]
        for_belocity = 0
        left_right_velocity = 0
        up_down_velocity = 0
        yaw_velocity = 0
        rc_update()
        print("RET is True")
        if qr_operator(qr_scanner_value[2], qr_scanner_value[3], qr_scanner_value[4]):

            value_new_of_front = mission_operation(qr_scanner_value[1], frame, line_tresh, frame_cap)
            now_front = value_new_of_front
            print("now_front from func" + now_front)

            print("now_front global" + now_front)

        else:
            print("Start Follow Line Function!!!!!!!!!!!!!!")
            error_line = line_err_detect(frame, line_tresh)
            print("error_line" + str(error_line))
            value_of_line = line_follow(error_line)
            # value_of_line = line_follow(line_err_detect(frame,line_tresh))
            for_back_velocity = int(value_of_line[0])
            print("for_back_velocity" + str(for_back_velocity))

            left_right_velocity = int(value_of_line[1])
            print("left_right_velocit" + str(left_right_velocity))


    else:
        print("RET is False")
        hy = cv2.getTrackbarPos('H_Yellow', 'Tello')
        sy = cv2.getTrackbarPos('S_Yellow', 'Tello')
        vy = cv2.getTrackbarPos('V_Yellow', 'Tello')

        # if color(frame, hy, sy, vy):
        #     stop()
        #     print("got it is Gate")
        #     att_control(700, False)
        #     tello.move_forward(90)
        #     print("move forward")
        #     time.sleep(qr_forward_sleep)
        #     att_control(1000, False)

        print("Start Follow Line Function!!!!!!!!!!!!!!")
        error_line = line_err_detect(frame, line_tresh)
        print("error_line" + str(error_line))
        value_of_line = line_follow(error_line)
        # value_of_line = line_follow(line_err_detect(frame,line_tresh))
        for_back_velocity = int(value_of_line[0])
        print("for_back_velocity" + str(for_back_velocity))

        left_right_velocity = int(value_of_line[1])
        print("left_right_velocit" + str(left_right_velocity))

    key = cv2.waitKey(1)

    if key == 27:
        tello.land()
        time.sleep(5)  # escape key

    elif key == 117:  # u key
        up_down_velocity = 50

    elif key == 106:  # j key
        up_down_velocity = -50

    elif key == 119:  # w key
        for_back_velocity = 50

    elif key == 115:  # s key
        for_back_velocity = -50

    elif key == 100:  # d key
        left_right_velocity = 50

    elif key == 97:  # a key
        left_right_velocity = -50

    elif key == 101:  # e key
        yaw_velocity = -50

    elif key == 113:  # q key
        yaw_velocity = 50

    elif key == 102:

        for_belocity = 0
        left_right_velocity = 0
        up_down_velocity = 0
        yaw_velocity = 0

    rc_update()
