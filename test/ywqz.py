# From Python
# It requires OpenCV installed for Python
import math
import sys
import time

import cv2
import os
from sys import platform
import argparse

import numpy
from PIL import Image, ImageDraw, ImageFont

count = []
num = 0
index = 0

#判断完成次数
def find01(a=[]):
    global num
    global index
    print("index:" + str(index))
    print("length a: " + str(len(a)))
    if (len(a) >= 2):
        for i in range(index, len(a) - 1, 1):
            if (a[i] == 0 and a[i + 1] == 1):
                num += 1
                index += 2
                break
def pushupCount(angle):
    global count
    flag = 2
    if (angle < 50):
        flag = 1
    if (angle > 130):
        flag = 0
    if ((len(count) == 0 or flag != count[-1]) and flag != 2):
        count.append(flag)
    print("flag", flag)

def angle_between_points(p0, p1, p2):
    # 计算角度
    a = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
    b = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    c = (p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2
    if a * b == 0:
        return -1.0

    return math.acos((a + b - c) / math.sqrt(4 * a * b)) * 180 / math.pi

def length_between_points(p0, p1):
    # 2点之间的距离
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])
def get_angle_point(human, pos):
    # 返回各个部位的关键点
    pnts = []

    if pos == 'left_elbow':
        pos_list = (5, 6, 7)
    elif pos == 'left_hand':
        pos_list = (1, 5, 7)
    elif pos == 'left_knee':
        pos_list = (12, 13, 14)
    elif pos == 'left_ankle':
        pos_list = (5, 12, 14)
    elif pos == 'right_elbow':
        pos_list = (2, 3, 4)
    elif pos == 'right_hand':
        pos_list = (1, 2, 4)
    elif pos == 'right_knee':
        pos_list = (9, 10, 11)
    elif pos == 'right_ankle':
        pos_list = (2, 9, 11)
    elif pos == '1810':
        pos_list = (1, 8, 10)
    else:
        print('Unknown  [%s]', pos)
        return pnts

    for i in range(3):
        if human[0][pos_list[i]][2] <= 0.1:
            print('component [%d] incomplete' % (pos_list[i]))
            return pnts

        pnts.append((int(human[0][pos_list[i]][0]), int(human[0][pos_list[i]][1])))
    return pnts

def angle_1810(human):
    pnts = get_angle_point(human, '1810')
    if len(pnts) != 3:
        print('component incomplete')
        return -1

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('1 8 10 angle:%f' % (angle))
        pushupCount(angle)
    return angle

def angle_left_hand(human):
    pnts = get_angle_point(human, 'left_hand')
    if len(pnts) != 3:
        print('component incomplete')
        return -1

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('left hand angle:%f' % (angle))
    return angle
def angle_left_elbow(human):
    pnts = get_angle_point(human, 'left_elbow')
    if len(pnts) != 3:
        print('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('left elbow angle:%f' % (angle))
        pushupCount(angle)
    return angle

def angle_left_knee(human):
    pnts = get_angle_point(human, 'left_knee')
    if len(pnts) != 3:
        print('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('left knee angle:%f' % (angle))
    return angle

def angle_left_ankle(human):
    pnts = get_angle_point(human, 'left_ankle')
    if len(pnts) != 3:
        print('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('left ankle angle:%f' % (angle))
    return angle

def angle_right_hand(human):
    pnts = get_angle_point(human, 'right_hand')
    if len(pnts) != 3:
        print('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('right hand angle:%f' % (angle))
    return angle

def angle_right_elbow(human):
    pnts = get_angle_point(human, 'right_elbow')
    if len(pnts) != 3:
        print('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('right elbow angle:%f' % (angle))
        pushupCount(angle)
    return angle

def angle_right_knee(human):
    pnts = get_angle_point(human, 'right_knee')
    if len(pnts) != 3:
        print('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('right knee angle:%f' % (angle))
    return angle

def angle_right_ankle(human):
    pnts = get_angle_point(human, 'right_ankle')
    if len(pnts) != 3:
        print('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('right ankle angle:%f' % (angle))
    return angle


try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('D:\GraduateDesign\openpose\\build\\bin');
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--video_path", default="", help="Process video. Read all standard formats (mp4).")
    parser.add_argument("-o", "--output_video_path", default="", help="Output video path.")
    args = parser.parse_args()

    videoPath = args.video_path
    outputVideoPath = args.output_video_path

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "D:\GraduateDesign\openpose\models"
    params["hand"] = False
    params["number_people_max"] = 1
    params["disable_blending"] = False  # for black background
    # params["display"] = 0

    # # Add others in path?
    # for i in range(0, len(args[1])):
    #     curr_item = args[1][i]
    #     if i != len(args[1]) - 1:
    #         next_item = args[1][i + 1]
    #     else:
    #         next_item = "1"
    #     if "--" in curr_item and "--" in next_item:
    #         key = curr_item.replace('-', '')
    #         if key not in params:  params[key] = "1"
    #     elif "--" in curr_item and "--" not in next_item:
    #         key = curr_item.replace('-', '')
    #         if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    #cap = cv2.VideoCapture("D:\Videos\\ywqz.mp4")
    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Total frames in this video: ' + str(framecount))
    # videoWriter = cv2.VideoWriter("output/1.avi", cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)
    videoWriter = cv2.VideoWriter(outputVideoPath, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)

    hasFrame, frame = cap.read()
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    # 返回当前时间戳
    start_time = time.time()
    counter = 0

    while cap.isOpened():
        hasFrame, frame = cap.read()
        if hasFrame:
            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            # print("Body keypoints: \n" + str(datum.poseKeypoints))
            angle_1810(datum.poseKeypoints)
            print(count)
            find01(count)
            print("num: " + str(num))
            opframe = datum.cvOutputData

            counter += 1  # 计算帧数
            fps = counter / (time.time() - start_time)

            img_PIL = Image.fromarray(cv2.cvtColor(opframe, cv2.COLOR_BGR2RGB))  # 图像从OpenCV格式转换成PIL格式
            font1 = ImageFont.truetype('arial.ttf', 36)  # 40为字体大小，根据需要调整
            fillColor = (255, 0, 0)
            position1 = (100, 120)  # 第一个数值是距左，第二个数值是距上
            str1 = "num: " + str(num)
            draw = ImageDraw.Draw(img_PIL)
            draw.text(position1, str1, font=font1, fill=fillColor)
            opframe = cv2.cvtColor(numpy.asarray(img_PIL), cv2.COLOR_RGB2BGR)  # 转换回OpenCV格式
            cv2.namedWindow("main", 0)  # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
            cv2.imshow("main", opframe)
            videoWriter.write(opframe)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(e)
    sys.exit(-1)
