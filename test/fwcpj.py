# From Python
# It requires OpenCV installed for Python
import math
import sys

import cv2
import os
from sys import platform
import argparse

import numpy
from PIL import Image, ImageDraw, ImageFont
from scipy import signal  # 滤波等

totalAngle = []
count = []
num = 0
index = 0


# 计算握距
def length_shoulder(human):
    pnts = get_length_point(human, 'shoulder')
    if len(pnts) != 2:
        print('component incomplete')
        return -1

    length = 0
    if pnts is not None:
        length = length_between_points(pnts[0], pnts[1])
        print('shoulder length:%f' % (length))
    return length
def length_hand(human):
    pnts = get_length_point(human, 'hand')
    if len(pnts) != 2:
        print('component incomplete')
        return -1

    length = 0
    if pnts is not None:
        length = length_between_points(pnts[0], pnts[1])
        print('hand length:%f' % (length))
    return length


# 判断完成次数
def find010(a=[]):
    global num
    global index
    print("index:" + str(index))
    print("length a: " + str(len(a)))
    if (len(a) >= 3):
        for i in range(index, len(a) - 2, 1):
            if (a[i] == 0 and a[i + 1] == 1 and a[i + 2] == 0):
                num += 1
                index += 2
                break

def pushupCount(angle):
    global count
    flag = 2
    if (angle < 100):
        flag = 1
    if (angle > 150):
        flag = 0
    if ((len(count) == 0 or flag != count[-1]) and flag != 2):
        count.append(flag)
    print("flag", flag)


# 计算关节角度
def angle_between_points(p0, p1, p2):
    # 计算角度
    a = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
    b = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    c = (p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2
    if a * b == 0:
        return -1.0
    return (math.acos((a + b - c) / math.sqrt(4 * a * b)) * 180 / math.pi)
    # return math.acos((a + b - c) / math.sqrt(4 * a * b)) * 180 / math.pi

# 计算关节距离
def length_between_points(p0, p1):
    # 2点之间的距离
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])

def get_length_point(human, pos):
    pnts = []
    if pos == 'shoulder':
        pos_list = (2, 5)
    elif pos == 'hand':
        pos_list = (4, 7)
    else:
        print('Unknown  [%s]', pos)
        return pnts

    for i in range(2):
        if human[0][pos_list[i]][2] <= 0.1:
            print('component [%d] incomplete' % (pos_list[i]))
            return pnts
        pnts.append((float(human[0][pos_list[i]][0]), float(human[0][pos_list[i]][1])))

    return pnts

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
    else:
        print('Unknown  [%s]', pos)
        return pnts

    for i in range(3):
        if human[0][pos_list[i]][2] <= 0.1:
            print('component [%d] incomplete' % (pos_list[i]))
            return pnts

        pnts.append((int(human[0][pos_list[i]][0]), int(human[0][pos_list[i]][1])))
    return pnts


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
    angle = 0
    pnts = get_angle_point(human, 'left_elbow')
    if len(pnts) != 3:
        print('component incomplete')
        return angle
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('left elbow angle:%f' % (angle))
        pushupCount(angle)
        totalAngle.append(angle)
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

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    cap = cv2.VideoCapture("D:\\Videos\\fwc.mp4")
    # cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Total frames in this video: ' + str(framecount))
    # videoWriter = cv2.VideoWriter("E:\Desktop\openpose-1.6.0\openpose-1.6.0\examples\output/fwc1.avi", cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)
    videoWriter = cv2.VideoWriter(outputVideoPath, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)

    flag = 1
    while cap.isOpened():
        hasFrame, frame = cap.read()
        if hasFrame:
            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            # print("Body keypoints: \n" + str(datum.poseKeypoints))
            angle = round(angle_left_elbow(datum.poseKeypoints), 2)
            ls = length_shoulder(datum.poseKeypoints)
            lh = length_hand(datum.poseKeypoints)
            percent = lh / ls
            str2 = ""
            if (percent > 1.6 and percent < 2.3):
                print("握距适中")
                str2 = "握距适中"
            elif (percent < 1.6):
                print("握距偏窄，请加宽握距")
                str2 = "握距偏窄，请加宽握距"
            elif (percent > 2.3):
                print(("握距偏宽，请减少握距"))
                str2 = "握距偏宽，请减少握距"
            print("percent: ", percent)

            str3 = ""
            if len(totalAngle) > 1:
                if ((count[-1] == 0 or len(count) == 0) and angle > 90 and totalAngle[-1] < totalAngle[-2]):
                    print("向下高度未达到，请继续向下")
                    str3 = "向下高度未达到，请继续向下"
                elif (count[-1] == 1 and angle < 90):
                    print("向下高度已达到")
                    str3 = "向下高度已达到"
                elif (count[-1] == 1 and angle < 150):
                    print("向上高度未达到，请继续向上")
                    str3 = "向上高度未达到，请继续向上"
                elif (count[-1] == 0 and angle > 150 and totalAngle[-1] > totalAngle[-2]):
                    print("向上高度已达到")
                    str3 = "向上高度已达到"
            print(count)
            find010(count)
            print("num: " + str(num))

            opframe = datum.cvOutputData
            img_PIL = Image.fromarray(cv2.cvtColor(opframe, cv2.COLOR_BGR2RGB))  # 图像从OpenCV格式转换成PIL格式
            font1 = ImageFont.truetype('msyhbd.ttc', 15, encoding="gb2312")  # 40为字体大小，根据需要调整
            fillColor = (255, 0, 0,)
            # position = (100, 120) #第一个数值是距左，第二个数值是距
            str1 = "num: " + str(num)

            draw = ImageDraw.Draw(img_PIL)
            draw.text((10, 12), str1, font=font1, fill=fillColor)
            # draw.text((100,200), str2, font=font1, fill=fillColor)
            draw.text((10, 30), "手肘角度：" + str(angle), font=font1, fill=fillColor)
            draw.text((10, 48), str3, font=font1, fill=fillColor)
            opframe = cv2.cvtColor(numpy.asarray(img_PIL), cv2.COLOR_RGB2BGR)  # 转换回OpenCV格式
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
