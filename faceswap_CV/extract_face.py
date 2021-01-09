import os
import sys
import uuid
sys.path.append(os.path.expanduser("./face_detection/s3fd"))

import matplotlib
matplotlib.use("agg")
import cv2

from detector import Detector


model_file = "./face_detection/face_detection.pth.tar"
detector = Detector(model_file)#获取模型检测人脸，使用faster-R-CNN检测实现


def extract(video_path, face_folder):
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)# 5 //帧率
    length = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))# 7 //视频帧数
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),# 3 //帧宽
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))# 4 //帧高度

    if not os.path.exists(face_folder):
        os.mkdir(face_folder)

    print("begin extracting...")
    success, frame = videoCapture.read()# success是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图
    frame = np.rot90(frame,-1)#读取图片顺时针旋转90°
    freq = length // 500 #// 先做除法(/),然后向下取整(floor)，最终freq是总共提取的人脸数

    count = 0
    while success:
        raw_frame = frame.copy()       
        bboxes = detector.infer(frame)#获得检测到人脸的边框
        if count % freq == 0 and bboxes is not None and len(bboxes) == 1:# 只有是freq的整数倍时，才继续
            ymin, xmin, ymax, xmax, _, _ = bboxes[0]#每个bounding box的左上角和右下角的座标，形如（Y_min,X_min, Y_max,X_max）,第Y行，第X列
            face_image = raw_frame[int(ymin):int(ymax), int(xmin):int(xmax)]#将检测到的人脸裁剪出来
            random_name = str(uuid.uuid4()) + ".jpg" # 生成每张图片的名字
            face_image_path = os.path.join(face_folder, random_name)# 获取图片路径
            cv2.imwrite(face_image_path, face_image) #将图片写入
        success, frame = videoCapture.read() #接着逐帧读取图片
        count += 1


if __name__ == "__main__":
    video_A = ["./videos/face_A.flv"]
    video_B = ["./videos/face_B.flv"]

    for video_path in video_A:
        extract(video_path, "./face_A")
    for video_path in video_B:
        extract(video_path, "./face_B")
