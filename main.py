import ffmpeg
import numpy as np
import cv2
from deepface import DeepFace
import threading
import queue
import time

# 全局变量，用于存储情绪识别结果
emotion_results = queue.Queue()  # 线程安全的队列
current_emotion = None  # 当前帧的情绪结果
last_emotion_time = 0  # 上一次情绪识别的时间


def analyze_emotion(frame):
    """
    情绪识别函数，运行在单独的线程中
    """
    global emotion_results
    try:
        # 使用 DeepFace 进行情绪识别
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        for result in results:
            # 获取情绪分析结果
            emotion = result['dominant_emotion']  # 主要情绪
            emotions = result['emotion']  # 所有情绪的概率值
            face_region = result['region']  # 人脸区域
            # 将结果放入队列
            emotion_results.put((emotion, emotions, face_region))
    except Exception as e:
        print(f"DeepFace 分析失败: {e}")


def main(source):
    args = {
        "fflags": "nobuffer",
        "flags": "low_delay"
    }  # 添加参数
    probe = ffmpeg.probe(source)
    cap_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
    print("fps: {}".format(cap_info['r_frame_rate']))
    width = cap_info['width']  # 获取视频流的宽度
    height = cap_info['height']  # 获取视频流的高度
    up, down = str(cap_info['r_frame_rate']).split('/')
    fps = eval(up) / eval(down)
    print("fps: {}".format(fps))  # 读取可能会出错错误
    process1 = (
        ffmpeg
        .input(source, **args)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .overwrite_output()
        .run_async(pipe_stdout=True)
    )

    global current_emotion, last_emotion_time

    while True:
        in_bytes = process1.stdout.read(width * height * 3)  # 读取图片
        if not in_bytes:
            break
        # 转成ndarray
        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
        frame = cv2.cvtColor(in_frame, cv2.COLOR_RGB2BGR)  # 转成BGR

        # 每隔 10 秒进行一次情绪识别
        current_time = time.time()
        if current_time - last_emotion_time >= 4:  # 10 秒后启动新的情绪识别
            # 启动情绪识别线程
            threading.Thread(target=analyze_emotion, args=(frame.copy(),)).start()
            last_emotion_time = current_time  # 更新上一次情绪识别的时间

        # 从队列中获取最新的情绪识别结果
        if not emotion_results.empty():
            current_emotion, emotions, face_region = emotion_results.get()

        # 如果当前有情绪识别结果，则显示
        if current_emotion is not None:
            x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
            # 在图像上绘制情绪结果
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {current_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 显示所有情绪的概率值
            y_offset = y + h + 20  # 从人脸区域下方开始显示
            for emotion_name, emotion_value in emotions.items():
                text = f"{emotion_name}: {emotion_value:.2f}"
                cv2.putText(frame, text, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30  # 每行间隔 30 像素

        # 显示帧
        cv2.imshow("ffmpeg", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    process1.kill()  # 关闭


if __name__ == "__main__":

    alhua_rtsp = f"rtsp://admin:Admin123456@192.168.254.4:554/Streaming/Channels/101"

    main(alhua_rtsp)