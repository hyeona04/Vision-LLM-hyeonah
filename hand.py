import cv2
import json
import subprocess
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 중간 점 기준으로 각도 계산하는 함수
def calculate_angle(p1, p2, p3):
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


def speak_text(text):

    subprocess.run(
        [PIPER_PYTHON,
         "-m",
         "piper",

         "-m",
         PIPER_MODEL,

         "-f",
         OUTPUT_FILE,

         "--",
         text,
         ],
         check=True,
    )

    subprocess.run(
        [ "aplay",OUTPUT_FILE,],check=True
    )

base_option = python.BaseOptions(
    model_asset_path="src/models/MediaPipe/hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_option,
    num_hands=2
)
hand_detector = vision.HandLandmarker.create_from_options(options)
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

finger_tips = (4, 8, 12, 16, 20)
angle_threshold = 160

# 엄지와 다른 손가락 접촉 기준
touch_threshold = 40

with open("gesture_text.json","r",encoding="utf-8") as f:
    user_text=json.load(f)

touch_map = {
    8:(user_text["8"],user_text["8"]),
    12:(user_text["12"],user_text["12"]),
    16:(user_text["16"],user_text["16"]),
    20:(user_text["20"],user_text["20"]),
}

PIPER_PYTHON = ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
OUTPUT_FILE = "src/audio/response.wav"
SPEAKER_DEVICE = "plughw:3,0"

last_note = None
display_txt = ""

# 각 손가락의 각도를 계산할 landmark index
finger_angle_points = (
    (1, 2, 3), # 엄지
    (5, 6, 7), # 검지
    (9, 10, 11), # 중지
    (13, 14, 15), # 약지
    (17, 18, 19), # 소지
)

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # 프레임 높이와 너비
    h, w = frame.shape[:2]

    # 이미지 반전 및 RGB 변환
    frame = cv2.flip(frame, 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 손 탐지
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
    result = hand_detector.detect(mp_image)

    # 왼손 / 오른손 정보
    labels = [
        "Left"
        if handedness[0].category_name == "Right"
        else "Right"
        for handedness in result.handedness
    ]


    # 펼쳐진 손가락 개수 계산
    total_finger_count = 0

    for hand in result.hand_landmarks:
        for point1_idx, point2_idx, point3_idx in finger_angle_points:
            angle = calculate_angle(
                hand[point1_idx],
                hand[point2_idx],
                hand[point3_idx],
            )
            if angle >= angle_threshold:
                total_finger_count += 1

    # 화면 좌측 상단
    cv2.putText(
        frame,
        f"Hands: {len(result.hand_landmarks)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )
    cv2.putText(
        frame,
        f"Fingers: {total_finger_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # 오른손 / 왼손 표시
    handedness_text = " / ".join(labels)
    text_size = cv2.getTextSize(handedness_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = w - text_size[0] - 20

    cv2.putText(frame, handedness_text,(text_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 255, 255),2)


    # 각 손 처리
    for hand in result.hand_landmarks:
        h, w = frame.shape[:2]
        points = [(int(p.x * w), int(p.y * h)) for p in hand]

        detected_note = None
        touched_tip = None
        detected_speech = None

        # 엄지 끝 4번
        thumb_x, thumb_y = points[4]

        # 엄지와 8, 12, 16, 20 거리 계산
        for tip_idx, (display_note,speech_note) in touch_map.items():
            tip_x, tip_y = points[tip_idx]
            distance = np.hypot(
                thumb_x - tip_x,
                thumb_y - tip_y
            )

            if distance < touch_threshold:
                detected_note = display_note
                detected_speech = speech_note
                touched_tip = tip_idx
                break

        # skeleton 그리기
        for c in connections:

            cv2.line(
                frame,
                points[c.start],
                points[c.end],
                (0, 255, 0),
                2
            )

        # landmark 점 그리기
        for i, point in enumerate(points):
            if (detected_note is not None and(i == 4 or i == touched_tip)):
                color = (0, 255, 255)
            elif i in finger_tips:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            cv2.circle(frame,point,6 if i in finger_tips else 4,color,-1)

        if detected_note is not None:
            display_txt = display_note

        cv2.putText(frame,display_txt,(w // 2 - 50, 120),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 255, 255),4)

        # 음계 출력
        if detected_note is not None:
            if detected_note != last_note:
                speak_text(detected_note)
                last_note = detected_note

        else:
            last_note = None

    cv2.imshow(
        "MediaPipe Hand Detection",
        frame
    )

hand_detector.close()
cap.release()
cv2.destroyAllWindows()

