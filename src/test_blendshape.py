"""临时探针（W2 路线 B）：验证 FaceLandmarker 的 52 个 Blendshape 输出。

只用于环境验证，验证通过后可删除。
"""

import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

MODEL = Path("models/face_landmarker.task")
VIDEO = Path("data/raw/samples/adult_frontal.mp4")
FRAME_INDEX = 54  # 该样本的微笑峰值帧（W1 曲线上 mar=0.343）

# 见 KI-003：close()/析构在 Windows 上会阻塞约 42 秒，用全局引用保活。
_KEEPALIVE: list = []


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"无法读取第 {FRAME_INDEX} 帧")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    timestamp_ms = int(round(FRAME_INDEX * 1000 / fps))

    # 不使用 with / close()：mediapipe 1.2.2 在 Windows 上 close() 会阻塞约 42 秒（KI-003）。
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    _KEEPALIVE.append(landmarker)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    print(f"frame={FRAME_INDEX}, fps={fps:.2f}, ts={timestamp_ms}ms")
    print(f"faces detected: {len(result.face_landmarks)}")

    if not result.face_blendshapes:
        print("NO BLENDSHAPES RETURNED")
        return

    shapes = result.face_blendshapes[0]
    print(f"blendshape count: {len(shapes)}")
    print("--- all (model order) ---")
    for item in shapes:
        print(f"{item.category_name:24s} {item.score:.4f}")

    print("--- top 10 ---")
    for item in sorted(shapes, key=lambda c: c.score, reverse=True)[:10]:
        print(f"{item.category_name:24s} {item.score:.4f}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
