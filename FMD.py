import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

st.title("顔の表情変化の可視化アプリ")

uploaded_file = st.file_uploader("動画ファイルをアップロード", type=["mp4"])

threshold = st.slider("変化検出の閾値", 0.0, 5.0, 0.4, 0.1)

if uploaded_file:
    with open("input.mp4", "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture("input.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

    facial_groups = {
        "forehead": [10, 338, 297],
        "glabella": [9],
        "left_eye_outer": [130, 133, 246],
        "right_eye_outer": [359, 362, 466],
        "left_cheek": [205, 50, 101],
        "right_cheek": [425, 280, 347],
        "nose_left": [97, 2, 326],
        "nose_center": [168, 195],
        "nose_right": [429, 358, 327],
        "upper_lip": [13, 14, 15],
        "lower_lip": [17, 84, 87],
        "left_mouth_corner": [61, 40],
        "right_mouth_corner": [291, 270],
        "left_nasolabial": [50, 101, 206],
        "right_nasolabial": [280, 347, 426],
        "left_eyelid_upper": [159],
        "left_eyelid_lower": [145],
        "right_eyelid_upper": [386],
        "right_eyelid_lower": [374],
        "left_eye_ball": [468],
        "right_eye_ball": [473],
        "left_eye_corner": [33],
        "right_eye_corner": [263],
        "left_eye_tail": [133],
        "right_eye_tail": [362],
        "left_eye_vertical": [159, 145, 153, 144],
        "right_eye_vertical": [386, 374, 373, 390],
        "mouth_opening": [13, 14, 17, 87],
        "jawline": [0, 17, 18, 19, 20, 21, 22, 23, 24]
    }

    prev_coords = None
    frame_id = 0
    movement_log = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        overlay = frame.copy()
        movement = {"frame": frame_id}

        timestamp_text = f"{int(frame_id / fps):02d}s"

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            coords = np.array([(int(p.x * width), int(p.y * height)) for p in face_landmarks])

            if prev_coords is not None:
                for region, indices in facial_groups.items():
                    if all(i < len(coords) for i in indices):
                        prev = np.mean(prev_coords[indices], axis=0)
                        now = np.mean(coords[indices], axis=0)

                        if prev is not None and now is not None:
                            diff = np.linalg.norm(prev - now)
                            movement[region] = diff

                            if diff > threshold:
                                pos = tuple(np.mean(coords[indices], axis=0).astype(int))

                                # 色分け（変化の大きさに応じて）
                                if diff < 1.0:
                                    color = (0, 255, 0)      # 緑
                                elif diff < 2.0:
                                    color = (0, 165, 255)    # オレンジ
                                else:
                                    color = (0, 0, 255)      # 赤

                                # 円の大きさを半分に（半径12）
                                overlay = cv2.circle(overlay, pos, 12, color, -1)

            prev_coords = coords.copy()

        # 時刻表示（左上）
        cv2.putText(overlay, timestamp_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        out.write(blended)
        frame_id += 1
        if len(movement) > 1:
            movement_log.append(movement)

    cap.release()
    out.release()
    face_mesh.close()

    df = pd.DataFrame(movement_log)
    df["time (s)"] = df["frame"] / fps
    df["time (s)"] = df["time (s)"].astype(int)
    df["time (s)"] = pd.to_datetime(df["time (s)"], unit='s')
    df_resampled = df.set_index("time (s)").drop(columns=["frame"]).resample("1S").mean()

    def mark_large_changes(x, threshold=1.0):
        return [f"{v}*" if v > threshold else v for v in x]

    df_resampled_marked = df_resampled.apply(mark_large_changes, axis=0)

    csv_buffer = BytesIO()
    df_resampled_marked.to_csv(csv_buffer)
    csv_buffer.seek(0)

    st.download_button("変化量が大きい部分を含むデータ (CSV) をダウンロード", data=csv_buffer, file_name="movement_data_with_changes.csv", mime="text/csv")

    st.subheader("1秒ごとの平均変化量")
    fig, ax = plt.subplots(figsize=(12, 6))
    df_resampled.plot(kind='bar', ax=ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Movement")
    ax.set_title("Facial Region Movements Over Time (Bar Graph)")
    st.pyplot(fig)

    graph_buffer = BytesIO()
    fig.savefig(graph_buffer, format="png")
    graph_buffer.seek(0)

    st.download_button("グラフ画像をダウンロード", data=graph_buffer, file_name="movement_graph.png", mime="image/png")

    with open("output.mp4", "rb") as f:
        st.download_button("生成された動画をダウンロード", f, file_name="facial_changes.mp4")
