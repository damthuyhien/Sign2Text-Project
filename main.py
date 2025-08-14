import torch
import cv2
import mediapipe as mp
import numpy as np
import time
from model import SignSequenceNet
from utils import map_sign_to_letter, correct_sentence, speak_text  # giữ nguyên utils tiếng Anh

# --- Setup model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SignSequenceNet(num_classes=29).to(device)
model.load_state_dict(torch.load('saved_model.pth', map_location=device))
model.eval()

# --- Mediapipe hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# --- Video capture ---
cap = cv2.VideoCapture(0)
current_sentence = ""

# --- Buffer để ổn định nhận diện ---
prev_letter = ""
frame_buffer = []
required_frames = 7  # số frame liên tiếp cùng ký hiệu
prev_confirmed_letter = ""
last_update_time = time.time()
update_interval = 0.5  # giây giữa 2 ký hiệu

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lật camera ngang
    frame = cv2.flip(frame, 1)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    letter = ""
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

        # --- Hiển thị ROI để debug ---
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0), 2)

        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size != 0:
            roi_resized = cv2.resize(roi, (28, 28))
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
            roi_tensor = torch.tensor(roi_rgb / 255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(roi_tensor)
                pred = torch.argmax(outputs, dim=1).item()
                letter = map_sign_to_letter(pred)  # dùng ký hiệu tiếng Anh

    # --- Thêm vào buffer để ổn định ---
    if letter not in ["", "nothing"]:
        frame_buffer.append(letter)
        if len(frame_buffer) > required_frames:
            frame_buffer.pop(0)

        if len(frame_buffer) == required_frames and all(l == frame_buffer[0] for l in frame_buffer):
            confirmed_letter = frame_buffer[0]

            # Kiểm tra thời gian giữa các ký hiệu
            if confirmed_letter != prev_confirmed_letter and time.time() - last_update_time > update_interval:
                current_sentence += confirmed_letter
                current_sentence = correct_sentence(current_sentence)  # vẫn dùng English correction
                speak_text(current_sentence)
                prev_confirmed_letter = confirmed_letter
                last_update_time = time.time()
    else:
        frame_buffer.clear()

    # --- Hiển thị thông tin debug ---
    cv2.putText(frame, f"Sentence: {current_sentence}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if letter != "":
        cv2.putText(frame, f"Current letter: {letter}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow('Sign2Text Debug', frame)

    # ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
