import cv2
import mediapipe as mp
import torch
from model import SignSequenceNet
from utils import map_sign_to_letter, correct_sentence, speak_text
import numpy as np

# Load model
model = SignSequenceNet()
model.load_state_dict(torch.load('saved_model.pth'))
model.eval()

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
sentence = ""
seq_len = 10
keypoint_seq = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Lấy 21 keypoints x,y,z
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            keypoint_seq.append(keypoints)

            # nếu đủ seq_len mới predict
            if len(keypoint_seq) >= seq_len:
                input_seq = torch.tensor(np.array([keypoint_seq[-seq_len:]])).float()  # [1, seq_len, 63]
                with torch.no_grad():
                    output = model(input_seq)
                    _, pred = torch.max(output, 1)
                    letter = map_sign_to_letter(pred.item())
                    sentence += letter
                    keypoint_seq = []  # reset sequence
                    # auto-correct & speak
                    corrected = correct_sentence(sentence)
                    speak_text(corrected)

    # hiển thị text
    cv2.putText(frame, sentence, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    cv2.imshow("Sign2Text Advanced", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        sentence = ""  

cap.release()
cv2.destroyAllWindows()
