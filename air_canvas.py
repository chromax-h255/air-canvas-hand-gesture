import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

canvas = None
prev_x, prev_y = 0, 0
color = (255, 0, 0)
brush_size = 8

colors = {
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "eraser": (0, 0, 0)
}

color_boxes = {
    "blue": (50, 10, 100, 60),
    "red": (120, 10, 170, 60),
    "green": (190, 10, 240, 60),
    "yellow": (260, 10, 310, 60),
    "eraser": (330, 10, 380, 60)
}

cap = cv2.VideoCapture(0)

def hand_center(hand, frame_shape):
    x_list = [lm.x for lm in hand.landmark]
    y_list = [lm.y for lm in hand.landmark]
    cx = int(np.mean(x_list) * frame_shape[1])
    cy = int(np.mean(y_list) * frame_shape[0])
    return cx, cy

def index_up(hand):
    return hand.landmark[8].y < hand.landmark[6].y

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    for name, (x1, y1, x2, y2) in color_boxes.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[name], -1)
        if colors[name] == color:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        cx, cy = hand_center(hand, frame.shape)

        if cy < 70:
            for name, (x1, y1, x2, y2) in color_boxes.items():
                if x1 < cx < x2 and y1 < cy < y2:
                    color = colors[name]
            prev_x, prev_y = 0, 0
        else:
            if index_up(hand):
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = cx, cy
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), color, brush_size)
                prev_x, prev_y = cx, cy
            else:
                prev_x, prev_y = 0, 0

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = 0, 0

    frame_with_canvas = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    cv2.imshow("Air Canvas", frame_with_canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




