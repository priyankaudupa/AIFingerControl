import cv2 as cv
import mediapipe
import numpy as np
import pyautogui

mphands = mediapipe.solutions.hands
detector = mphands.Hands()
draw = mediapipe.solutions.drawing_utils

# Set the desired frame size
frame_width = 2560
frame_height = 1440

wscreen, hscreen = pyautogui.size()
px, py = 0, 0
cx, cy = 0, 0

# Smoothing factor (adjust as needed)
smoothing_factor = 0.75

def handLandmarks(img, frame):
    landmarkList = []
    landmarkPositions = detector.process(img)

    landmarkCheck = landmarkPositions.multi_hand_landmarks

    if landmarkCheck:
        for hand in landmarkCheck:
            for i, landmark in enumerate(hand.landmark):
                draw.draw_landmarks(frame, hand, mphands.HAND_CONNECTIONS)
                h, w, c = img.shape
                centerX, centerY = int(landmark.x * w), int(landmark.y * h)
                landmarkList.append([i, centerX, centerY])
    return landmarkList, frame

def fingers(landmarks):
    fingerTips = []
    tipIds = [4, 8, 12, 16, 20]

    if landmarks[4][1] > landmarks[3][1]:
        fingerTips.append(1)
    else:
        fingerTips.append(0)

    for i in range(1, 5):
        if landmarks[tipIds[i]][2] < landmarks[tipIds[i] - 3][2]:
            fingerTips.append(1)
        else:
            fingerTips.append(0)
    return fingerTips

cap = cv.VideoCapture(0)
# Set the frame size
cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        rgbFrame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        lmList, frame = handLandmarks(rgbFrame, frame)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            finger = fingers(lmList)
            
            if finger[4] == 1:  
                right_click_active = True
                pyautogui.rightClick()

            # Check for fist (any finger closed)
            if any(finger[i] == 0 for i in range(1, 5)) and right_click_active:
                right_click_active = False

            if finger[1] == 1 and finger[2] == 0:
                x3 = np.interp(x1, (0, int(cap.get(cv.CAP_PROP_FRAME_WIDTH))), (0, wscreen))
                y3 = np.interp(y1, (0, int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))), (0, hscreen))

                cx = px + (x3 - px) * smoothing_factor
                cy = py + (y3 - py) * smoothing_factor

                if (abs(cx - px) + abs(cy - py)) > 5:
                    pyautogui.moveTo(wscreen - cx, cy)
                px, py = cx, cy

            if finger[1] == 1 and finger[2] == 1:
                pyautogui.click()

            if finger[0] == 1:
                pyautogui.scroll(50)

            if all(finger[i] == 0 for i in [0,1, 2, 3, 4]):
                pyautogui.scroll(-50)

        cv.imshow('frame', frame)

        # Check for key events
        key = cv.waitKey(1)
        if key == 27:  # ASCII value for Escape key
            break
        elif key == ord('c') or key == ord('C'):  # ASCII value for 'C' key
            # Add any cleanup or cancellation logic here
            break

cap.release()
cv.destroyAllWindows()
