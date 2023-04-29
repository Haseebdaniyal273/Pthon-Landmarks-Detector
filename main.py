import cv2
import mediapipe as mp
import pyautogui
cap = cv2.VideoCapture(0)
hand_dectector = mp.solutions.hands.Hands() # For hand land marks detection
drawing_utils = mp.solutions.drawing_utils # For drawing and poiniting
screen_height, screen_width = pyautogui.size() # for full screen
index_y = 0
while True: # For contiuies camera run
    _,frame = cap.read() #real krrha frame
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # Frame ko colors dena
    output = hand_dectector.process(rgb_frame) #hand detect krrhe us frame m
    hands = output.multi_hand_landmarks #hand py multi landmarks le rhe
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame , hand) #idr landmarks py dots arhe
            landmarks = hand.landmark #hand k landmarks daal rhe han
            for id, landmark in enumerate (landmarks):
                x = int(landmark.x*frame_width)
                y = int(landmark.y*frame_height)
                print(x,y)
       
    cv2.imshow('Camera', frame)
    cv2.waitKey(1)