from os.path import isfile
import cv2
import mediapipe as mp
import os
import pandas as pd
import openpyxl
import argparse

parser = argparse.ArgumentParser(description='Run Mediapipe pose estimation and record kps coordinates ')
parser.add_argument('-id','--id', help='Subject ID (Folder name from /Data_Image path)', required=True)
args = vars(parser.parse_args())

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
kps_result = pd.DataFrame([], columns=['idx', 'image_name', 'nose_x', 'nose_y', 'right_shoulder_x', 'right_shoulder_y',
                                       'left_shoulder_x', 'left_shoulder_y', 'right_elbow_x', 'right_elbow_y',
                                       'left_elbow_x', 'left_elbow_y', 'right_wrist_x', 'right_wrist_y', 'left_wrist_x',
                                       'left_wrist_y', 'right_hip_x', 'right_hip_y', 'left_hip_x', 'left_hip_y',
                                       'right_knee_x', 'right_knee_y', 'left_knee_x', 'left_knee_y', 'right_ankle_x',
                                       'right_ankle_y', 'left_ankle_x', 'left_ankle_y'])

############################################### For static image input #################################################
# configs and path directory
src_folder = 'D:/SmartRehab/Data_Image/' + args['id'] +'/'
# SUBJECT_ID = sorted([sub_num for sub_num in os.listdir(src_folder)])
# Subject_num = 1                                                      # Decide to perform the loop on which subject here
IMAGE_FILES = sorted([img_name for img_name in os.listdir(src_folder) if isfile(os.path.join(src_folder, img_name))])
annotated_output_path = 'D:/SmartRehab/Data_Image/' + args['id'] + '/annotated/'
kps_output_path = 'D:/SmartRehab/Data_Keypoints/' + args['id'] + '_phone(mediapipe)_kps.xlsx'

if not os.path.exists(annotated_output_path):
    os.makedirs(annotated_output_path)

print(f'========== Annotating subject: [' + args['id'] + '] ==========')
with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5) as pose:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(os.path.join(src_folder + file))
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            continue
        print(
            f'Processing image: ('
            f'{IMAGE_FILES[idx]}) |'
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
        )
        # Write landmarks coordinate
        kp0x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width
        kp0y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height
        kp1x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width
        kp1y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_height
        kp2x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width
        kp2y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height
        kp3x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x * image_width
        kp3y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y * image_height
        kp4x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x * image_width
        kp4y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y * image_height
        kp5x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x * image_width
        kp5y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y * image_height
        kp6x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x * image_width
        kp6y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y * image_height
        kp7x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x * image_width
        kp7y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y * image_height
        kp8x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x * image_width
        kp8y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y * image_height
        kp9x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].x * image_width
        kp9y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].y * image_height
        kp10x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].x * image_width
        kp10y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].y * image_height
        kp11x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE].x * image_width
        kp11y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE].y * image_height
        kp12x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE].x * image_width
        kp12y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE].y * image_height

        temp = pd.DataFrame(
            [[idx, IMAGE_FILES[idx], kp0x, kp0y, kp1x, kp1y, kp2x, kp2y, kp3x, kp3y, kp4x, kp4y, kp5x, kp5y, kp6x, kp6y,
              kp7x, kp7y, kp8x, kp8y, kp9x, kp9y, kp10x, kp10y, kp11x, kp11y, kp12x, kp12y]],
            columns=['idx', 'image_name', 'nose_x', 'nose_y', 'right_shoulder_x', 'right_shoulder_y',
                     'left_shoulder_x', 'left_shoulder_y', 'right_elbow_x', 'right_elbow_y',
                     'left_elbow_x', 'left_elbow_y', 'right_wrist_x', 'right_wrist_y', 'left_wrist_x',
                     'left_wrist_y', 'right_hip_x', 'right_hip_y', 'left_hip_x', 'left_hip_y',
                     'right_knee_x', 'right_knee_y', 'left_knee_x', 'left_knee_y', 'right_ankle_x',
                     'right_ankle_y', 'left_ankle_x', 'left_ankle_y'])

        kps_result = kps_result.append(temp)

        # Draw pose landmarks on the image.
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite(annotated_output_path + str(idx) + '.jpg', annotated_image)
        # Plot pose world landmarks.
        # mp_drawing.draw_landmarks(
        #    results.pose_landmarks.landmark, mp_pose.POSE_CONNECTIONS)

############################################### For webcam/video input #################################################
# cap = cv2.VideoCapture(0)
# with mp_pose.Pose(
#    min_detection_confidence=0.5,
#    min_tracking_confidence=0.5) as pose:
#  while cap.isOpened():
#    success, image = cap.read()
#    if not success:
#      print("Ignoring empty camera frame.")
#      # If loading a video, use 'break' instead of 'continue'.
#      break

#    # Flip the image horizontally for a later selfie-view display, and convert
#    # the BGR image to RGB.
#    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#    # To improve performance, optionally mark the image as not writeable to
#    # pass by reference.
#    image.flags.writeable = False
#    results = pose.process(image)

#    # Draw the pose annotation on the image.
#    image.flags.writeable = True
#    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#    mp_drawing.draw_landmarks(
#        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#    cv2.imshow('MediaPipe Pose', image)
#    if cv2.waitKey(5) & 0xFF == 27:
#      break
# cap.release()


################################################### For kps output #####################################################
# create excel writer object
writer = pd.ExcelWriter(kps_output_path)
# write dataframe to excel
kps_result.to_excel(writer)
# save the excel
writer.save()
print('DataFrame is written successfully to ' + kps_output_path)
