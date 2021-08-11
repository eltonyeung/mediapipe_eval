# mediapipe_eval
Extract bodily keypoints from still images from Video2img output or Video files using mediapipe pose estimation model.

**Requirements:**
os
mediapipe
cv2
pandas


**Configs:**
- line 19: src_folder -- locate source folder.
- line 23: Subject_num -- Identify subject number from the list of SUBJECT_ID in which we want to extract kps from.
- line 24: kps_output_path -- path for outputting final kps excel sheet 


**Sample input (still image)**                ------------------------------->                    **Sample output (annotated image)**

<img width="400" alt="portfolio_view" src="https://github.com/eltonyeung/mediapipe_eval/blob/main/00786.jpg">     <img width="400" alt="portfolio_view" src="https://github.com/eltonyeung/mediapipe_eval/blob/main/786.jpg">



**kps (x, y)_rounded**
- Nose (801, 629)
- R_shoulder (725, 702)
- L_shoulder (900, 742)
- R_elbow (644, 571)
- L_elbow (933, 894)
- R_wrist (609, 427)
- L_wrist (926,1030)
- R_hip (747, 1041)
- L_hip (859, 1046)
- R_knee (738, 1277)
- L_knee (877, 1271)
- R_ankle (736, 1492)
- L_ankle (864, 1494)

