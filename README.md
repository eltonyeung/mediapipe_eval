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





