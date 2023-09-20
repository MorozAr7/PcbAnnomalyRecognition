import os
import cv2
import numpy as np

sum_array_np = np.zeros(shape=(300, 300))

for image_name in os.listdir("/Users/artemmoroz/Desktop/CIIRC_projects/PcbAnnomalyRecognition/Images/CroppedImagesPositive"):
    image = cv2.imread("/Users/artemmoroz/Desktop/CIIRC_projects/PcbAnnomalyRecognition/Images/CroppedImagesPositive/" + image_name, 0)
    image = image/255
    sum_array_np += image

print(np.mean(sum_array_np)/len(os.listdir("/Users/artemmoroz/Desktop/CIIRC_projects/PcbAnnomalyRecognition/Images/CroppedImagesPositive")))
    