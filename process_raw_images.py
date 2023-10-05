import os
import cv2
import numpy as np

images_names = os.listdir("/Users/artemmoroz/Desktop/CIIRC_projects/PcbAnnomalyRecognition/nova")
counter = 0
for image_name in images_names:
    image_paded = np.zeros(shape=(500, 500, 3), dtype=np.uint8)
    image = cv2.imread(f"/Users/artemmoroz/Desktop/CIIRC_projects/PcbAnnomalyRecognition/nova/{image_name}")
    image_paded[34:-34, ...] = image
    cv2.imwrite(f"/Users/artemmoroz/Desktop/CIIRC_projects/PcbAnnomalyRecognition/ImagesNew/pcb_{counter}.png", image_paded)
    counter += 1
    #print(np.max(image_paded), np.min(image_paded))
    #cv2.imshow("image paded", image_paded)
    #cv2.waitKey(0)