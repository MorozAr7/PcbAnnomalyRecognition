import cv2
import os



counter = 0

coords = ((310, 545), (620, 545))
size = 265
path = "/Users/artemmoroz/Desktop/CIIRC_projects/PcbAnnomalyRecognition/Images/ImagesNegative/"
for image_name in os.listdir(path):
    image = cv2.imread(path + image_name)
    for x, y in coords:
        cropped = image[y:y + size, x:x + size]
        cv2.imshow("image", cropped)
        cv2.waitKey(0)
        print(counter)
        cv2.imwrite("/Users/artemmoroz/Desktop/CIIRC_projects/PcbAnnomalyRecognition/Images/CroppedNegative/" + f"pcb_{counter}.png", cropped)
        counter += 1
        
