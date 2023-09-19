import cv2
import os



counter = 0

coords = ((5, 575), (355, 575), (705, 575))
size = 300
path = "/Users/artemmoroz/Desktop/CIIRC_projects/PcbAnnomalyRecognition/Images/ImagesPositive/"
for image_name in os.listdir(path):
    image = cv2.imread(path + image_name)
    for x, y in coords:
        cropped = image[y:y + size, x:x + size]
        print(counter)
        cv2.imwrite("/Users/artemmoroz/Desktop/CIIRC_projects/PcbAnnomalyRecognition/Images/CroppedImagesPositive/" + f"pcb_{counter}.png", cropped)
        counter += 1
        
