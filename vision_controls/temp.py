import cv2 

d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
print(d.markerSize, len(d.bytesList)) 