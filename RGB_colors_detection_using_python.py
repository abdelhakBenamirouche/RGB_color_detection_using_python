import numpy as np 
import cv2
  
webcam = cv2.VideoCapture(0)

while(1):
    
    _, imageFrame = webcam.read()

    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 
    
    # Range of red color and creating mask
    red_min = np.array([136, 87, 111], np.uint8) 
    red_max = np.array([180, 255, 255], np.uint8) 
    red_mask = cv2.inRange(hsvFrame, red_min, red_max) 
    
    # Range of green color and creating mask
    green_min = np.array([25, 52, 72], np.uint8) 
    green_max = np.array([102, 255, 255], np.uint8) 
    green_mask = cv2.inRange(hsvFrame, green_min, green_max) 

    # Range of blue color and creating mask
    blue_min = np.array([94, 80, 2], np.uint8) 
    blue_max = np.array([120, 255, 255], np.uint8) 
    blue_mask = cv2.inRange(hsvFrame, blue_min, blue_max) 
    

    # Morphological transformation

    kernal = np.ones((5, 5), "uint8") 

    # red color
    red_mask = cv2.dilate(red_mask, kernal) 
    res_red = cv2.bitwise_and(imageFrame, imageFrame, mask = red_mask) 
    
    # green color
    green_mask = cv2.dilate(green_mask, kernal) 
    res_green = cv2.bitwise_and(imageFrame, imageFrame, mask = green_mask) 
    
    # blue color
    blue_mask = cv2.dilate(blue_mask, kernal) 
    res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask = blue_mask) 
    
    # Creating contour to detect red color
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for _, contour in enumerate(contours):
        area = cv2.contourArea(contour) 
        if(area > 500): 
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            
            cv2.putText(imageFrame, "Red Color", (x, y), cv2.FONT_HERSHEY_PLAIN , 1.0, (0, 0, 255))     
    
    # Creating contour to detect green color
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    for _, contour in enumerate(contours):
        area = cv2.contourArea(contour) 
        if(area > 500): 
            x, y, w, h = cv2.boundingRect(contour) 
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            cv2.putText(imageFrame, "Green Color", (x, y), cv2.FONT_HERSHEY_PLAIN , 1.0, (0, 255, 0))
    
    # Creating contour to detect blue color
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for _, contour in enumerate(contours):
        area = cv2.contourArea(contour) 
        if(area > 500):
            x, y, w, h = cv2.boundingRect(contour) 
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            
            cv2.putText(imageFrame, "Blue Color", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0)) 
              
    cv2.imshow("RGB color detection in Real-Time", imageFrame)

    if cv2.waitKey(10) & 0xFF == ord('q'): 
        webcam.release()
        cv2.destroyAllWindows() 
        break