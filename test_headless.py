import cv2
import numpy as np
from Utils import *
from Noise_Filters import *
from Filtering_Methods import *

def main():
    print("Headless Test Starting...")
    image = cv2.imread("resources/lena.bmp")
    if image is None:
        print("Error: No image")
        return
        
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    SPnoise_img = myImNoise(imgGray, param="saltandpepper", noise_percent=90)
    
    print("1. Running Base Filter...")
    Base_Filtered = myImFilter(SPnoise_img, param="mean")
    
    print("2. Running Deviation Filter...")
    try:
        Filter1_Result = deviation_filter(SPnoise_img)
        print("Deviation Filter succeeded!")
    except Exception as e:
        print(f"Deviation Filter failed: {e}")
        return
        
    print("3. Running Group Filter...")
    try:
        Filter2_Result = group_filter(SPnoise_img)
        print("Group Filter succeeded!")
    except Exception as e:
        print(f"Group Filter failed: {e}")
        return

    print("Headless test completed successfully.")

if __name__ == "__main__":
    main()
