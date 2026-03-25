import cv2
import numpy as np
from Utils import *
from Noise_Filters import *
from Filtering_Methods import *

def main():
    print("Headless Test Starting...")
    image = cv2.imread("resources/testimg.png")
    if image is None:
        print("Error: No image")
        return
        
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    SPnoise_img = myImNoise(imgGray, param="saltandpepper", noise_percent=90)
    
    print("1. Running Base Filter...")
    Base_Filtered = myImFilter(SPnoise_img, param="mean")
    
    print("2. Running Filter 1 (Corner Group)...")
    try:
        Filter1_Result = custom_corner_filter(SPnoise_img)
        print("Filter 1 succeeded!")
    except Exception as e:
        print(f"Filter 1 failed: {e}")
        return
        
    print("3. Running Filter 2 (Min Noise Dir)...")
    try:
        Filter2_Result = custom_direction_filter(SPnoise_img)
        print("Filter 2 succeeded!")
    except Exception as e:
        print(f"Filter 2 failed: {e}")
        return

    print("Headless test completed successfully.")

if __name__ == "__main__":
    main()
