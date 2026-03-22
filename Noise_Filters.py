import numpy as np
from Utils import *
import random

def sp_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob        # 노이즈가 나타날 확률에 대한 임계값
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

# 어떤 노이즈를 구현할지 결정하는 함수 정의입니다.
def myImNoise(A, param, noise_percent=50):
    # 'saltandpepper'를 선택한 경우 점잡음을 적용합니다.
    if param =="saltandpepper":
        prob = (noise_percent / 100.0) / 2.0
        B = sp_noise(A, prob=prob)

    else:
        print("잘못된 파라미터 이름입니다.!! 노이즈를 위해 saltandpepper를 선택해주세요.")
    return B