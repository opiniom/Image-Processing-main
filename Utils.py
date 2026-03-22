import numpy as np

def myConv2D(A, B, strides, param):
    # 합성곱 연산의 출력 매개변수를 정의합니다.
    A_pad, pad = pad_Checker(B, A, param)  # 패딩이 적용된 새로운 행렬을 생성합니다.
    B = np.flipud(np.fliplr(B))
    # 커널 행렬(필터)의 x, y 길이를 지정합니다.
    xKernShape = B.shape[0]
    yKernShape = B.shape[1]

    # 패딩 처리된 행렬의 x, y 길이를 지정합니다.

    xApadShape = A_pad.shape[0]
    yApadShape = A_pad.shape[1]

    # 출력 행렬의 형태(shape)를 정의합니다.

    xOutput = int(((xApadShape - xKernShape + 2 * pad) / strides) + 1)
    yOutput = int(((yApadShape - yKernShape + 2 * pad) / strides) + 1)

    output = np.zeros((xOutput, yOutput),dtype=np.float32)  # 새로운 출력 행렬을 0으로 채웁니다.

    # 합성곱 연산을 시작합니다.

    for y in range(A_pad.shape[1]):

        if y > A_pad.shape[1] - yKernShape:  # 커널이 경계를 벗어나면 다음 행으로 이동합니다.
            break

        if y % strides == 0:

            for x in range(A_pad.shape[0]):

                if x > A_pad.shape[0] - xKernShape:
                    break
                try:
                    if x % strides ==0:
                    # 합성곱 연산을 수행합니다.

                        output[x, y] = (B * A_pad[x: x + xKernShape, y: y + yKernShape]).sum()

                except:
                    break
    return output

def pad_Checker(B, A, param):
    n, n = B.shape
    padv = int((n - 1) / 2)

    if param == 'same':  # param='same'인 경우, 출력 형태가 입력 형태와 동일하도록 설정합니다.
        C = pad_image(A, size=padv)

    elif param == 'valid':
        padv = 0
        C = A
    else:
        print("Invalid param name.!! Choose same or valid.")
    return C, padv

def pad_image(A, size):
    # 모든 면에 동일한 크기의 패딩을 적용합니다.
    A_pad = np.zeros((A.shape[0] + 2 * size, A.shape[1] + 2 * size), dtype=np.float32)
    A_pad[1 * size:-1 * size,1 * size:-1 * size] = A
    return A_pad
