import cv2
import cv2.cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from A1_image_filtering import *
from A1_edge_detection import *

def compute_corner_response(img):
    sobel_x=np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    sobel_y=np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    grad_x = cross_correlation_2d(img,sobel_x)
    grad_y = cross_correlation_2d(img,sobel_y)

    window = np.array([
        [1, 1, 1, 1, 1]
    ])

    Ixx = cross_correlation_1d(grad_x*grad_x,window)
    Ixx = cross_correlation_1d(Ixx, np.transpose(window))
    Ixy = cross_correlation_1d(grad_x*grad_y,window)
    Ixy = cross_correlation_1d(Ixy, np.transpose(window))
    Iyy = cross_correlation_1d(grad_y*grad_y,window)
    Iyy = cross_correlation_1d(Iyy, np.transpose(window))

    R = np.zeros_like(img)
    for i,x in enumerate(R):
        for j,y in enumerate(x):
            R[i][j] = (Ixx[i][j]*Iyy[i][j]-Ixy[i][j]*Ixy[i][j]) - 0.04 * (Ixx[i][j]+Iyy[i][j]) * (Ixx[i][j]+Iyy[i][j])

    R = (R>0) * R

    R /= np.max(R)

    return R


def non_maximum_supression_win(R,winSize = 11):
    padlen_x = 5
    padlen_y = 5
    padded = np.zeros((R.shape[0] + padlen_x * 2, R.shape[1] + padlen_y * 2))

    padded[padlen_x:-padlen_x:, padlen_y:-padlen_y] = R

    for i, x in enumerate(padded):
        for j, y in enumerate(x):
            if j < padlen_y and i < padlen_x:
                padded[i][j] = R[0][0]
            elif j < padlen_y and i >= R.shape[0] + padlen_x:
                padded[i][j] = padded[R.shape[0] + padlen_x - 1][padlen_y]
            elif j >= R.shape[1] + padlen_y and i < padlen_x:
                padded[i][j] = padded[padlen_x][R.shape[1] + padlen_y - 1]
            elif j >= R.shape[1] + padlen_y and i >= R.shape[0] + padlen_x:
                padded[i][j] = padded[R.shape[0] + padlen_x - 1][R.shape[1] + padlen_y - 1]
            elif j < padlen_y:
                padded[i][j] = padded[i][padlen_y]
            elif i < padlen_x:
                padded[i][j] = padded[padlen_x][j]
            elif j >= R.shape[1] + padlen_y:
                padded[i][j] = padded[i][R.shape[1] + padlen_y - 1]
            elif i >= R.shape[0] + padlen_x:
                padded[i][j] = padded[R.shape[0] + padlen_x - 1][j]

    for i,x in enumerate(R):
        for j,y in enumerate(x):
            if R[i][j] < padded[i:i+winSize,j:j+winSize].max() or R[i][j]<=0.1:
                R[i][j] = 0

    return R


if __name__=="__main__":
    img = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./shapes.png', cv2.IMREAD_GRAYSCALE)

    img_CLR = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img2_CLR = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    out = cross_correlation_1d(img,get_gaussian_filter_1d(7,1.5))
    out = cross_correlation_1d(out,np.transpose(get_gaussian_filter_1d(7,1.5)))\

    out2 = cross_correlation_1d(img2,get_gaussian_filter_1d(7,1.5))
    out2 = cross_correlation_1d(out2,np.transpose(get_gaussian_filter_1d(7,1.5)))

    start1 = time.time()
    R = compute_corner_response(out)
    end1 = time.time()

    start2 = time.time()
    R2 = compute_corner_response(out2)
    end2 = time.time()

    print('computational time of compute_corner_response of lenna.png: ', end1-start1)

    plt.imshow(R,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    fig = plt.gcf()
    fig.savefig('./result/part_3_corner_raw_lenna.png')
    plt.title('raw lenna')
    plt.show()

    print('computational time of compute_corner_response of shapes.png: ', end2-start2)

    plt.imshow(R2,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    fig = plt.gcf()
    fig.savefig('./result/part_3_corner_raw_shapes.png')
    plt.title('raw shapes')
    plt.show()

    img_CLR[R>0.1*R.max()]=[0,255,0]
    img2_CLR[R2 > 0.1 * R2.max()] = [0, 255, 0]

    plt.imshow(img_CLR)
    plt.xticks([])
    plt.yticks([])
    fig = plt.gcf()
    fig.savefig('./result/part_3_corner_bin_lenna.png')
    plt.title('bin lenna')
    plt.show()

    plt.imshow(img2_CLR)
    plt.xticks([])
    plt.yticks([])
    fig = plt.gcf()
    fig.savefig('./result/part_3_corner_bin_shapes.png')
    plt.title('bin shapes')
    plt.show()

    img_CLR = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img2_CLR = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    start1 = time.time()
    supressed_R = non_maximum_supression_win(R,11)
    end1 = time.time()

    print('computational time taken by non_maximum_suppression_win of lenna: ',end1-start1)

    for i,x in enumerate(supressed_R):
        for j,y in enumerate(x):
            if y>0.1:
                cv2.circle(img_CLR,(j,i),5,[0,255,0],2)

    start2 = time.time()
    supressed_R2 = non_maximum_supression_win(R2, 11)
    end2 = time.time()

    print('computational time taken by non_maximum_suppression_win of shapes: ',end2-start2)

    for i, x in enumerate(supressed_R2):
        for j, y in enumerate(x):
            if y > 0.1:
                cv2.circle(img2_CLR, (j, i), 5, [0, 255, 0], 2)

    cv2.imwrite('./result/part_3_corner_sup_lenna.png',img_CLR)
    cv2.imwrite('./result/part_3_corner_sup_shapes.png', img2_CLR)

    cv2.imshow('supressed lenna',img_CLR)
    cv2.imshow('supressed shapes',img2_CLR)
    cv2.waitKey()
    cv2.destroyAllWindows()