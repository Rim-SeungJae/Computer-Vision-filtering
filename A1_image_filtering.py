import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

def cross_correlation_1d(img,kernel):
    padlen_x = int(kernel.shape[0] / 2)
    padlen_y = int(kernel.shape[1] / 2)
    padded = np.zeros((img.shape[0] + padlen_x * 2, img.shape[1] + padlen_y * 2))
    dst = np.zeros(img.shape)

    #kernel.shape is (1,n)
    if (padlen_x == 0):
        padded[0:, padlen_y:-padlen_y] = img

        for i,x in enumerate(padded):
            for j,y in enumerate(x):
                if j<padlen_y:
                    padded[i][j] = padded[i][padlen_y]
                if j>=img.shape[1]+padlen_y:
                    padded[i][j] = padded[i][img.shape[1]+padlen_y-1]

        for i,x in enumerate(dst):
            for j,y in enumerate(x):
                    for k in range(-padlen_y,padlen_y+1):
                            dst[i][j] += kernel[0][padlen_y+k] * padded[i][j+padlen_y+k]


    #kernel.shape is (n,1)
    else:
        padded[padlen_x:-padlen_x, 0:] = img

        for i,x in enumerate(padded):
            for j,y in enumerate(x):
                if i<padlen_x:
                    padded[i][j] = padded[padlen_x][j]
                if i>=img.shape[0]+padlen_x:
                    padded[i][j] = padded[img.shape[0]+padlen_x-1][j]

        for i,x in enumerate(dst):
            for j,y in enumerate(x):
                    for k in range(-padlen_x,padlen_x+1):
                            dst[i][j] += kernel[padlen_x+k][0] * padded[i+padlen_x+k][j]

    return dst


def cross_correlation_2d(img, kernel):
    padlen_x = int(kernel.shape[0] / 2)
    padlen_y = int(kernel.shape[1] / 2)
    padded = np.zeros((img.shape[0] + padlen_x * 2, img.shape[1] + padlen_y * 2))
    dst = np.zeros(img.shape)

    padded[padlen_x:-padlen_x:, padlen_y:-padlen_y] = img

    for i, x in enumerate(padded):
        for j, y in enumerate(x):
            if j < padlen_y and i < padlen_x:
                padded[i][j] = img[0][0]
            elif j<padlen_y and i>=img.shape[0]+padlen_x:
                padded[i][j] = padded[img.shape[0]+padlen_x-1][padlen_y]
            elif j>=img.shape[1]+padlen_y and i < padlen_x:
                padded[i][j] = padded[padlen_x][img.shape[1]+padlen_y-1]
            elif j>=img.shape[1]+padlen_y and i>=img.shape[0]+padlen_x:
                padded[i][j] = padded[img.shape[0]+padlen_x-1][img.shape[1]+padlen_y-1]
            elif j <padlen_y:
                padded[i][j] = padded[i][padlen_y]
            elif i <padlen_x:
                padded[i][j] = padded[padlen_x][j]
            elif j >= img.shape[1] + padlen_y:
                padded[i][j] = padded[i][img.shape[1] + padlen_y - 1]
            elif i >= img.shape[0] + padlen_x:
                padded[i][j] = padded[img.shape[0] + padlen_x - 1][j]

    for i,x in enumerate(dst):
        for j,y in enumerate(x):
                for k in range(-padlen_x,padlen_x+1):
                    for l in range(-padlen_y,padlen_y+1):
                        dst[i][j] += kernel[padlen_x+k][padlen_y+l] * padded[i+padlen_x+k][j+padlen_y+l]

    return dst


def get_gaussian_filter_1d(size,sigma):
    r = int(size/2)
    i, j = np.ogrid[0:1, -r:r+1]
    gaussian_filter = np.exp(-(i*i+j*j)/(2. * sigma * sigma))
    sum = gaussian_filter.sum()
    gaussian_filter /= sum
    return gaussian_filter


def get_gaussian_filter_2d(size,sigma):
    r = int(size/2)
    i, j = np.ogrid[-r:r+1, -r:r+1]
    gaussian_filter = np.exp(-(i*i+j*j)/(2. * sigma * sigma))
    sum = gaussian_filter.sum()
    gaussian_filter /= sum
    return gaussian_filter

if __name__=="__main__":
    img = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./shapes.png', cv2.IMREAD_GRAYSCALE)

    print('result of get_gaussian_filter_1d(5,1): ', get_gaussian_filter_1d(5,1))
    print('result of get_gaussian_filter_2d(5,1): ')
    print(get_gaussian_filter_2d(5,1))

    kernels = [
        (5,1),
        (5,6),
        (5,11),
        (11,1),
        (11,6),
        (11,11),
        (17,1),
        (17,6),
        (17,11)
    ]

    for i, kernel in enumerate(kernels):
        output = cross_correlation_2d(img,get_gaussian_filter_2d(kernel[0],kernel[1]))
        plt.subplot(int('33{}'.format(i+1)))
        plt.imshow(output, cmap='gray')
        plt.title('{}x{} s={}'.format(kernel[0],kernel[0],kernel[1]))
        plt.xticks([])
        plt.yticks([])
        print("...")

    fig = plt.gcf()
    fig.savefig('./result/part_1_gaussian_filtered_lenna.png')
    plt.show()

    for i, kernel in enumerate(kernels):
        output = cross_correlation_2d(img2,get_gaussian_filter_2d(kernel[0],kernel[1]))
        plt.subplot(int('33{}'.format(i+1)))
        plt.imshow(output, cmap='gray')
        plt.title('{}x{} s={}'.format(kernel[0],kernel[0],kernel[1]))
        plt.xticks([])
        plt.yticks([])

    fig = plt.gcf()
    fig.savefig('./result/part_1_gaussian_filtered_shapes.png')
    plt.show()

    #1D gaussian filtering start
    start_1d = time.time()

    output_1d = cross_correlation_1d(img,get_gaussian_filter_1d(17,6))
    output_1d = cross_correlation_1d(output_1d,np.transpose(get_gaussian_filter_1d(17,6)))

    end_1d = time.time()
    #1D gaussian filtering end

    #2D gaussian filtering start
    start_2d = time.time()

    output_2d = cross_correlation_2d(img,get_gaussian_filter_2d(17,6))

    end_2d = time.time()
    #2D gaussian filtering end

    difference = output_2d - output_1d

    print('')
    print('below are compare results of 1D gaussian filtering and 2D gaussian filtering with lenna.png(17x17, s=6)')
    plt.imshow(difference,cmap='gray')
    plt.title('lenna.png difference map')
    plt.show()
    print('sum of intensity differences: ', np.absolute(difference).sum())
    print('computational time of 1D filtering: ', end_1d-start_1d)
    print('computational time of 2D filtering: ', end_2d-start_2d)

    #1D gaussian filtering start
    start_1d = time.time()

    output_1d = cross_correlation_1d(img2,get_gaussian_filter_1d(17,6))
    output_1d = cross_correlation_1d(output_1d,np.transpose(get_gaussian_filter_1d(17,6)))

    end_1d = time.time()
    #1D gaussian filtering end

    #2D gaussian filtering start
    start_2d = time.time()

    output_2d = cross_correlation_2d(img2,get_gaussian_filter_2d(17,6))

    end_2d = time.time()
    #2D gaussian filtering end

    difference = output_2d - output_1d

    print('')
    print('below are compare results of 1D gaussian filtering and 2D gaussian filtering with shapes.png(17x17, s=6)')
    plt.imshow(difference,cmap='gray')
    plt.title('shapes.png difference map')
    plt.show()
    print('sum of intensity differences: ', np.absolute(difference).sum())
    print('computational time of 1D filtering: ', end_1d-start_1d)
    print('computational time of 2D filtering: ', end_2d-start_2d)
