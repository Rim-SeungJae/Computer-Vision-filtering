import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from A1_image_filtering import *


def compute_image_gradient(img):
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

    mag = np.sqrt(grad_x*grad_x+grad_y*grad_y)
    dir = np.arctan(grad_y/grad_x)

    return mag, dir

def non_maximum_suppression_dir(mag, dir):
    imax = mag.shape[0]
    jmax = mag.shape[1]
    for i,x in enumerate(mag):
        for j,y in enumerate(x):
            if (dir[i][j] <= np.pi / 2 and dir[i][j] >= np.pi / 8 * 3):
                if i+1 < imax and mag[i+1][j]>=mag[i][j] or i+2<imax and mag[i+2][j]>=mag[i][j]:
                    mag[i][j] = 0
                if i-1>=0 and mag[i-1][j]>=mag[i][j] or i-2>=0 and mag[i-2][j]>=mag[i][j]:
                    mag[i][j] = 0

            elif (dir[i][j] <= -np.pi / 8 * 3 and dir[i][j] >= -np.pi / 2):
                if i+1 < imax and mag[i+1][j]>=mag[i][j] or i+2 < imax and mag[i+2][j]>=mag[i][j]:
                    mag[i][j] = 0
                if i-1>=0 and mag[i-1][j]>=mag[i][j] or i-2>=0 and mag[i-2][j]>=mag[i][j]:
                    mag[i][j] = 0

            elif dir[i][j] <= np.pi / 8 * 3 and dir[i][j] >= np.pi / 8:
                if i+1<imax and j+1<jmax and mag[i+1][j+1]>=mag[i][j] or i+2<imax and j+2<jmax and mag[i+2][j+2]>=mag[i][j]:
                    mag[i][j] = 0
                if i-1>=0 and j-1>=0 and mag[i-1][j-1]>=mag[i][j] or i-2>=0 and j-2>=0 and mag[i-2][j-2]>=mag[i][j]:
                    mag[i][j] = 0

            elif dir[i][j] <= -np.pi / 8 and dir[i][j] >= -np.pi / 8 * 3:
                if i-1>=0 and j+1<jmax and mag[i-1][j+1]>=mag[i][j] or i-2>=0 and j+2<jmax and mag[i-2][j+2]>=mag[i][j]:
                    mag[i][j] = 0
                if i+1<imax and j-1>=0 and mag[i+1][j-1]>=mag[i][j] or i+2<imax and j-2>=0 and mag[i+2][j-2]>=mag[i][j]:
                    mag[i][j] = 0

            elif dir[i][j] <= np.pi/8 and dir[i][j] >= -np.pi/8:
                if j+1<jmax and mag[i][j+1]>=mag[i][j] or j+2<jmax and mag[i][j+2]>=mag[i][j]:
                    mag[i][j] = 0
                if j-1>=0 and mag[i][j-1]>=mag[i][j] or j-2>=0 and mag[i][j-2]>=mag[i][j]:
                    mag[i][j] = 0

    return mag



if __name__=="__main__":
    img = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./shapes.png', cv2.IMREAD_GRAYSCALE)

    out = cross_correlation_1d(img,get_gaussian_filter_1d(7,1.5))
    out = cross_correlation_1d(out,np.transpose(get_gaussian_filter_1d(7,1.5)))

    start = time.time()
    mag, dir = compute_image_gradient(out)
    end = time.time()

    print('computational time taken by compute_image_gradient with lenna.png: ',end-start)

    plt.imshow(mag,cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.savefig('./result/part_2_edge_raw_lenna.png')

    out = cross_correlation_1d(img2,get_gaussian_filter_1d(7,1.5))
    out = cross_correlation_1d(out,np.transpose(get_gaussian_filter_1d(7,1.5)))

    start = time.time()
    mag2, dir2 = compute_image_gradient(out)
    end = time.time()

    print('computational time taken by compute_image_gradient with shapes.png: ',end-start)

    plt.imshow(mag2,cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.savefig('./result/part_2_edge_raw_shapes.png')

    plt.subplot(121)
    plt.imshow(mag, cmap='gray')
    plt.title('raw lenna')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(mag2, cmap='gray')
    plt.title('raw shapes')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    start = time.time()
    suppressed_mag = non_maximum_suppression_dir(mag,dir)
    end = time.time()

    print('computational time taken by non_maximum_suppression with lenna.png: ', end - start)

    plt.imshow(suppressed_mag,cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.savefig('./result/part_2_edge_sup_lenna.png')

    start = time.time()
    suppressed_mag2 = non_maximum_suppression_dir(mag2,dir2)
    end = time.time()

    print('computational time taken by non_maximum_suppression with shapes.png: ', end - start)

    plt.imshow(suppressed_mag2,cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.savefig('./result/part_2_edge_sup_shapes.png')

    plt.subplot(121)
    plt.imshow(suppressed_mag, cmap='gray')
    plt.title('suppressed lenna')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(suppressed_mag2, cmap='gray')
    plt.title('suppressed shapes')
    plt.xticks([])
    plt.yticks([])
    plt.show()