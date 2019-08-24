"""
Edge Detection
(Due date: March 8th, 11: 59 P.M.)

The goal of this task is to experiment with two commonly used edge detection operator, i.e., Prewitt operator and Sobel operator,
and familiarize you with 'tricks', e.g., padding, commonly used by computer vision 'researchers'.

Please complete all the functions that are labelled with '# TODO'. Hints or steps are provided to make your lives easier.
Whem implementing the functions, comment the lines 'raise NotImplementedError' instead of deleting them. As we have
written lots of utility functions for you, you only need to write about 30 lines of code. The functions defined in utils.py
are building blocks you could use when implementing the functions labelled with 'TODO'.

I strongly suggest you to read the function zero_pad that is defined in utils.py. It is quite important!

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""

import argparse
import copy
import os
import math
import cv2
import numpy as np

import utils

# Prewitt operator
prewitt_x = [[1, 0, -1]] * 3
prewitt_y = [[1] * 3, [0] * 3, [-1] * 3]

# Sobel operator
sobel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="",
        help="path to the image used for edge detection")
    parser.add_argument(
        "--kernel", type=str, default="sobel",
        choices=["prewitt", "sobel", "Prewitt", "Sobel"],
        help="type of edge detector used for edge detection")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if not img.dtype == np.uint8:
        pass

    if show:
        show_image(img)

    img = [list(row) for row in img]
    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def write_image(img, img_saving_path):
    """Writes an image to a given path.
    """
    # print(img)
    # if isinstance(img, list):
    #     img = np.asarray(img, dtype=np.uint8)
    # elif isinstance(img, np.ndarray):
    #     if not img.dtype == np.uint8:
    #         assert np.max(img) <= 1, "Maximum pixel value {:.3f} is greater than 1".format(np.max(img))
    #         img = (255 * img).astype(np.uint8)
    #     else:
    #         raise TypeError("img is neither a list nor a ndarray.")
    # img = (img).astype(np.uint8)
    cv2.imwrite(img_saving_path, img)

def convolve2d(img, kernel):
    """Convolves a given image and a given kernel.

    Steps:
        (1) flips the either the img or the kernel.
        (2) pads the img or the flipped img.
            this step handles pixels along the border of the img,
            and makes sure that the output img is of the same size as the input image.
        (3) applies the flipped kernel to the image or the kernel to the flipped image,
            using nested for loop.

    Args:
        img: nested list (int), image.
        kernel: nested list (int), kernel.

    Returns:
        img_conv: nested list (int), image.
    """
    # TODO: implement this function.
    # raise NotImplementedError
    img = np.array(img)
    row = img.shape[0]  # rows
    col = img.shape[1]  # colms
    # Flip the kernel
    kernel_copy = np.zeros(shape=(3, 3))
    for i in range(3):
        for j in range(3):
            kernel_copy[i][j] = kernel[2 - i][2 - j]

    # Pad the image with ()
    paddedImg = np.zeros(shape=(row + 1, col + 1))
    paddedImg[:img.shape[0], :img.shape[1]] = img

    # Convolve the kernel on image
    img_conv = np.zeros(paddedImg.shape)
    for i in range(1, row):  # (row-2 and col-2 )
        for j in range(1, col):
            sum_ = 0
            for m in range(3):  # (filter 3x3)
                for n in range(3):
                    sum_ = sum_ + (kernel[m][n] * paddedImg[i - 1 + m][j - 1 + n])
            img_conv[i][j] = sum_
            #(sum_ / 27) + 0.1

    # Denoise using Gaussian Filter(The Gaussian outputs a `weighted average' of each pixel's neighborhood,
    # with the average weighted more towards the value of the central pixels. This is in contrast to the mean filter's
    # uniformly weighted average. Because of this, a Gaussian provides gentler smoothing and preserves
    # edges better than a similarly sized mean filter.)
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
    # Best is to use median filter - clarity
    # blur = cv2.GaussianBlur(img_conv, (5, 5), 0)
    return img_conv


def normalize(img):
    """
    The linear normalization of a grayscale digital image is performed according to the formula
    I_new=(I-min)((newMax-newMin)/(max-min))+newMin
    """
    Min = np.amin(img)
    Max = np.amax(img)

    #New min and max
    newMin = 0
    newMax = 255

    #Normalization
    I=(img-Min)*int((newMax-newMin)/(Max-Min)) + newMin
    # TODO: implement this function.
    #raise NotImplementedError
    # I = abs(img)/abs(Max)
    return I

def detect_edges(img, kernel, norm=True):
    """Detects edges using a given kernel.

    Args:
        img: nested list (int), image.
        kernel: nested list (int), kernel used to detect edges.
        norm (bool): whether to normalize the image or not.

    Returns:
        img_edge: nested list (int), image containing detected edges.
    """
    # TODO: detect edges using convolve2d and normalize the image containing detected edges using normalize.
    # raise NotImplementedError
    if not norm:
        img_edges = convolve2d(img,kernel)
    else:
        img_edges = convolve2d(img, kernel)
        img_edges = normalize(img_edges)
    return img_edges


def edge_magnitude(edge_x, edge_y):
    """Calculate magnitude of edges by combining edges along two orthogonal directions.

    Hints:
        Combine edges along two orthogonal directions using the following equation:

        edge_mag = sqrt(edge_x ** 2 + edge_y **).

        Make sure that you normalize the edge_mag, so that the maximum pixel value is 1.

    Args:
        edge_x: nested list (int), image containing detected edges along one direction.
        edge_y: nested list (int), image containing detected edges along another direction.

    Returns:
        edge_mag: nested list (int), image containing magnitude of detected edges.
    """
    # TODO: implement this function.
    # raise NotImplementedError

    x_2 = np.zeros(shape=(edge_x.shape[0],edge_x.shape[1]))
    y_2 = np.zeros(shape=(edge_x.shape[0],edge_x.shape[1]))
    for i in range(edge_x.shape[0]):
        for j in range(edge_x.shape[1]):
            x_2[i][j] = edge_x[i][j]*edge_x[i][j]
            y_2[i][j] = edge_y[i][j] * edge_y[i][j]
    edge_mag = np.sqrt(x_2 + y_2)
    return edge_mag


def sum_edges(img_edge_x, img_edge_y):
    return img_edge_x+img_edge_y


def stripes_removal(img_edge_x, img_edge_y):
    alpha = 0.5
    input_alpha = 0.1
    if 0 <= alpha <= 1:
        alpha = input_alpha

    img_edge_x = cv2.imread("D:/UB/summer/CVIP/p1/project1/results/prewitt_edge_x.jpg")
    img_edge_y = cv2.imread("D:/UB/summer/CVIP/p1/project1/results/prewitt_edge_y.jpg")
    # [blend_images]
    beta = (1.0 - alpha)
    cpy = cv2.addWeighted(img_edge_y - img_edge_x, alpha, img_edge_x, beta, 0.0)
    for i in range(cpy.shape[0]):
        for j in range(cpy.shape[1]):
            for k in range(cpy.shape[2]):
                if cpy[i][j][k] >= 200:
                    cpy[i][j][k] = 1
    cpy = cv2.medianBlur(cpy, 5, 3)

    for i in range(4):
        cpy = cv2.bilateralFilter(cpy, 5, 145, 145)

    for i in range(cpy.shape[0]):
        for j in range(cpy.shape[1]):
            for k in range(cpy.shape[2]):
                if cpy[i][j][k] <= 40:
                    cpy[i][j][k] = 1
    return cpy


def main():
    args = parse_args()

    img = read_image(args.img_path)

    if args.kernel in ["prewitt", "Prewitt"]:
        kernel_x = prewitt_x
        kernel_y = prewitt_y
    elif args.kernel in ["sobel", "Sobel"]:
        kernel_x = sobel_x
        kernel_y = sobel_y
    else:
        raise ValueError("Kernel type not recognized.")

    if not os.path.exists(args.rs_directory):
        os.makedirs(args.rs_directory)

    img = normalize(np.asarray(img))
    img_edge_x = np.asarray(detect_edges(img, kernel_x, False))
    # x=normalize(img_edge_x)
    print("img_edge_x-Max = "+str(np.max(img_edge_x)))
    print("img_edge_x-Min = "+str(np.min(img_edge_x)))
    # for i in range(img_edge_x.shape[0]):
    #     for j in range(img_edge_x.shape[1]):
    #         if img_edge_x[i][j] >= 10:
    #             img_edge_x[i][j] = 255
    #         else:
    #             img_edge_x[i][j] = 0
    # show_image(x,0)
    # img_edge_x = np.invert(img_edge_x.astype(np.int))
    write_image(img_edge_x, os.path.join(args.rs_directory, "{}_edge_x.jpg".format(args.kernel.lower())))

    img_edge_y = np.asarray(detect_edges(img, kernel_y, False))
    # img_edge_y=normalize(img_edge_y)
    print("img_edge_y-Max = " + str(np.max(img_edge_y)))
    print("img_edge_y-Min = " + str(np.min(img_edge_y)))
    # for i in range(img_edge_y.shape[0]):
    #     for j in range(img_edge_y.shape[1]):
    #         if img_edge_y[i][j] >= 0.2:
    #             img_edge_y[i][j] = 255
    #         else:
    #             img_edge_y[i][j] = 0
    # img_edge_y = np.invert(img_edge_y.astype(np.int))
    write_image(img_edge_y, os.path.join(args.rs_directory, "{}_edge_y.jpg".format(args.kernel.lower())))

    img_edges = edge_magnitude((img_edge_x), (img_edge_y))
    # img_edges=normalize(img_edges)
    print("img_edge-Max = " + str(np.max(img_edges)))
    print("img_edge-Min = " + str(np.min(img_edges)))
    # for i in range(img_edges.shape[0]):
    #     for j in range(img_edges.shape[1]):
    #         if img_edges[i][j] >= 200:
    #             img_edges[i][j] = 255
    #         else:
    #             img_edges[i][j] = 0
    # img_edges = np.invert(img_edges.astype(np.int))
    write_image(img_edges, os.path.join(args.rs_directory, "{}_edge_mag.jpg".format(args.kernel.lower())))

    img_sum_edges = np.array(sum_edges(img_edge_x,img_edge_y))
    write_image(img_sum_edges, os.path.join(args.rs_directory, "{}_sum_edges.jpg".format(args.kernel.lower())))

    img_stripes_removal = np.array(stripes_removal(img_edge_x,img_edge_y))
    write_image(img_stripes_removal, os.path.join(args.rs_directory, "{}_stripes_removal.jpg".format(args.kernel.lower())))

if __name__ == "__main__":
    main()