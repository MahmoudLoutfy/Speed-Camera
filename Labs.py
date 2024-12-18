import math
import skimage.io as io
from scipy.signal import convolve2d
from skimage.color import rgb2gray,rgb2hsv
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar

def show_images(images,titles=None):
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()

def showHist(img,histogramImg):
    plt.figure()
    bar(histogramImg[1]*255, histogramImg[0], width=0.8, align='center')

def getHistogram(img):
    # works for grayscale images represented from 0 to 255
    hist = np.zeros(256)
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            hist[img[i, j]] += 1
    return hist

def toBinary(grey_img):
    # works only on gray images
    img = grey_img.copy()
    threshold = np.median(img)
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            if(img[i][j]<threshold):
                img[i][j]=0
            else:
                img[i][j]=1
    return img

def medianfilter(img):
    img1 = img.copy()
    for y in range (1, img.shape[0]-3):
       for x in range (1, img.shape[1]-3): 
           window = img[y-1:y+2, x-1:x+2]
           median = np.median(window)
           img1[y, x]=median
    return img1

def negativeTransform(img):
    # works for grayscale images represented from 0 to 255
    return 255 - img

def gammaCorrection(img, C, gamma):
    return C * (img ** gamma)

def equalizeHist(img):
    # works for grayscale images represented from 0 to 255
    img1=img.copy()
    H = getHistogram(img)
    H_c = np.zeros(256)
    T = np.zeros(256)
    [y , x]=img.shape[0:2]
    H_c[0] = H[0]
    for i in range(1, 256):
        H_c[i] = H_c[i - 1] + H[i]
    for j in range(256):
        T[j] = math.floor(255*(H_c[j])/(y*x))
    for i in range(x):
        for j in range(y):
            img1[j][i] = T[img[j][i]]
    return img1

def sobel_x(img):
    # to get binary image of edges, use a threshold where binary_img = horizontal_img > threshold
    filter = np.array([[-1, 0, 1], 
                       [-2, 0, 2],
                       [-1, 0, 1]])
    horizontal_img = convolve2d(img, filter)
    return horizontal_img

def sobel_y(img):
    # to get binary image of edges, use a threshold where binary_img = vertical_img > threshold
    filter = np.array([[-1, -2, -1], 
                       [0, 0, 0],
                       [1, 2, 1]])
    vertical_img = convolve2d(img, filter)
    return vertical_img

def sobel(img):
    # to get binary image of edges, use a threshold where binary_img = edge_strength > threshold
    horizontal_img = sobel_x(img)
    verticalal_img = sobel_y(img)
    edge_strength = np.sqrt(horizontal_img**2 + verticalal_img**2)
    return edge_strength

# Another edge detection method: Laplace of Gaussian(LoG) found in Lab5

def erode(img, window_size=3):
    # works only on binary images
    img1 = img.copy()
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            x1 = max(0, i - window_size // 2)
            y1 = max(0, j - window_size // 2)
            x2 = min(img.shape[1] - 1, i + window_size // 2)
            y2 = min(img.shape[0] - 1, j + window_size // 2)
            img1[j, i] = 1 if np.all(img[y1:y2,x1:x2] == 1) else 0
    return img1

def dilate(img, window_size=3):
    # works only on binary images
    img1 = img.copy()
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            x1 = max(0, i - window_size // 2)
            y1 = max(0, j - window_size // 2)
            x2 = min(img.shape[1] - 1, i + window_size // 2)
            y2 = min(img.shape[0] - 1, j + window_size // 2)
            img1[j, i] = 1 if np.any(img[y1:y2,x1:x2] == 1) else 0
    return img1

def iterativeThreshold(image):
    # only works on grayscale images
    img = (image.copy()*255).astype('uint8') % 255
    hist = getHistogram(img)
    sum = 0
    weighted_sum = 0
    for i in range(256):
        sum += hist[i]
        weighted_sum += hist[i] * i
    Tinit = int(weighted_sum // sum)
    while(True):
        sum1 = 0
        weighted_sum1 = 0
        for j in range(Tinit):
            sum1 += hist[j]
            weighted_sum1 += hist[j] * j
        T1 = int(weighted_sum1 // sum1)

        sum2 = 0
        weighted_sum2 = 0
        for k in range(Tinit,256):
            sum2 += hist[k]
            weighted_sum2 += hist[k] * k
        T2 = int(weighted_sum2 // sum2)

        Told = Tinit
        Tinit = int((T1 + T2) // 2)

        if (Tinit == Told):
            break
    binary_mask = img >= Tinit
    return binary_mask

def localThreshold(img):
    # only works on grayscale images
    height, width = np.shape(img)
    center_x, center_y = width // 2, height // 2
    top_left = img[:center_y, :center_x]
    top_right = img[:center_y, center_x:]
    bottom_left = img[center_y:, :center_x]
    bottom_right = img[center_y:, center_x:]
    top_left = iterativeThreshold(top_left)
    top_right = iterativeThreshold(top_right)
    bottom_left = iterativeThreshold(bottom_left)
    bottom_right = iterativeThreshold(bottom_right)
    top_half = np.hstack((top_left, top_right))
    bottom_half = np.hstack((bottom_left, bottom_right))
    image = np.vstack((top_half, bottom_half))
    return image

def adaptiveThreshold(frame):
    # only works on grayscale images represented from 0 to 1 as float, works best with sudden changes such as text
    height, width = np.shape(frame)
    threshold = 15
    window_size = width // 8
    mask = frame.copy()
    integral = frame.copy()
    # calculate the integral of the input image
    for i in range(width):
        sum = 0
        for j in range(height):
            sum += frame[j, i]
            if i == 0:
                integral[j, i] = sum
            else:
                integral[j, i] = integral[j, i - 1] + sum
    i = 0
    j = 0
    for i in range(width):
        for j in range(height):
            x1 = max(0, i - window_size // 2)
            x2 = min (width - 1, i + window_size // 2)
            y1 = max(0, j - window_size // 2)
            y2 = min (height - 1, j + window_size // 2)
            count = (x2 - x1) * (y2 - y1)
            sum = integral[y2, x2]
            if y1 > 0:
                sum -= integral[y1 - 1, x2]
            if x1 > 0:
                sum -= integral[y2, x1 - 1]
            if x1 > 0 and y1 > 0:
                sum += integral[y1 - 1, x1 - 1]
            if (frame[j, i] * count) <= (sum * (100 - threshold) / 100):
                mask[j, i] = 255
            else:
                mask[j, i] = 0
    return mask