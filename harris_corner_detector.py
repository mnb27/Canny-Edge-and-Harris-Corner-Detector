########################################## HARRIS CORNER DETECTOR FROM FIRST PRINCIPLES ###################################################
# Aman Bilaiya 2018CSB1069

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from scipy import ndimage

# RGB --> grayscale
def rgb2gray(img):
    return (np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])).astype('float32')

#display image                                                                  
def view_img(img_name):
    plt.imshow(img_name, cmap = plt.get_cmap('gray'))
    plt.show() 
    
'''
Part A) Filtered gradient: Compute x and y gradients Fx and Fy, the same as in the Canny edge detector.
'''    
# Method to do convolution of input image with sobel filter to smoothen the image and finding x & y gradients
def convolveWithGaussianDerivative(img):
  sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
  sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
  
  Ix = ndimage.filters.convolve(img, sobel_x)
  Iy = ndimage.filters.convolve(img, sobel_y)
  
  return Ix, Iy

'''
B) Find corners: For each pixel (x, y), look in a window of size 2m+1 x 2m+1around the pixel (you can use m = 4). Accumulate over
this window the covariance matrix C, which contains the average of the products of x and y gradients:
'''

# Method to determine R score for each pixel [R_mat]
def Calculate_Rscore_mat(img, Iy, Ix, m, k):
    Ixx = np.square(Ix)
    Ixy = Iy*Ix
    Iyy = np.square(Iy)
    height, width = img.shape[0], img.shape[1]

    R_mat = np.zeros([height, width],dtype='float64')

    for y in range(m, height-m):
        for x in range(m, width-m):
            Kxx = Ixx[y-m:y+m+1, x-m:x+m+1]
            Kxy = Ixy[y-m:y+m+1, x-m:x+m+1]
            Kyy = Iyy[y-m:y+m+1, x-m:x+m+1]
            Sxx = np.sum(Kxx)
            Sxy = np.sum(Kxy)
            Syy = np.sum(Kyy)

            determinant = (Sxx * Syy) - np.square(Sxy)
            trace = Sxx + Syy
            R_mat[y][x] =  determinant - (k*np.square(trace))
    return R_mat

# Function to detect corners based on threshold values and r score
def DetectCorners(img, r_mat,Th_ratio):
  corner_list = []
  Th = Th_ratio*r_mat.max()
  for y in range(img.shape[0]):
    for x in range(img.shape[1]):
      if r_mat[y][x] > Th:
        corner_list.append([y,x,r_mat[y][x]])
  return corner_list

# Function to highlight corners in the input image
def MakeCornerImg(img, corner_list):
  out1 = np.zeros(img.shape,img.dtype)
  out2 = img.copy()
  
  for i in range(len(corner_list)):
    y = corner_list[i][0]
    x = corner_list[i][1]
    out1[y][x][0] = 1.0
    out1[y][x][1] = 1.0
    out1[y][x][2] = 1.0
    out2[y][x][0] = 1.0
    out2[y][x][1] = 0.0
    out2[y][x][2] = 0.0
  return out1, out2

def GenerateResults(input_img,Th_ratio): 
    # input_img = 'einstein' # Input image name
    img = mpimg.imread('data/'+input_img+'.bmp').astype('float32')/255.0 #Read Image and convert to float
    #plt.imshow(img)

    Ix, Iy = convolveWithGaussianDerivative(rgb2gray(img))

    R_mat = Calculate_Rscore_mat(img, Ix, Iy, m=4, k=0.04)
    temp = R_mat/np.max(R_mat)

    plt.imsave('Results/harris_corner/'+input_img+'_R_Value.jpg',temp,cmap=plt.get_cmap('gray'))

    corner_list = DetectCorners(img, R_mat,Th_ratio)

    out1, out2 = MakeCornerImg(img,corner_list)
    plt.imsave('Results/harris_corner/'+input_img+'_corner_points_beforeNMS.jpg',out1)
    plt.imsave('Results/harris_corner/'+input_img+'_corners_beforeNMS.jpg',out2)


    '''
    C) Non-maximum suppression: Sort L in decreasing order of the corner response e. See numpy.sort and numpy.argsort,
    or the Python built-in function sorted. For each point p, remove all points in the 8-connected neighborhood of p that occur
    later in the list L.
    '''
    sorted_corner_list = sorted(corner_list, key = lambda x: x[2], reverse = True)
    final_corner_list = [] #final_l contains list after non maximal suppression
    final_corner_list.append(sorted_corner_list[0][:-1])
    dis = 3
    xc, yc = [], []
    for i in sorted_corner_list :
        for j in final_corner_list :
            if(abs(i[0] - j[0] <= dis) and abs(i[1] - j[1]) <= dis) :
                break
        else :
            final_corner_list.append(i[:-1])
            xc.append(i[1])
            yc.append(i[0])

    print('Corner pixels detected : ' + str(len(corner_list)))
    print('Corner pixels detected after NMS : ' + str(len(final_corner_list)))     

    corner_img = np.zeros(img.shape)

    for i in final_corner_list :
        y, x = i[0], i[1]
        corner_img[y][x] = 1

    output_path = "Results/harris_corner/"

    plt.imshow(img, cmap = plt.get_cmap('gray'))
    plt.plot(xc, yc, '+', color='red')
    plt.savefig(output_path + "/_corner_points_marked.jpg")
    plt.show()

    plt.imshow(corner_img, cmap = plt.get_cmap('gray'))
    plt.imsave('Results/harris_corner/'+input_img+'_corner_points_dots.jpg',corner_img)
    plt.show()

img_Names = ["bicycle", "bird", "dog", "einstein", "plane", "toy_image"]
Th_ratio = [0.05, 0.05, 0.05, 0.05, 0.01, 0.001]
GenerateResults(img_Names[1],Th_ratio[1])
