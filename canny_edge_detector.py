######################################## CANNY EDGE DETECTOR FROM FIRST PRINCIPLES ########################################################
# AMAN BILAIYA 2018CSB1069

import numpy as np
import skimage
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.ndimage.filters import convolve
from scipy import misc

# RGB --> grayscale
def rgb2gray(img):
    return (np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])).astype('float32')

#display image                                                                  
def view_img(img_name):
    plt.imshow(img_name, cmap = plt.get_cmap('gray'))
    plt.show() 

'''
Phase 1: Compute smoothed gradients:
a) Load an image, convert it to float format, and extract its luminance as a 2D array. Alternatively, you may also convert the input image from color to gray-scale if required.
b) Find the x and y components Fx and Fy of the image gradient after smoothing with a Gaussian (for the Gaussian, you can use σ = 1). There are two ways to go about doing this: either (A) smooth with a Gaussian using a convolution, followed by computation of the gradient; or (B) convolve with the x and y derivatives of the Gaussian.
c) At each pixel, compute the edge strength F (gradient magnitude), and the edge orientation D = atan(Fy/Fx). Include the gradient magnitude and direction images in your report.
'''
# Method to do convolution of input image with sobel filter to smoothen the image and compute x and y gradients
def convolveWithGaussianDerivative(img):
    
  gaussian_3x3fil = np.asarray([[0.1019,0.1154,0.1019],[0.1154,0.1308,0.1154],[0.1019,0.1154,0.1019]],dtype=np.float32)  
  sobel_x = np.asarray([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]], np.float32)
  Ix = ndimage.filters.convolve(img, sobel_x)  # x-dir
                                                                  
  sobel_y = np.asarray([[+1, +2, +1], [0, 0, 0], [-1, -2, -1]], np.float32)
  Iy = ndimage.filters.convolve(img, sobel_y) # y-dir
                                                                  
  theta_angle = np.arctan2(Iy, Ix) # gradient-dir D = atan(Iy/Ix)
  return Ix, Iy, theta_angle
                                                                  
'''
Phase 2: Non-maximal Suppression
Create a "thinned edge image" I[y, x] as follows:
a) For each pixel, find the direction D* in (0, π/4, π/2, 3π/4) that is closest to the orientation D at that pixel.
b) If the edge strength F[y, x] is smaller than at least one of its neighbors along the direction D*, set I[y, x] to zero, otherwise, set I[y, x] to F[y, x]. Note: Make a copy of the edge strength array before thinning, and perform comparisons on the copy, so that you are not writing to the same array that you are making comparisons on.
After thinning, your "thinned edge image" should not have thick edges any more (so edges should not be more than 1 pixel wide). Include the thinned edge output in your report.
'''
# Function to perform non maximal supppression by quantizing angles to 4 bins- 0, 45, 95 and 135 degrees
# Gradient - gradient magnitude
# theta_angle - gradient direction                                                                  
def Non_maximal_Suppression(Gradient, theta_angle):  # Idea taken from online resource mentioned in report refernces 
  
  output = np.zeros(Gradient.shape, dtype=np.float32) #thinned edge image                                                                  
  angle = np.degrees(theta_angle)                                                              

  for i in range(1, Gradient.shape[0]-1):
    for j in range(1, Gradient.shape[1]-1):
      x = 1.0
      y = 1.0
      
      if(angle[i][j] < 0):
        angle[i][j] = angle[i][j] + 180        

      #for angle 0
      if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
        x = Gradient[i, j+1]
        y = Gradient[i, j-1]
      
      #for angle pi/4 = 45
      elif (22.5 <= angle[i,j] < 67.5):
        x = Gradient[i+1, j-1]
        y = Gradient[i-1, j+1]
    
      #for angle pi/2 = 90          
      elif (67.5 <= angle[i,j] < 112.5):
        x = Gradient[i+1, j]
        y = Gradient[i-1, j]
        
      #for angle 3pi/4 = 135
      elif (112.5 <= angle[i,j] < 157.5):
        x = Gradient[i-1, j-1]
        y = Gradient[i+1, j+1]

      if (Gradient[i,j] >= x) and (Gradient[i,j] >= y):
        output[i,j] = Gradient[i,j]
      else:
        output[i,j] = 0.0
        
  return output
                                                                  

'''
Phase 3: Hysteresis thresholding:
a) Assume two thresholds: T_low, and T_high (they can be manually set or determined automatically).
b) Mark pixels as "definitely not edge" if less than T_low.
c) Mark pixels as "strong edge" if greater than T_high.
d) Mark pixels as "weak edge" if within [T_low, T_high].
e) Strong pixels are definitely part of the edge. Whether weak pixels should be included needs to be examined.
f) Only include weak pixels that are connected in a chain to a strong pixel. How to do this?
Visit pixels in chains starting from the strong pixels. For each strong pixel, recursively visit the weak pixels that are in the 8 connected neighborhood around the strong pixel, and label those also as strong (and as edge). Label as "not edge" any weak pixels that are not visited by this process. Hint: This is really a connected components algorithm, which can be solved by depth first search. Make sure to not revisit pixels when performing your depth first search!
'''

# Method for double thresholding to mark strong and weak edges
# 1.0 --> strong pixels in white    0.5 --> weak pixels in grey
def threshold(img, low_Th, high_Th):
  output = np.zeros(img.shape,dtype='float32')
  
  definitely_not_edges = np.where(img < low_Th)

  strong_edges = np.where(img >= high_Th)                                                                
  strong_rows = strong_edges[0]
  strong_cols = strong_edges[1]
  
  weak_edges =  np.where((img >= low_Th) & (img < high_Th))                                                               
  weak_rows = weak_edges[0]
  weak_cols = weak_edges[1]
                                                                  
  output[strong_rows, strong_cols] = 1.0 #strong edge pixel  value
  output[weak_rows, weak_cols] = 0.5 #weak edge pixel  value
  return output


def check_8_neighbourhood(img,row,col):
    
    dirs=[ # 8 neighbours
          np.asarray([[0,1],[0,-1]]),
          np.asarray([[-1,0],[1,0]]),
          np.asarray([[-1,-1],[1,-1]]),
          np.asarray([[-1,1],[1,1]])
        ]
    
    index = np.asarray([row,col])
    check = False
    for i in range(4):
        for j in range(2):
            move = index + dirs[i][j]                                                      
            check = check or (img[move[0]][move[1]] == 1.0) # 1.0 --> strong pixels in white    0.5 --> weak pixels in grey

    if(check==True):
        return True
    return False                                                                  
                                                                  
                                                                  
# Method to perform hysteresis to get connected edges and discarding disconnected weak edges
def perfom_hysteresis(img):
  # 1.0 --> strong pixels in white    0.5 --> weak pixels in grey
  dim1, dim2 = img.shape
  _img = np.copy(img)
                                                                  
  for i in range(1, dim1):
    for j in range(1, dim2):
      if _img[i, j] == 0.5:
        if check_8_neighbourhood(img,i,j):
          _img[i, j] = 1.0
 
  for i in range(dim1 - 1, 0, -1):
    for j in range(dim2 - 1, 0, -1):
      if _img[i, j] == 0.5:
        if check_8_neighbourhood(img,i,j):
          _img[i, j] = 1.0

  for i in range(1, dim1):
    for j in range(dim2 - 1, 0, -1):
      if _img[i, j] == 0.5:
        if check_8_neighbourhood(img,i,j):
          _img[i, j] = 1.0     
          
  for i in range(dim1-1,0,-1):
    for j in range(1,dim2):
      if _img[i, j] == 0.5:
        if check_8_neighbourhood(img,i,j):
          _img[i, j] = 1.0
        else:
          _img[i,j] = 0.0
 
  return _img

def GenerateResults(input_img):                                                                                                             
  # input_img = 'lena' #input image name
  img = mpimg.imread('data/'+input_img+'.bmp').astype('float32')/255.0 # load,read Image and convert to float
  #plt.imshow(img)

  img = rgb2gray(img)                                                                 
  # view_img(img)
  plt.imsave('Results/canny_edge/'+input_img+'_luminance.jpg',img,cmap=plt.get_cmap('gray'))

  Ix, Iy, theta_angle = convolveWithGaussianDerivative(img) #x and y gradients and angle
  Gradient = np.hypot(Ix, Iy)
  Gradient = Gradient / np.max(Gradient)

  plt.imsave('Results/canny_edge/'+input_img+'_gradient_magnitude_x.jpg',Ix,cmap=plt.get_cmap('gray'))
  #view_img(Ix)

  plt.imsave('Results/canny_edge/'+input_img+'_gradient_magnitude_y.jpg',Iy,cmap=plt.get_cmap('gray'))
  #view_img(Iy)

  plt.imsave('Results/canny_edge/'+input_img+'_gradient_magnitude.jpg',Gradient,cmap=plt.get_cmap('gray'))
  #view_img(Gradient)

  plt.imsave('Results/canny_edge/'+input_img+'_gradient_direction.jpg',theta_angle,cmap=plt.get_cmap('gray'))
  #view_img(theta_angle)

  thinned_edge_image = Non_maximal_Suppression(Gradient, theta_angle)
  plt.imsave('Results/canny_edge/'+input_img+'_thinned_edge.jpg',thinned_edge_image,cmap=plt.get_cmap('gray'))
  #view_img(thinned_edge_image)

  low_Th = 0.1 # Threshold lower bound
  high_Th = 0.25 # Threshold upper bound
  thresholded_img = threshold(thinned_edge_image, low_Th, high_Th)
  plt.imsave('Results/canny_edge/'+input_img+'_threshold.jpg',thresholded_img,cmap=plt.get_cmap('gray'))
  # view_img(thresholded_img)
                                                                  
  hystersis_img = perfom_hysteresis(thresholded_img)
  plt.imsave('Results/canny_edge/'+input_img+'_final_edges.jpg',hystersis_img,cmap=plt.get_cmap('gray'))
  # view_img(hystersis_img) 

img_Names = ['bicycle','bird','dog','einstein','leena','plane','toy_image']
GenerateResults(img_Names[1])