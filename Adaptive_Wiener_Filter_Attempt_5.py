import os
import cv2
import numpy as np
import time

#start_time = time.time()

def wiener_filter(image, w):   #w = window size
    desired_width = 102
    desired_height = 102

    padded_image = np.zeros((desired_height, desired_width))
    mean = np.empty([padded_image.shape[1], padded_image.shape[0]])
    variance = np.empty([padded_image.shape[1], padded_image.shape[0]])

    padded_image[1:101, 1:101] = image

    # Repeat the edge row and column values
    padded_image[0, :] = padded_image[1, :]
    padded_image[101, :] = padded_image[100, :]
    padded_image[:, 0] = padded_image[:, 1]
    padded_image[:, 101] = padded_image[:, 100]
    
    filtered_image = np.empty([padded_image.shape[1], padded_image.shape[0]])
    
     #print(image.shape)
    for i in range(padded_image.shape[1]):
        for j in range(padded_image.shape[0]):
            mean_sum = 0
            variance_sum = 0
            for k in range(-(w-1),w):
                if ((i+k) < 0) or ((i+k) >= padded_image.shape[1]):
                    mean_sum += 0
                    variance_sum += 0
                else:
                    for l in range(-(w-1),w):
                        if ((j+l) < 0) or ((j+l) >= padded_image.shape[0]):
                            mean_sum += 0
                            variance_sum += 0
                        else:
                            mean_sum += padded_image[i+k][j+l]
                            variance_sum += (padded_image[i+k][j+l])**2
            #print(mean_sum)
            mean[i][j] = mean_sum/(2*w+1)**2
            #print(mean[i][j])
            variance[i][j] = variance_sum/(2*w+1)**2 - (mean[i][j])**2
    #print(mean)
    print(variance)
    white_noise = sum(sum(row) for row in variance)
    white_noise = white_noise/(padded_image.shape[0]*padded_image.shape[1])

    for i in range(padded_image.shape[1]):
        for j in range(padded_image.shape[0]):
            filtered_image[i][j] = mean[i][j] + (padded_image[i][j] - mean[i][j])*( variance[i][j] - white_noise ) / variance[i][j]
    
    
    return filtered_image[1:101, 1:101]
    
  
image_path = '660nm.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#print(image)
filtered_image = wiener_filter(image,3)
#print(filtered_image)
#print(filtered_image.shape)
#print(variance) 
output_filename = image_path.replace('.png', '_wiener_filtered_padded.png')
cv2.imwrite(output_filename, filtered_image)
#end_time = time.time()

# Calculate and display the running time
#running_time = end_time - start_time
#print(f"Running time: {running_time} seconds")

