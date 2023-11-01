import os
import cv2
import random
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

numOfImgs = 14
numberOfSamples =8
numOfLevels = 9
pixelSize = 10
# Picking part of the samples for testing from each adulteration level
numberOfTestSamples = 2
# For the Median Filter
kernel_size = 3

def MakeMainMatrix_DarkCurrentImage():
    pwd_main = os.getcwd()
    folders = glob(pwd_main + "/*/", recursive = True)
    folders = sorted(folders)
    #print(folders)
    
    trainDataMatrix = np.empty((0,13))
    testDataMatrix = np.empty((0,13))
    
    for f in folders:
        subFolders = glob(f + "/*/", recursive = True)
        subFolders = sorted(subFolders)
        # Extract the names of the final folders
        #subFolderNames = [os.path.basename(os.path.normpath(folder)) for folder in subFolders]
        # Pick sub folders for Testing
        testFolders = random.sample(subFolders, numberOfTestSamples)
        trainFolders = [folder for folder in subFolders if folder not in testFolders]

        testimageMatrixforOneLevel = np.ones((0,13))
        trainimageMatrixforOneLevel = np.ones((0,13))
##        print("Randomly selected folders:")
##        for folder in testFolders:
##            print(folder)
##
##        print("\nRemaining folders:")
##        for folder in trainFolders:
##            print(folder)
        
        for sf in testFolders:
            file_arr = os.listdir(sf)

            imageMatrixForOneSample = np.empty((pixelSize**2,0))
            darkImage = file_arr[0]
            theDarkFile = os.path.join(sf, darkImage)
            darkImageMatrix = cv2.imread(theDarkFile, cv2.IMREAD_GRAYSCALE)
            for img in file_arr[1:]:
                theFile = os.path.join(sf , img)
                imgMatrix = cv2.imread(theFile , cv2.IMREAD_GRAYSCALE)
                darkReductedImage = cv2.subtract(imgMatrix,darkImageMatrix)

                #Median Filter
                #filtered_image = cv2.medianBlur(darkReductedImage, kernel_size)

                #Adaptive Wiener Filter
                # Estimate the noise variance in the image
                #noise_variance = restoration.estimate_sigma(darkReductedImage)
                # Apply the adaptive Wiener filter with the estimated noise variance
                #filtered_image = restoration.wiener(darkReductedImage, noise_variance, window_size, k)

                # Apply the Adaptive Wiener Filter
                filtered_image = wiener_filter(darkReductedImage,3)
                
                # Apply the bilateral filter
                #filtered_image = cv2.bilateralFilter(darkReductedImage, 9, 75, 75)

                superImage = getSuperPixel(filtered_image, pixelSize)
                columnMatrix = superImage.reshape(pixelSize**2, 1)
                imageMatrixForOneSample = np.hstack((imageMatrixForOneSample, columnMatrix))

            print('Test')
            testimageMatrixforOneLevel = np.vstack((testimageMatrixforOneLevel,imageMatrixForOneSample))

        testDataMatrix = np.vstack((testDataMatrix,testimageMatrixforOneLevel))

        for sf in trainFolders:
            file_arr = os.listdir(sf)

            imageMatrixForOneSample = np.empty((pixelSize**2,0))
            darkImage = file_arr[0]
            theDarkFile = os.path.join(sf, darkImage)
            darkImageMatrix = cv2.imread(theDarkFile, cv2.IMREAD_GRAYSCALE)
            for img in file_arr[1:]:
                theFile = os.path.join(sf , img)
                imgMatrix = cv2.imread(theFile , cv2.IMREAD_GRAYSCALE)
                darkReductedImage = cv2.subtract(imgMatrix,darkImageMatrix)

                #Median Filter
                #filtered_image = cv2.medianBlur(darkReductedImage, kernel_size)

                #Adaptive Wiener Filter
                # Estimate the noise variance in the image
                #noise_variance = restoration.estimate_sigma(darkReductedImage)
                # Apply the adaptive Wiener filter with the estimated noise variance
                #filtered_image = restoration.wiener(darkReductedImage, noise_variance, window_size, k)

                # Apply the Adaptive Wiener Filter
                filtered_image = wiener_filter(darkReductedImage,3)
                #print("***")                
                # Apply the bilateral filter
                #filtered_image = cv2.bilateralFilter(darkReductedImage, 9, 75, 75)

                superImage = getSuperPixel(filtered_image, pixelSize)
                columnMatrix = superImage.reshape(pixelSize**2, 1)
                imageMatrixForOneSample = np.hstack((imageMatrixForOneSample, columnMatrix))

            print('Train')
            trainimageMatrixforOneLevel = np.vstack((trainimageMatrixforOneLevel,imageMatrixForOneSample))
        
        trainDataMatrix = np.vstack((trainDataMatrix,trainimageMatrixforOneLevel))
    #print(testDataMatrix.shape)
    #print(trainDataMatrix.shape)
    return testDataMatrix, trainDataMatrix

def getSuperPixel(img, size):
    
    shp = img.shape
    hight = shp[0]
    width = shp[1]

    superImage = np.zeros( (round(hight/size) , round(width/size)) )

    for i in range(0,round(width/size)):
        for j in range(0,round(hight/size)):
            piece = img[i*size:size*(i+2)-size, j*size:size*(j+2)-size]
            superImage[i,j] = np.sum(piece)/(size**2)

    return superImage

def trainLabels():
    trainLabels = np.zeros(((pixelSize**2)*(numberOfSamples-numberOfTestSamples)*numOfLevels,1))
    for i in range((pixelSize**2)*numberOfSamples):
        trainLabels[i*(pixelSize**2)*(numberOfSamples-numberOfTestSamples):(i+1)*(pixelSize**2)*(numberOfSamples-numberOfTestSamples),0] = 5*i

    return trainLabels

def testLabels():
    testLabels = np.zeros(((pixelSize**2)*(numberOfTestSamples)*numOfLevels,1))
    for i in range((pixelSize**2)*numberOfSamples):
        testLabels[i*(pixelSize**2)*(numberOfTestSamples):(i+1)*(pixelSize**2)*(numberOfTestSamples),0] = 5*i

    return testLabels

def makingDataSet():
    testDataMatrix, trainDataMatrix = MakeMainMatrix_DarkCurrentImage()
    testLabelsMatrix = testLabels()
    trainLabelsMatrix = trainLabels()
    testDataSet = np.append(testDataMatrix, testLabelsMatrix, axis = 1)
    trainDataSet = np.append(trainDataMatrix, trainLabelsMatrix, axis = 1)
    return testDataSet, trainDataSet

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
            mean[i][j] = mean_sum/((2*w+1)**2)
            #print(mean[i][j])
            variance[i][j] = variance_sum/((2*w+1)**2) - (mean[i][j])**2
            if variance[i][j] == 0:
                variance[i][j] = 1e-10
    #print(mean)
    #print(variance)
    white_noise = sum(sum(row) for row in variance)
    white_noise = white_noise/(padded_image.shape[0]*padded_image.shape[1])

    for i in range(padded_image.shape[1]):
        for j in range(padded_image.shape[0]):
            filtered_image[i][j] = mean[i][j] + (padded_image[i][j] - mean[i][j])*( variance[i][j] - white_noise ) / variance[i][j]
    
    
    return filtered_image[1:101, 1:101]


#matrixCube()
#yMatrix()
testDataSet, trainDataSet = makingDataSet()
np.savetxt("Test 1.csv", testDataSet, delimiter=',')
np.savetxt("Train 1.csv", trainDataSet, delimiter=',')


