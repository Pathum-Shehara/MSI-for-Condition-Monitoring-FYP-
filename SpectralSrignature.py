import os
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def MakeMainMatrix():
    pwd_main = os.getcwd()
    folders = glob(pwd_main + "/*/", recursive=True)
    numOfImgs = 14
    numberOfSamples = 15
    numOfLevels = 5
    mainMatrix3D = np.ones((numOfLevels, numberOfSamples, 14))
    mainMatCount = 0
    for f in folders:
        subFolders = glob(f + "/*/", recursive=True)
        avgMatrix = np.ones((1, numOfImgs))
        for sf in subFolders:
            file_arr = os.listdir(sf)

            avgImg = []
            for img in file_arr:
                theFile = os.path.join(sf, img)
                imgMatrix = cv2.imread(theFile, cv2.IMREAD_GRAYSCALE)
                avg = np.sum(np.sum(imgMatrix))
                avgImg.append(avg / 10000)

            avgMatrix = np.vstack((avgMatrix, np.array(avgImg)))

        mainMatrix3D[mainMatCount, :, :] = avgMatrix[1:, :]
        mainMatCount += 1

    return mainMatrix3D


def plotSignature():
    mainMatrix = MakeMainMatrix()
    numAflLevel = 5
    colorArray = ['orange', 'blue', 'red','green','pink']

    plt.figure(figsize=(8, 6))  # set the size of the figure

    for i in range(0, numAflLevel):
        imgMatrix = mainMatrix[i, :, :]
        numRows = imgMatrix.shape[0]
        for j in range(0, numRows):
            plt.plot([0, 365, 405, 473, 530, 575, 621, 660, 735, 770, 830, 850, 890, 940], imgMatrix[j],
                     color=colorArray[i], linestyle='-',  # Set linestyle to solid
                     linewidth=1)

    # Create custom legend lines with solid linestyle
    legend_lines = [
        Line2D([0], [0], color='orange', linestyle='-', lw=2, label='0%'),
        Line2D([0], [0], color='blue', linestyle='-', lw=2, label='25%'),
        Line2D([0], [0], color='red', linestyle='-', lw=2, label='50%'),
        Line2D([0], [0], color='green', linestyle='-', lw=2, label='75%'),
        Line2D([0], [0], color='pink', linestyle='-', lw=2, label='100%'),
    ]

    plt.legend(handles=legend_lines, fontsize=11, loc='upper left')
    plt.grid(linestyle='--')  # Set grid lines as dashed
    plt.title('Mean Spectral Signature of Coke and Fanta Mixture', fontsize=13)
    plt.xlabel('Wavelength/(nm)', fontsize=11)
    plt.ylabel('Mean Intensity', fontsize=11)

    # Set the x-axis limits to compress the plot
    plt.xlim(300, 940)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    #plt.savefig('Spectral_signature_CokenFanta.png', dpi=300, bbox_inches='tight')
    plt.show()


plotSignature()
