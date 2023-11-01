import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, exp, Mul , pi, sqrt, DiracDelta

from sympy.utilities.lambdify import lambdify
import pandas as pd
import csv
from scipy.stats import norm
import math



reference_level = 0
numberOfSamples = 8
numberOfLevels = 9

csv_file = '4.Palm Oil Transmittance Dark Current Reducted Bilateral Filtered Data Set (Super Pixel) with PCA.csv'
dataset = pd.read_csv(csv_file)
datasetNP = dataset.iloc[:, :-1].values
#print(dataset)

labels = sorted(set(dataset['Labels'].unique()))
#print(labels)

#print(datasetNP.shape)


def separate_data_by_level(df, level): #Used the Dataframe
    levelData = df[df['Labels'] == level]
    return levelData


def separate_data_by_sample(df):
    for i in range (numberOfSamples):
        return dataset.iloc[i*100:i*100+100,:]
        

#print(separate_data_by_level(dataset, 0))


def get_meanVector(df):   #Used Dataframe
    meanVector = df.groupby('Labels').mean().values.T
    return meanVector


#`print(get_meanVector(dataset))
#print(get_meanVector(dataset).type)


def get_CovarianceMatrix(refData,meanVector): #used Numpy
    dfNP = refData.iloc[:, :-1].values
    dfNP = dfNP - meanVector.T
    covarianceMatrix = np.dot(dfNP.T, dfNP)
    #covarianceMatrix = np.cov(df.iloc[:, :-1].T)
    
    return covarianceMatrix

#print(get_CovarianceMatrix(dataset))
#print(get_CovarianceMatrix(dataset).shape)


def get_bhattacharyyaDistance(levelData):
    
    #print(ref_meanVector)
    level_meanVector = get_meanVector(levelData)
    #print(level_meanVector.shape)
    
    #print(ref_CovMatrix)
    level_CovMatrix = get_CovarianceMatrix(levelData,level_meanVector)
    
    #print(level_CovMatrix.shape)
    
    mean_diff = ref_meanVector - level_meanVector
    mean_cov = 0.5*(ref_CovMatrix + level_CovMatrix)
    det_mean_cov = np.linalg.det(mean_cov)
    ref_Cov_Det = np.linalg.det(ref_CovMatrix)
    level_Cov_Det = np.linalg.det(level_CovMatrix)
    
    BhattDistance = (1/8)*(mean_diff.T).dot(np.linalg.inv(mean_cov)).dot(mean_diff) + (0.5)*(np.log(det_mean_cov/np.sqrt(abs(ref_Cov_Det)*abs(level_Cov_Det))))
    #print(BhattDistance)
    return np.sum(BhattDistance)


def get_JMdistace(BhattDistance):
    return np.sqrt(2*(1-math.exp(-BhattDistance)))


refData = separate_data_by_level(dataset, 0)
ref_meanVector = get_meanVector(refData)
ref_CovMatrix = get_CovarianceMatrix(refData,ref_meanVector)
#print(refData.type)
#levelData = separate_data_by_level(dataset, 25)



#levelDataNP = levelData.values

#print(levelDataNP)

#meanVector = get_meanVector(refData).iloc[:, 0]

#meanVectorNP = meanVector.values
#print(meanVectorNP)


#print(refDataVal.shape)
#covMat = get_CovarianceMatrix(levelDataNP,meanVectorNP)
#print(covMat)


#BhattDistance=get_bhattacharyyaDistance(refData, levelData)
#print(get_JMdistace(BhattDistance))

#print(get_CovarianceMatrix(refData))

#print(get_CovarianceMatrix(refData).shape)
#print(get_CovarianceMatrix(levelData))
#print(get_CovarianceMatrix(levelData).shape)

BhattachryyaDisMat = np.empty((numberOfSamples,numberOfLevels))
JMDisMat = np.empty((numberOfSamples,numberOfLevels))

levelData = separate_data_by_level(dataset, 10)
sampleDataSet = levelData.iloc[2*100:2*100+100,:]
BhattDistance=get_bhattacharyyaDistance(sampleDataSet)
#print(BhattDistance)

for i,level in enumerate(range (0,numberOfLevels*5,5)):
    #(level)
    levelData = separate_data_by_level(dataset, level)
    for j in range (numberOfSamples):
        sampleDataSet = levelData.iloc[j*100:j*100+100,:]
        BhattDistance=get_bhattacharyyaDistance(sampleDataSet)
        BhattachryyaDisMat[j][i] = BhattDistance
        #print(BhattachryyaDisMat[j][i])
        JM_Distance = get_JMdistace(BhattDistance)
        JMDisMat[j][i] = JM_Distance
        #print(JM_Distance)

#print(BhattachryyaDisMat)
#print(BhattachryyaDisMat.shape)
#print(JMDisMat)
#print(JMDisMat.shape)



# Reshape the matrix into a 1D array
#flattened_data = JMDisMat.ravel()
flattened_data = BhattachryyaDisMat.ravel()

# Create x-values corresponding to columns of the matrix, repeated for each row
percentages = np.arange(0, 41, 5)
x_values = np.tile(percentages, 8)


# Reverse the x_values array to reverse the order on the x-axis
#x_values = x_values[::-1]


# Plot the scatter plot
plt.scatter(x_values, flattened_data)

# Add labels and title
plt.xlabel("Adulteration level(%)")
plt.ylabel(" Bhattacharyya distance")
#plt.ylabel(" JM distance")
#plt.title(" The variation of the mean  Bhattacharyya distance with the adulteration level")



#Curve fitting
# Perform polynomial regression of degree 3 (you can change the degree as needed)
degree = 1
coefficients = np.polyfit(x_values, flattened_data, degree)

# Generate the curve using the polynomial coefficients
x_fit = np.linspace(x_values.min(), x_values.max(), 100)
y_fit = np.polyval(coefficients, x_fit)

# Plot the fitted curve
plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')



# Calculate the R-squared value
y_mean = np.mean(flattened_data)
y_pred = np.polyval(coefficients, x_values)
ss_tot = np.sum((flattened_data - y_mean) ** 2)
ss_res = np.sum((flattened_data - y_pred) ** 2)
r_squared = 1 - (ss_res / ss_tot)
r_squared = round(r_squared, 4)

# Calculate Root Mean Square Error (RMSE)
rmse = np.sqrt(np.mean((flattened_data - y_pred) ** 2))
# Calculate RSS
rss = np.sum((flattened_data - y_pred) ** 2)


# Define a function to calculate AIC
def calculate_aic(n, k, rss):
    return n * np.log(rss / n) + 2 * k

# Calculate AIC value
n = len(flattened_data)
k = degree + 1  # number of parameters (degree + intercept)
aic = calculate_aic(n, k, rss)

# Output the fitted polynomial equation and R-squared value
equation = f"Fitted Equation: y = {' + '.join([f'{coeff:.4f} * x^{degree - i}' for i, coeff in enumerate(coefficients)])}"
print(equation)
print("R^2 Value:", r_squared)
print("RMSE:", rmse)
print("AIC:", aic)



plt.legend()
# Show the plot
plt.show()



