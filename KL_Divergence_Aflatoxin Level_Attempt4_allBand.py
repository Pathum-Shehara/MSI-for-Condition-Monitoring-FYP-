import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import log

# Calculate the kl divergence
def kl_divergence(p, q):
    return sum(p[i] * log((p[i] + 1e-10) / (q[i] + 1e-10)) for i in range(len(p)))


numOfLevel = 9
numOfSamples = 15

# Load the dataset from the CSV file
dataset = pd.read_csv('Aflatoxin level detection Reflectance Dark Current Reducted Bilateral Filtered Data Set (Super Pixel) Dataset with LDA.csv')

# Extract the feature vectors and labels
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

kl_divergence_values = np.zeros([numOfLevel * numOfSamples])
print(kl_divergence_values.shape)

# Iterate through all columns (x1 to x13)
for band in range(4):
    kl_divergence_sum = np.zeros([numOfLevel * numOfSamples])
    
    for i in range(numOfLevel * numOfSamples):
        reference = features[0:100, band]
        level = features[i:i+100, band]
     
        # Calculate the histogram
        hist_ref, bin_edges_ref = np.histogram(reference, bins=15)
        hist_level, bin_edges_level = np.histogram(level, bins=15)
    
        # Calculate the probabilities by normalizing the histogram
        total_count = len(reference)
        p = hist_ref / total_count
        q = hist_level / total_count
    
        # calculate (P || Q)
        kl_pq = kl_divergence(p, q)
        kl_divergence_sum[i % 100] = kl_pq
    
    # Sum the KL divergence values for this band
    kl_divergence_values += kl_divergence_sum

# x values for the plot
values = [0, 5, 10, 15, 20, 25, 30, 35, 40]
x = []
for i in range(numOfLevel):
    x.extend([values[i % len(values)]] * numOfSamples)

# Fit a polynomial curve to the summed KL divergence values
degree = 2  # Adjust the degree as needed
coefficients = np.polyfit(x, kl_divergence_values, degree)
polynomial = np.poly1d(coefficients)

# Generate points for the best fit curve
x_fit = np.linspace(min(x), max(x), 40)
y_fit = polynomial(x_fit)

plt.figure(figsize=(8, 6))
plt.scatter(x, kl_divergence_values, marker='o', label='Samples')
plt.plot(x_fit, y_fit, color='red', label=f'Fitted Curve: y = {" + ".join([f"{coeff:.4f} * x^{degree - i}" for i, coeff in enumerate(coefficients)])}')
plt.xlabel('Aflatoxin Level (%)')
plt.ylabel('KL Divergence')
plt.legend()
plt.grid(True)
plt.title('Aflatoxin Level vs KL divergence')
plt.show()

# Calculate the R-squared value
y_mean = np.mean(kl_divergence_values)
y_pred = np.polyval(coefficients, x)
ss_tot = np.sum((kl_divergence_values - y_mean) ** 2)
ss_res = np.sum((kl_divergence_values - y_pred) ** 2)
r_squared = 1 - (ss_res / ss_tot)
r_squared = round(r_squared, 4)

# Calculate Root Mean Square Error (RMSE)
rmse = np.sqrt(np.mean((kl_divergence_values - y_pred) ** 2))
# Calculate RSS
rss = np.sum((kl_divergence_values - y_pred) ** 2)

# Define a function to calculate AIC
def calculate_aic(n, k, rss):
    return n * np.log(rss / n) + 2 * k

# Calculate AIC value
n = len(kl_divergence_values)
k = degree + 1  # number of parameters (degree + intercept)
aic = calculate_aic(n, k, rss)

# Output the fitted polynomial equation and R-squared value
equation = f"Fitted Equation: y = {' + '.join([f'{coeff:.4f} * x^{degree - i}' for i, coeff in enumerate(coefficients)])}"
print(equation)
print("R^2 Value:", r_squared)
print("RMSE:", rmse)
print("AIC:", aic)
