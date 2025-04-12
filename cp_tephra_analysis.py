#
# The script performs a change-point (cp) analysis on tephra timeseries presented in 
# this paper using the library Rbeast from Zhao et al., (2019).
#

# Import modules
import numpy as np
import Rbeast as rb
import matplotlib.pyplot as plt
import pandas as pd

# Define variables
ORD_MAGNITUDE = 10**3
test_cp_record = None  # For Rbeast library testing.


# ---------------------------------
# a) Load and show first 10 data rows from .csv dataset.
#    Rows -> Time series
#    Columns -> Weighted tephra volume

# b) Convert into Numpy array and transpose it to use in Rbeast.

print("-"*70)
print("Read the .csv dataset... ")
tephra_df = pd.read_csv("input_files/aggregated_resampled_timeseries_WEIGHTED_1000y.csv",
                        header = None)

print("First 10 elements:\n")
print("-"*70)
print(tephra_df.head())
print("\n")
print("-"*70)
print("Convert .csv record into Numpy array...")
tephra_df_numpy = tephra_df.to_numpy()
print("-"*70)
print("\n")
print(f"array shape: {tephra_df_numpy.shape}\n")
print("\n")
print("-"*70)
print("\nTranspose the matrix...")
tephra_df_numpy = tephra_df_numpy.T
print(f"Transposed array shape: {tephra_df_numpy.shape}")
print("\n")

# ---------------------------------
# a) Create a time array between 1 - 0 million years (Ma). data-point spacing 1 thousand year (ka)
time = np.arange(10**6, -1, -10**3)

# ---------------------------------
# +++ N.B.! This section is for testing cp analysis on a single tephra record +++. 
# 

if test_cp_record:
    column_rec = 0
    o = rb.beast(tephra_df_numpy[:,column_rec], start = 0, deltat=1, season = "none")
    cp = o.trend.cpOccPr
    print(len(cp))
    print(cp)
    plt.plot(time/ORD_MAGNITUDE, cp)
    plt.xlabel("Time [Ka]")
    plt.ylabel("Posterior probability")
    plt.xlim([min(time/ORD_MAGNITUDE), max(time/ORD_MAGNITUDE)])
    plt.title(f"First tephra record - Change point probability")
    plt.show()

    
# ---------------------------------
# a) Loop through the entire records (50,000 tephra records).
#    At every iteration, store the cp posterior probability of each time step in an array

print("-"*70)
cp_occ = []
for t in range(50000):
    print(f"---> Dataset N° {t}/{tephra_df_numpy.shape[1]}")  # Track process advancement 
    o = rb.beast(tephra_df_numpy[:,t], start = 0, deltat = "1000 year", 
                    mcmc_seed = 1, season = "none")
    cp = o.trend.cpOccPr
    cp = cp.reshape(-1,1)
    cp = cp[:,0]
    cp_occ.append(cp)

# ---------------------------------
# a) Retrieve the median, 0.25, and 0.75 percentiles
# b) Retrieve the Interquartile range (IQR)
# c) Plot the median result.
print("-"*70)
print("\n")
print("Calculate the Median, 0.25 and 0.75 percentiles, and the IQR range...")
cp_occ_arr = np.array(cp_occ)    
medians = np.median(cp_occ_arr, axis = 0)
percentile_25 = np.percentile(cp_occ_arr, 25, axis= 0)    
percentile_75 = np.percentile(cp_occ_arr, 75, axis = 0)
iqr = percentile_75 - percentile_25
print("\n")
print("-"*70)


plt.plot(time/ORD_MAGNITUDE, medians, c = "darkred", lw = 0.7)
plt.fill_between(time/ORD_MAGNITUDE, percentile_25, percentile_75, color = "orange", alpha = 0.4)
plt.xlabel("Time [Ka]")
plt.ylabel("Posterior probability")
plt.xlim([min(time/ORD_MAGNITUDE), max(time/ORD_MAGNITUDE)])
plt.title(f"Change point distribution (median - IQR)")
plt.show()

# ---------------------------------




