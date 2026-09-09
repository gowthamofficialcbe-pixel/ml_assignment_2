import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Cluster Purity Function

def cluster_purity(cluster_labels, true_labels):

    total_correct = 0 

    for cluster in np.unique(cluster_labels): # For every cluster

        labels_in_cluster = true_labels[cluster_labels == cluster]  # Ground-truth labels belonging to this cluster
        most_common_label = np.bincount(labels_in_cluster).argmax() # Most frequent ground-truth symbol
        total_correct += np.sum(labels_in_cluster == most_common_label) # Number of correctly assigned in this cluster

    purity = total_correct / len(true_labels) # Overall purity
    return purity

true_symbol_id = np.repeat(np.arange(16),200) # Creates 3200 ground-truth labels where each of the 16 symbols is assigned 200 samples

# 16-QAM Constellation over AWGN

np.random.seed(67) # Makes all NumPy random operations reproducible

samples_per_point = 200  # Setting the samples per constellation point
SNR_dB = [0, 5, 10, 15, 20, 25, 30] # Mentioning the SNR values
levels = np.array([-3, -1, 1, 3]) # 16 - QAM constellation levels
levels = levels / np.sqrt(10) # Normalizing it to make average symbol energy Es =1

tx_I, tx_Q = np.meshgrid(levels, levels) # Generating inphase and quadrature parts into matrix form
tx_points = tx_I.flatten() + 1j * tx_Q.flatten() # Making it into complex form
tx = np.repeat(tx_points, samples_per_point) # Repeat every constellation point 200 times
N = len(tx) # Total number of transmitted symbols

print("Number of constellation points =", len(tx_points))
print("Samples per constellation point =", samples_per_point)
print("Total transmitted samples =", N)

Es = np.mean(np.abs(tx)**2) # Average symbol energy Es = average of squares of absolute values of tx
print("Average symbol energy Es =", Es)

all_data = [] # Store data with respect to each SNR values

# Plot constellation for each SNR

fig, axes = plt.subplots(2, 4, figsize=(16, 8)) # Creates 8 subplots (2x4) with 4 on each row with appropriate dimensions
axes = axes.flatten() # Converts 2-D matrix into 1-D array

for k, snr_db in enumerate(SNR_dB):

    EbN0 = 10 ** (snr_db / 10) # Converts the SNR from dB to linear scale
    Eb = Es / 4 # Since its 16-QAM, it has 4 bits per symbol
    N0 = Eb / EbN0 # Calculating noise from SNR
    noise_std = np.sqrt(N0 / 2) # Standard deviation of each noise component acts as normalizing factor

    noise = noise_std * (np.random.randn(N) + 1j * np.random.randn(N)) #  Generating complex AWGN
    
    rx = tx + noise # Received signal

    # Feature Extraction
    
    RxI = np.real(rx) # In-phase part of the received signal
    RxQ = np.imag(rx) # Quadrature part of the received signal

    amplitude = np.sqrt(RxI**2 + RxQ**2) # Instantaneous amplitude

    phase = np.arctan2(RxQ, RxI) # Instantaneous phase

    # Create DataFrame for this SNR
    
    temp_df = pd.DataFrame({'RxI': RxI,'RxQ': RxQ,'Amplitude': amplitude,'Phase': phase,'SNR': snr_db}) # Creating a temporary dataframe
    all_data.append(temp_df) # Storing this dataframe as a matrix form

    # Plot
    
    ax = axes[k] # Constellation diagram for each SNR in each subplot

    ax.scatter(rx.real,rx.imag,s=5,alpha=0.35,label="Received") # Received signal points is plotted

    ax.scatter(tx_points.real,tx_points.imag,color="red",marker="x",s=60,linewidths=2,label="Ideal") # Transmitted signal points is plotted

    ax.set_title(f"Eb/N0 = {snr_db} dB") # Title for each SNR
    ax.set_xlabel("In-phase (I)") # Mentioning the in-phase part
    ax.set_ylabel("Quadrature (Q)") # Mentioning the quadrature part
    ax.grid(True) # Enables the grid
    ax.axis("equal") # To ensure same physical scale used for both axes



# Combine all SNR data into one DataFrame

df = pd.concat(all_data, ignore_index=True) # Creating a complete dataframe

print("\nComplete dataset:")
print(df) # Displaying all information in dataframe

print("\nDataset shape:")
print(df.shape) # Gives no. of rows and columns

fig.delaxes(axes[-1]) # Used to remove unused subplot as we need only 7 SNRs

fig.suptitle("16-QAM over AWGN Channel " "(200 Samples per Constellation Point, Es = 1)", fontsize=16, fontweight="bold") #Title of the plot

plt.tight_layout() # Adjusts the spacing between plots 


# Isolate SNR = 25 dB

df_25 = df[df['SNR'] == 25].copy() # Copying the information related to only SNR = 25 dB

print("\nSNR = 25 dB dataset:")
print(df_25) # Displaying the information related to only SNR = 25 dB

print("\nNumber of samples at SNR = 25 dB:")
print(len(df_25)) # Displaying the number of samples at SNR = 25 dB


# K-MEANS using Cartesian Coordinates

X = df_25[['RxI', 'RxQ']].values # Use only RxI and RxQ

# Store results
kmeans_models = {}
inertia = []
silhouette_scores = []

K_values = range(2, 21) # K = 2 to K = 20
for K in range(2, 21):

    kmeans = KMeans(n_clusters=K,init='k-means++',n_init=10,random_state=42) # Uses K-Means algorithm for each of K clusters                                                                       
    a=kmeans.fit_predict(X) # Fit K-means by learning centroids and assigning nearest centroid cluster for each sample
    kmeans_models[K] = kmeans # Stores model

    if K==16:
        kmeans_16 = KMeans(n_clusters=K,init='k-means++',n_init=10,random_state=42) # Uses K-Means algorithm for each of K clusters
        cluster_labels = a; # Cluster assignment for every received sample
        centroids = kmeans.cluster_centers_
        
        
    inertia.append(kmeans.inertia_) # Stores inertia

    score = silhouette_score(X, a) # Calculates Silhouette coefficient with respect to received points and predicted centroids
    silhouette_scores.append(score) # Stores Silhouette coefficients

    print(f"K = {K:2d}, " f"Inertia = {kmeans.inertia_:.4f}" f" ,Silhouette = {score:.4f}")


# Two side-by-side subplots

fig, ax = plt.subplots(1, 2, figsize=(14, 5)) # Plotting 2 subplots in same figure


# Subplot 1: Inertia vs K

ax[0].plot(K_values,inertia,marker='o') # Plotting Inertia vs K

ax[0].set_xlabel("Number of clusters (K)") # Title of x-axis
ax[0].set_ylabel("Within-Cluster Sum of Squares (Inertia)") # Title of y-axis
ax[0].set_title("Inertia vs. K") # Tile of subplot 1
ax[0].set_xticks(list(K_values)) # Sets the position of the tick marks on the x-axis
ax[0].grid(True) # Enables the grid


# Subplot 2: Silhouette Coefficient VS K

ax[1].plot(K_values,silhouette_scores,marker='o') # Plotting Silhouette Coefficients vs K

ax[1].set_xlabel("Number of clusters (K)") # Title of x-axis
ax[1].set_ylabel("Silhouette Coefficient") # Title of y-axis
ax[1].set_title("Silhouette Coefficient vs. K") # Tile of subplot 2
ax[1].set_xticks(list(K_values)) # Sets the position of the tick marks on the x-axis
ax[1].grid(True) # Enables the grid


# Final Figure

fig.suptitle("K-means Clustering Evaluation at SNR = 25 dB", fontsize=15, fontweight='bold') #Title of the plot

plt.tight_layout() # Adjusts the spacing between plots

print("\n16 Learned Cluster Centroids:") # Print centroid coordinates
for i, centroid in enumerate(centroids):
    print(f"Cluster {i}: " f"I = {centroid[0]:.4f}, " f"Q = {centroid[1]:.4f}")

# 2D Scatter Plot

plt.figure(figsize=(9, 7)) # Plotting 1 figure 

plt.scatter(X[:, 0],X[:, 1],c=cluster_labels,cmap='tab20',s=10,alpha=0.5) # Received samples colored according to cluster index

plt.scatter(centroids[:, 0],centroids[:, 1],marker='X',c='black',s=180,edgecolors='white',linewidths=1.5,label='K-means Centroids') # Overlay the 16 learned centroids

# Labels and title
plt.xlabel("In-phase (I)") # Title of x-axis
plt.ylabel("Quadrature (Q)") # Title of y-axis
plt.title("K-means Clustering of 16-QAM at SNR = 25 dB") # Title of plot

plt.grid(True) # Enables the grid
plt.axis('equal') # Makes the x-axis and y-axis of the same scale.
plt.legend() # Differentiates between received symbols and centroids.

# Received Features at SNR = 25 dB

rxI = df_25['RxI'].values # Getting the in-phase part of received signal with SNR = 25 dB
rxQ = df_25['RxQ'].values # Getting the quadrature part of received signal with SNR = 25 dB

r = df_25['Amplitude'].values # Getting the instantaneous amplitude of received signal with SNR = 25 dB
theta = df_25['Phase'].values # Getting the instantaneous phase of received signal with SNR = 25 dB

# Feature Set 1: (rx I, rx Q)

X1 = np.column_stack((rxI, rxQ)) # Gets the received in-phase and quadrature parts into 2 columns

labels1 = kmeans_16.fit_predict(X1) # Cluster assignment for every received sample

purity1 = cluster_purity(labels1,true_symbol_id) #Calculating cluster purity between predicted and original labels for feature set 1


# Feature Set 2: (r, theta)

X2 = np.column_stack((r, theta)) # Gets the received instantaneous amplitude and phase into 2 columns

labels2 = kmeans_16.fit_predict(X2) # Cluster assignment for every received sample

purity2 = cluster_purity(labels2,true_symbol_id) #Calculating cluster purity between predicted and original labels for feature set 2


# Feature Set 3: (rx I, rx Q, r, theta)

X3 = np.column_stack((rxI, rxQ, r, theta)) #Gets the received in-phase parts, quadrature parts, instantaneous amplitude and phase into 4 columns

scaler = StandardScaler() # Applying StandardScaler
X3_scaled = scaler.fit_transform(X3) #Scales features with different numerical scales between (rx I,rx Q) and (r, theta)

labels3 = kmeans_16.fit_predict(X3_scaled) # Cluster assignment for every received and scaled sample

purity3 = cluster_purity(labels3,true_symbol_id) #Calculating cluster purity between predicted and original labels for feature set 3


# Final Results

print("\nCLUSTER PURITY RESULTS - SNR = 25 dB")

print(f"Feature Set 1 (RxI, RxQ)       : {purity1:.4f}") # Displays cluster purity of feature set 1
print(f"Feature Set 2 (r, theta)       : {purity2:.4f}") # Displays cluster purity of feature set 2
print(f"Feature Set 3 (RxI, RxQ, r, theta) : {purity3:.4f}") # Displays cluster purity of feature set 3


# Feature Set with highest Cluster Purity

purities = [purity1, purity2, purity3] # Cluster purity of all 3 feature sets

#best_index = np.argmax(purities) # Getting the highest of 3 cluster purity

feature_names = ["Feature Set 1 (RxI, RxQ)","Feature Set 2 (r, theta)","Feature Set 3 (RxI, RxQ, r, theta)"] # Title of 3 feature sets

print("\nHighest Cluster Purity:")
print("All the 3 features have same Cluster Purity")
print(f"Purity = 1.0000")


purity_strategy1 = [] # Store cluster purity of strategy 1
purity_strategy2 = [] # Store cluster purity of strategy 2


# Strategy 1: Fresh K-means model for EACH SNR

print("\nSTRATEGY 1: FRESH K-MEANS AT EACH SNR\n")

for snr_db in SNR_dB:

    df_snr = df[df['SNR'] == snr_db].copy() # Gets data corresponding to this SNR
    X_snr = df_snr[['RxI', 'RxQ']].values # Cartesian features

    labels = kmeans_16.fit_predict(X_snr) # Cluster assignment for every received sample

    purity = cluster_purity(labels,true_symbol_id) # Calculating cluster purity between predicted and original labels
    purity_strategy1.append(purity) # Appends the data to the empty list

    print(f"SNR = {snr_db:2d} dB  " f"Cluster Purity = {purity:.4f}")

# Strategy 2: Train K-means ONLY at SNR = 25 dB

kmeans_16.fit(X) # Fit K-means by learning centroids and assigning nearest centroid cluster
fixed_centroids = kmeans_16.cluster_centers_ # Get the fixed 25 dB centroids

# Display Fixed 25 dB Centroids

print("\nFIXED CENTROIDS LEARNED AT SNR = 25 dB\n")
for i, centroid in enumerate(fixed_centroids): # Print centroid coordinates
    print(f"Cluster {i:2d}: " f"I = {centroid[0]:.4f}, " f"Q = {centroid[1]:.4f}")

print("\nSTRATEGY 2: FIXED K-MEANS FROM 25 dB\n")

for snr_db in SNR_dB: # Predict every SNR using fixed centroids

    df_snr = df[df['SNR'] == snr_db].copy() # Copying the information related to only given SNR
    X_snr = df_snr[['RxI', 'RxQ']].values # Use only RxI and RxQ

    distances = np.linalg.norm(X_snr[:, np.newaxis, :] - fixed_centroids[np.newaxis, :, :], axis=2) # Finding norm between fixed centroids and received symbols
    labels = np.argmin(distances, axis=1) #Finding nearest centroids for received symbols

    purity = cluster_purity(labels,true_symbol_id) #Calculating cluster purity between predicted and original labels
    purity_strategy2.append(purity) # Appends the data to the empty list

    print(f"SNR = {snr_db:2d} dB  " f"Cluster Purity = {purity:.4f}")


# Display Comparison

print("\nCLUSTER PURITY COMPARISON")

print( "\nSNR(dB) | Strategy 1 | Strategy 2")
print(    "------------------------------------")

for i, snr_db in enumerate(SNR_dB):
    print(f"{snr_db:7d} | " f"{purity_strategy1[i]:10.4f} | " f"{purity_strategy2[i]:10.4f}")

# Plot Cluster Purity vs SNR

plt.figure(figsize=(9, 6)) # Plotting a figure 

plt.plot(SNR_dB,purity_strategy1,marker='o',linewidth=2,label='Strategy 1: Adaptive K-Means') # Plotting Cluster Purity at each SNR using strategy 1

plt.plot(SNR_dB,purity_strategy2,marker='s',linewidth=2,label='Strategy 2: Fixed Template')  # Plotting Cluster Purity at each SNR using strategy 2

plt.xlabel("Eb/N0 (dB)") # Title for x-axis
plt.ylabel("Cluster Purity") # Title for y-axis
plt.title("Cluster Purity vs SNR for Two Demodulation Strategies") # Title for plot

plt.xticks(SNR_dB) # Sets the position of the tick marks on the x-axis
plt.ylim(0, 1.05) # Keep the y-axis between 0 and 1.05
plt.grid(True) # Enables the grid
plt.legend() # Differentiates between 2 strategies

plt.tight_layout() # Automatically adjusts the spacing in plot

plt.show() # Plotting figure 1

plt.show() # Plotting figure 2

plt.show() # Plotting figure 3

plt.show() # Plotting figure 4

