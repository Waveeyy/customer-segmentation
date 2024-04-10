import pandas as pd
import numpy as np
from datetime import datetime

# Load the dataset
df = pd.read_excel('online_retail_II.xlsx')

# Remove cancelled orders
df = df[df['Quantity'] > 0]

# Remove rows where customerID is NA
df.dropna(subset=['Customer ID'], inplace=True)

# Convert the InvoiceDate from object to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Create a TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['Price']

# Set the observation point as the maximum invoice date plus one day
observation_point = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Calculate RFM metrics
rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (observation_point - x.max()).days,
    'Invoice': 'count',
    'TotalPrice': 'sum'
}).rename(columns={'InvoiceDate': 'Recency',
                   'Invoice': 'Frequency',
                   'TotalPrice': 'MonetaryValue'})

# Check the first few rows
print(rfm.head())

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Scale the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Choose the number of clusters using the elbow method
sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    sse[k] = kmeans.inertia_  # Sum of squared distances to closest cluster center

# Plot SSE for each *k*
import matplotlib.pyplot as plt

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


kmeans = KMeans(n_clusters=5, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Check the distribution of clusters
print(rfm['Cluster'].value_counts())

# Further analysis can be done to profile and interpret clusters
