#!/usr/bin/env python
# coding: utf-8

# In[69]:


pip install pandas geopandas scikit-learn matplotlib


# In[68]:


pip install prophet


# In[72]:


pip install contextily


# In[71]:


pip install netCDF4


# In[73]:


pip install rasterio xarray


# In[74]:


pip install h5netcdf


# In[75]:


#import libraries
from shapely.geometry import Point
import numpy as np
import netCDF4
import xarray as xr
import pandas as pd
import geopandas as gpd #
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go #plotting
from sklearn.model_selection import train_test_split #machine learning 
from sklearn.ensemble import RandomForestRegressor #machine learnng
from sklearn.metrics import mean_squared_error #machine learning 
from sklearn.preprocessing import StandardScaler
import seaborn as sns



# In[76]:


#data expoloration
contaminant_df = pd.read_csv(r'C:\Users\ks\OneDrive\Documents\Datathon Data\RMD_TCEQ_SWQM_Metals.csv')
print(contaminant_df.columns)


# In[77]:


'''
time: The date of the measurement in the format YYYY-MM-DD.
latitude: The latitude coordinate of the measurement location.
longitude: The longitude coordinate of the measurement location.
adt: This could stand for "Absolute Dynamic Topography," which is the sea surface height above the geoid (a reference surface) due to ocean dynamics.
err_sla: The estimated error in the Sea Level Anomaly (SLA) measurement.
err_ugosa: The estimated error in the "ugosa" variable (possibly related to geostrophic velocities).
err_vgosa: The estimated error in the "vgosa" variable (possibly related to geostrophic velocities).
flag_ice: A flag indicating the presence of ice (0 might mean no ice, while a non-zero value could indicate the presence of ice).
sla: The Sea Level Anomaly, which is the difference between the observed sea surface height and the mean sea surface height over a reference period. in cm or mm 
tpa_correction: A correction term related to the "tpa" variable (possibly a tidal or atmospheric correction).
ugos: This could be the zonal component of geostrophic velocity (velocity related to pressure gradients and Earth's rotation).
ugosa: Similar to "ugos," but with an additional "a" (possibly indicating an anomaly or a specific processing step).
vgos: This could be the meridional component of geostrophic velocity.
vgosa: Similar to "vgos," but with an additional "a" (possibly indicating an anomaly or a specific processing step).





'''


# In[78]:


# Path to your SLR NetCDF file
slr_nc_path = r'C:\Users\ks\OneDrive\Documents\Datathon Data\slr_data.nc'


slr_data = xr.open_dataset(slr_nc_path)

sla_data = slr_data[['longitude', 'latitude', 'sla', 'adt']]
#slr_df = slr_data.to_dataframe().reset_index()
sla_df = sla_data.to_dataframe().reset_index()

# Open the NetCDF file
#slr_data = xr.open_dataset(slr_nc_path)
#print(slr_nc_path)
#print(slr_nc_path.variables.keys())
# Access data (adjust 'variable_name' to match the variable you're interested in)



print(sla_df.head())
#print(slr_df.columns)
#print(slr_df['adt'])
#print(slr_df['sla'])
#print(slr_df['latitude'])
#print(slr_df['longitude'])


# In[79]:


# longitude and longitud 
#convert 



gdf_contaminant = gpd.GeoDataFrame(
    contaminant_df, 
    geometry=gpd.points_from_xy(contaminant_df['longitude'], contaminant_df['latitude'])
)

# Set CRS to WGS84
gdf_contaminant.crs = "EPSG:4326"

# Future datasets should be integrated similarly


#convert Sea level rise data to a geodata frame
gdf_sla = gpd.GeoDataFrame(sla_df, geometry =[Point(xy) for xy in zip(sla_df['longitude'], sla_df['latitude'])])

# set crs to wgs84 (epsg:4326)

gdf_sla.crs = "EPSG:4326"



# In[80]:


#pre visualization 
#sea level anomaly over time 

sla_df['time'] = pd.to_datetime(sla_df['time'])  # Ensure time is in datetime format
sla_df.set_index('time')['sla'].plot(figsize=(10, 6))
plt.title('Sea Level Anomaly Over Time')
plt.ylabel('Sea Level Anomaly')
plt.show()


# In[81]:


#geocontaminant data frame


'''
sla_df['time'] = pd.to_datetime(sla_df['time'])  # Ensure time is in datetime format
sla_df.set_index('time')['sla'].plot(figsize=(10, 6))
plt.title('Sea Level Anomaly Over Time')
plt.ylabel('Sea Level Anomaly')
plt.show()


'''
unique_items = contaminant_df['Par_Desc'].unique()
#print(unique_items)

##USING Par_Desc##
#LEAD, TOTAL (UG/L AS PB)
#MERCURY, TOTAL, WATER, METHOD 1631 ug/L
#ARSENIC, BOTTOM DEPOSITS (MG/KG AS AS DRY WT)
#CHROMIUM, TOTAL (UG/L AS CD)

lead_df = contaminant_df[contaminant_df['Par_Desc'].notna() & contaminant_df['Par_Desc'].str.contains('LEAD, TOTAL')]

lead_df_new = lead_df[[ 'Par_Desc', 'Value', 'End_Date', 'longitude', 'latitude']].copy()


merc_df = contaminant_df[contaminant_df['Par_Desc'].notna() & contaminant_df['Par_Desc'].str.contains('MERCURY, TOTAL')]

merc_df_new = merc_df[[ 'Par_Desc', 'Value', 'End_Date', 'longitude', 'latitude']].copy()




ars_df = contaminant_df[contaminant_df['Par_Desc'].notna() & contaminant_df['Par_Desc'].str.contains('ARSENIC, TOTAL')]

ars_df_new = ars_df[[ 'Par_Desc', 'Value', 'End_Date', 'longitude', 'latitude']].copy()


chrome_df = contaminant_df[contaminant_df['Par_Desc'].notna() & contaminant_df['Par_Desc'].str.contains('CHROMIUM,TOTAL')]
#ph_df['ph_longitude'] = ph_df['LongitudeDD'].round(3)
#ph_df['ph_latitude'] = ph_df['LatitudeDD'].round(3)
chrome_df_new = chrome_df[[ 'Par_Desc', 'Value', 'End_Date', 'longitude', 'latitude']].copy()
#prepare both data sets by rounding their geographical coordinates 

#sla_df['rounded_longitude'] = sla_df['longitude'].round(3)
#sla_df['rounded_latitude'] = sla_df['latitude'].round(3)


# new simplified df for lead values 


temp = contaminant_df[
    (contaminant_df['Par_Desc'].notna()) & 
    (contaminant_df['Par_Desc'].str.contains('LEAD, TOTAL') | 
     contaminant_df['Par_Desc'].str.contains('MERCURY, TOTAL') | 
     contaminant_df['Par_Desc'].str.contains('ARSENIC, TOTAL') | 
    contaminant_df['Par_Desc'].str.contains('CHROMIUM, TOTAL'))
]


contaminant_data = temp[['Par_Desc', 'Value', 'End_Date', 'longitude', 'latitude']].copy()



print("sla = " + str(len(sla_df)))
print("contaminant_df = " + str(len(contaminant_df)))
print("chrome_df_new = " + str(len(chrome_df_new))) # 56
print("lead_df = " + str(len(lead_df_new))) # 17
print("ars_df_new = " + str(len(ars_df_new))) 
print("mer = " + str(len(merc_df_new)))


#print(contaminant_data)


# In[82]:


#temporal alignment 
#convert contaminant and sea level anomaly times to date time format

'''
NOTE FOR TOMORROW

- temporally align contaminant data and sla on a time window
   -aggregate into monthly/yearly averages

- merge data with the average time windows including the spatial coordinates

- For spatial alignment, you can use the latitude and longitude as the key. 
-However, direct matching might not be feasible due to the precision difference or measurement error. 
-You might consider rounding the coordinates to a certain number of decimal places or 
-using a spatial join with a tolerance range if the locations are very close but not exactly matching.

then 
2. Geospatial Analysis
- mapping and spatial correlation, use GIS to map distribution of contaminants and compare 
 with SLA and ADT variations
 
 
3. Model the influence of Sea Level on grounwater

- hydrodynamic and transport models 

4. Statistical and Machine Learning Models
- regression models

'''
'''
precision = 3  # Example: round to the nearest 0.001
sla_df['longitude'] = sla_df['longitude'].round(precision)
sla_df['latitude'] = sla_df['latitude'].round(precision)
contaminant_data['longitude'] = contaminant_data['longitude'].round(precision)
contaminant_data['latitude'] = contaminant_data['latitude'].round(precision)

contaminant_data['End_Date'] = pd.to_datetime(contaminant_data['End_Date'])
sla_df['time'] = pd.to_datetime(sla_df['time'])



sla_df['month_year'] = sla_df['time'].dt.to_period('M')
sla_df_monthly = sla_df.groupby(['month_year', 'longitude', 'latitude'])[['sla', 'adt']].mean().reset_index()

contaminant_data['month_year'] = contaminant_data['End_Date'].dt.to_period('M')
contaminant_data['Value'] = pd.to_numeric(contaminant_data['Value'], errors='coerce')
contaminant_monthly = contaminant_data.groupby(['month_year', 'longitude', 'latitude', 'Par_Desc'])['Value'].mean().reset_index()


# Rename for exact match if necessary
#sla_df_monthly.rename(columns={'longitude': 'Longitude', 'latitude': 'Latitude'}, inplace=True)

aligned_monthly_df = pd.merge(sla_df_monthly, contaminant_monthly, how='inner', on=['month_year', 'longitude', 'latitude'])


aligned_monthly_df

#print(sla_df['month_year'].unique())
#print(contaminant_data['month_year'].unique())

'''


#merging with overall contaminant data 
#spatial joining using geopandas to create spatial and temporal aggregation 

sla_df['time'] = pd.to_datetime(sla_df['time'])
contaminant_df['End_Date'] = pd.to_datetime(contaminant_df['End_Date'])

grid_resolution = 0.1  # adjust as needed

# function to assign grid cell IDs based on coordinates



def assign_grid_cell(lat, lon, resolution):
    grid_lat = int(lat // resolution)
    grid_lon = int(lon // resolution)
    return f"{grid_lat}_{grid_lon}"

# assign grid cell IDs to both DataFrames
sla_df['grid_cell'] = sla_df.apply(lambda row: assign_grid_cell(row['latitude'], row['longitude'], grid_resolution), axis=1)
contaminant_df['grid_cell'] = contaminant_df.apply(lambda row: assign_grid_cell(row['latitude'], row['longitude'], grid_resolution), axis=1)

# temporal aggregation interval
temporal_interval = 'M'  

# Perform spatial and temporal aggregation on sla and contaminant data
sla_agg_df = sla_df.groupby(['grid_cell', pd.Grouper(key='time', freq=temporal_interval)])[['sla','adt']].mean().reset_index()


contaminant_agg_df = contaminant_df.groupby(['grid_cell', pd.Grouper(key='End_Date', freq=temporal_interval)])['Value'].mean().reset_index()
contaminant_agg_df = contaminant_agg_df.rename(columns={'End_Date': 'time'})


# Merge the aggregated DataFrames based on grid cell and time interval
merged_df = pd.merge(sla_agg_df, contaminant_agg_df, on=['grid_cell', 'time'], how='inner')

# Print the merged DataFrame
print(merged_df)




# In[83]:


#arsenic predictive model: 
#known carcinogen, causes skin lesions, etc 

#data merging with arsenic df
sla_df['time'] = pd.to_datetime(sla_df['time'])
ars_df_new['End_Date'] = pd.to_datetime(ars_df_new['End_Date'])

grid_resolution = 0.1  # adjust as needed

# function to assign grid cell IDs based on coordinates



def assign_grid_cell(lat, lon, resolution):
    grid_lat = int(lat // resolution)
    grid_lon = int(lon // resolution)
    return f"{grid_lat}_{grid_lon}"

# assign grid cell IDs to both DataFrames
sla_df['grid_cell'] = sla_df.apply(lambda row: assign_grid_cell(row['latitude'], row['longitude'], grid_resolution), axis=1)
ars_df_new['grid_cell'] = ars_df_new.apply(lambda row: assign_grid_cell(row['latitude'], row['longitude'], grid_resolution), axis=1)

# temporal aggregation interval
temporal_interval = 'M'  

# Perform spatial and temporal aggregation on sla and contaminant data
sla_agg_df = sla_df.groupby(['grid_cell', pd.Grouper(key='time', freq=temporal_interval)])[['sla','adt']].mean().reset_index()


ars_agg_df = ars_df_new.groupby(['grid_cell', pd.Grouper(key='End_Date', freq=temporal_interval)])['Value'].mean().reset_index()
ars_agg_df = ars_agg_df.rename(columns={'End_Date': 'time'})


# Merge the aggregated DataFrames based on grid cell and time interval
merged_df_ars = pd.merge(sla_agg_df, ars_agg_df, on=['grid_cell', 'time'], how='inner')

# Print the merged DataFrame
print(merged_df_ars)



# In[84]:


#lead predictive model 
#Can affect the nervous system, kidney function, immune system, developmental and reproductive systems.
#print(sla_df)

#data merging with lead df
sla_df['time'] = pd.to_datetime(sla_df['time'])
lead_df_new['End_Date'] = pd.to_datetime(lead_df_new['End_Date'])

grid_resolution = 0.1  # adjust as needed

# function to assign grid cell IDs based on coordinates



def assign_grid_cell(lat, lon, resolution):
    grid_lat = int(lat // resolution)
    grid_lon = int(lon // resolution)
    return f"{grid_lat}_{grid_lon}"

# assign grid cell IDs to both DataFrames
sla_df['grid_cell'] = sla_df.apply(lambda row: assign_grid_cell(row['latitude'], row['longitude'], grid_resolution), axis=1)
lead_df_new['grid_cell'] = lead_df_new.apply(lambda row: assign_grid_cell(row['latitude'], row['longitude'], grid_resolution), axis=1)

# temporal aggregation interval
temporal_interval = 'M'  

# Perform spatial and temporal aggregation on sla and contaminant data
sla_agg_df = sla_df.groupby(['grid_cell', pd.Grouper(key='time', freq=temporal_interval)])[['sla','adt']].mean().reset_index()


lead_agg_df = lead_df_new.groupby(['grid_cell', pd.Grouper(key='End_Date', freq=temporal_interval)])['Value'].mean().reset_index()
lead_agg_df = lead_agg_df.rename(columns={'End_Date': 'time'})


# Merge the aggregated DataFrames based on grid cell and time interval
merged_df_lead = pd.merge(sla_agg_df, lead_agg_df, on=['grid_cell', 'time'], how='inner')

# Print the merged DataFrame
print(merged_df_lead)



# In[85]:


#mercury predictive model: 
#Neurotoxin that can impair neurological development in infants and children





#data merging with lead df
sla_df['time'] = pd.to_datetime(sla_df['time'])
merc_df_new['End_Date'] = pd.to_datetime(merc_df_new['End_Date'])

grid_resolution = 0.1  # adjust as needed

# function to assign grid cell IDs based on coordinates



def assign_grid_cell(lat, lon, resolution):
    grid_lat = int(lat // resolution)
    grid_lon = int(lon // resolution)
    return f"{grid_lat}_{grid_lon}"

# assign grid cell IDs to both DataFrames
sla_df['grid_cell'] = sla_df.apply(lambda row: assign_grid_cell(row['latitude'], row['longitude'], grid_resolution), axis=1)
merc_df_new['grid_cell'] = merc_df_new.apply(lambda row: assign_grid_cell(row['latitude'], row['longitude'], grid_resolution), axis=1)

# temporal aggregation interval
temporal_interval = 'M'  

# Perform spatial and temporal aggregation on sla and contaminant data
sla_agg_df = sla_df.groupby(['grid_cell', pd.Grouper(key='time', freq=temporal_interval)])[['sla','adt']].mean().reset_index()


merc_agg_df = merc_df_new.groupby(['grid_cell', pd.Grouper(key='End_Date', freq=temporal_interval)])['Value'].mean().reset_index()
merc_agg_df = merc_agg_df.rename(columns={'End_Date': 'time'})


# Merge the aggregated DataFrames based on grid cell and time interval
merged_df_merc = pd.merge(sla_agg_df, merc_agg_df, on=['grid_cell', 'time'], how='inner')

# Print the merged DataFrame
print(merged_df_merc)






# In[86]:


#chromium:  
#Linked to lung cancer and can cause skin irritation and ulceration.



sla_df['time'] = pd.to_datetime(sla_df['time'])
chrome_df_new['End_Date'] = pd.to_datetime(chrome_df_new['End_Date'])

grid_resolution = 0.1  # adjust as needed

# function to assign grid cell IDs based on coordinates



def assign_grid_cell(lat, lon, resolution):
    grid_lat = int(lat // resolution)
    grid_lon = int(lon // resolution)
    return f"{grid_lat}_{grid_lon}"

# assign grid cell IDs to both DataFrames
sla_df['grid_cell'] = sla_df.apply(lambda row: assign_grid_cell(row['latitude'], row['longitude'], grid_resolution), axis=1)
chrome_df_new['grid_cell'] = chrome_df_new.apply(lambda row: assign_grid_cell(row['latitude'], row['longitude'], grid_resolution), axis=1)

# temporal aggregation interval
temporal_interval = 'M'  

# Perform spatial and temporal aggregation on sla and contaminant data
sla_agg_df = sla_df.groupby(['grid_cell', pd.Grouper(key='time', freq=temporal_interval)])[['sla','adt']].mean().reset_index()


chrome_agg_df = chrome_df_new.groupby(['grid_cell', pd.Grouper(key='End_Date', freq=temporal_interval)])['Value'].mean().reset_index()
chrome_agg_df = chrome_agg_df.rename(columns={'End_Date': 'time'})


# Merge the aggregated DataFrames based on grid cell and time interval
merged_df_chrome = pd.merge(sla_agg_df, chrome_agg_df, on=['grid_cell', 'time'], how='inner')

# Print the merged DataFrame
print(merged_df_chrome)








# In[87]:


# Create a scatter plot of SLA, ADT, and Value over time
fig, ax = plt.subplots(figsize=(12, 8))

# Plot SLA
ax.scatter(merged_df['time'], merged_df['sla'], color='blue', label='SLA')
z_sla = np.polyfit(merged_df.index, merged_df['sla'], 1)
p_sla = np.poly1d(z_sla)
ax.plot(merged_df['time'], p_sla(merged_df.index), color='blue', linestyle='--', label='SLA Trendline')

# Plot ADT
ax.scatter(merged_df['time'], merged_df['adt'], color='red', label='ADT')
z_adt = np.polyfit(merged_df.index, merged_df['adt'], 1)
p_adt = np.poly1d(z_adt)
ax.plot(merged_df['time'], p_adt(merged_df.index), color='red', linestyle='--', label='ADT Trendline')

# Plot Contaminant Value
ax2 = ax.twinx()
ax2.scatter(merged_df['time'], merged_df['Value'], color='green', label='Contaminant Value')
z_value = np.polyfit(merged_df.index, merged_df['Value'], 1)
p_value = np.poly1d(z_value)
ax2.plot(merged_df['time'], p_value(merged_df.index), color='green', linestyle='--', label='Contaminant Value Trendline')

# Set labels and title
ax.set_xlabel('Time')
ax.set_ylabel('SLA/ADT')
ax2.set_ylabel('Contaminant Value')
ax.set_title('SLA, ADT, and Contaminant Value over Time')

# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()




# In[130]:


#model for mercury
from sklearn.linear_model import LinearRegression
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'sla_model', 'adt_model', and 'value_model' are already trained and 'merged_df_merc' is your dataset

# Prepare the future data
future_dates_merc = pd.date_range(start='2024-01-01', end='2026-12-31', freq='MS')
future_df_merc = pd.DataFrame({'time': future_dates_merc})
future_df_merc['month'] = future_df_merc['time'].dt.month
future_df_merc['year'] = future_df_merc['time'].dt.year

# Set 'sla' and 'adt' in future_df_merc to the last known values from merged_df_merc
last_sla = merged_df_merc['sla'].iloc[-1]
last_adt = merged_df_merc['adt'].iloc[-1]
future_df_merc['sla'] = last_sla
future_df_merc['adt'] = last_adt

# Make predictions for future SLA, ADT, and geocontaminant values using separate models
future_df_merc['predicted_sla'] = sla_model.predict(future_df_merc[['sla', 'adt', 'month', 'year']])
future_df_merc['predicted_adt'] = adt_model.predict(future_df_merc[['sla', 'adt', 'month', 'year']])
future_df_merc['predicted_value'] = value_model.predict(future_df_merc[['sla', 'adt', 'month', 'year']])

# Filter the merged_df_merc DataFrame to include data from 2024 onwards
merged_df_filtered_merc = merged_df_merc[merged_df_merc['time'] >= '2024-01-01']

fig, ax = plt.subplots(figsize=(12, 8))

# Plotting predicted SLA trend
ax.plot(future_df_merc['time'], future_df_merc['predicted_sla'], color='blue', label='Predicted SLA (m)')

# Plotting predicted ADT trend
ax.plot(future_df_merc['time'], future_df_merc['predicted_adt'], color='green', label='Predicted ADT (m)')

# Plotting predicted geocontaminant value trend
ax.plot(future_df_merc['time'], future_df_merc['predicted_value'], color='purple', label='Predicted Mercury Value (ug/L)')

# Setting the labels, title, and legend
ax.set_xlabel('Time')
ax.set_ylabel('Values')
ax.set_title('Predicted Trends vs. Time (2024 onwards)')
ax.legend()
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# In[128]:


#model for lead

from sklearn.linear_model import LinearRegression
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'sla_model', 'adt_model', and 'value_model' are already trained and 'merged_df_lead' is your dataset

# Prepare the future data
future_dates_lead = pd.date_range(start='2024-01-01', end='2025-12-31', freq='MS')
future_df_lead = pd.DataFrame({'time': future_dates_lead})
future_df_lead['month'] = future_df_lead['time'].dt.month
future_df_lead['year'] = future_df_lead['time'].dt.year

# Set 'sla' and 'adt' in future_df_lead to the last known values from merged_df_lead
last_sla = merged_df_lead['sla'].iloc[-1]
last_adt = merged_df_lead['adt'].iloc[-1]
future_df_lead['sla'] = last_sla
future_df_lead['adt'] = last_adt

# Make predictions for future SLA, ADT, and geocontaminant values using separate models
future_df_lead['predicted_sla'] = sla_model.predict(future_df_lead[['sla', 'adt', 'month', 'year']])
future_df_lead['predicted_adt'] = adt_model.predict(future_df_lead[['sla', 'adt', 'month', 'year']])
future_df_lead['predicted_value'] = value_model.predict(future_df_lead[['sla', 'adt', 'month', 'year']])

# Filter the merged_df_lead DataFrame to include data from 2024 onwards
merged_df_filtered_lead = merged_df_lead[merged_df_lead['time'] >= '2024-01-01']
import matplotlib.pyplot as plt

# Assuming future_df_lead contains the future predictions and merged_df_filtered_lead contains actual values from 2024 onwards

fig, ax = plt.subplots(figsize=(12, 8))

# Predicted trends
ax.plot(future_df_lead['time'], future_df_lead['predicted_sla'], color='blue', label='Predicted SLA (m)')
ax.plot(future_df_lead['time'], future_df_lead['predicted_adt'], color='green', label='Predicted ADT (m)')
ax.plot(future_df_lead['time'], future_df_lead['predicted_value'], color='purple', label='Predicted Lead Level (ug/L)')

ax.set_xlabel('Time')
ax.set_ylabel('Values')
ax.set_title('Predicted Trends vs. Time for Lead(2024 onwards)')
ax.legend()
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# In[135]:


#model for arsenic

from sklearn.linear_model import LinearRegression
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'sla_model', 'adt_model', and 'value_model' are already trained and 'merged_df_merc' is your dataset

# Prepare the future data
future_dates_ars = pd.date_range(start='2024-01-01', end='2025-12-31', freq='MS')
future_df_ars = pd.DataFrame({'time': future_dates_ars})
future_df_ars['month'] = future_df_ars['time'].dt.month
future_df_ars['year'] = future_df_ars['time'].dt.year

# Set 'sla' and 'adt' in future_df_merc to the last known values from merged_df_merc
last_sla = merged_df_ars['sla'].iloc[-1]
last_adt = merged_df_ars['adt'].iloc[-1]
future_df_ars['sla'] = last_sla
future_df_ars['adt'] = last_adt

# Make predictions for future SLA, ADT, and geocontaminant values using separate models
future_df_ars['predicted_sla'] = sla_model.predict(future_df_ars[['sla', 'adt', 'month', 'year']])
future_df_ars['predicted_adt'] = adt_model.predict(future_df_ars[['sla', 'adt', 'month', 'year']])
future_df_ars['predicted_value'] = value_model.predict(future_df_ars[['sla', 'adt', 'month', 'year']])

# Filter the merged_df_merc DataFrame to include data from 2024 onwards
merged_df_filtered_ars = merged_df_ars[merged_df_ars['time'] >= '2024-01-01']

# Visualize the predicted trends
fig, ax = plt.subplots(figsize=(12, 8))

# Predicted trends
ax.plot(future_df_ars['time'], future_df_ars['predicted_sla'], color='blue', label='Predicted SLA (m)')
ax.plot(future_df_ars['time'], future_df_ars['predicted_adt'], color='green', label='Predicted ADT (m)')
ax.plot(future_df_ars['time'], future_df_ars['predicted_value'], color='purple', label='Predicted Mercury Level (ug/L)')

ax.set_xlabel('Time')
ax.set_ylabel('Values')
ax.set_title('Predicted Trends vs. Time (2024 onwards)')
ax.legend()
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# In[123]:


from sklearn.linear_model import LinearRegression
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'sla_model', 'adt_model', and 'value_model' are already trained and 'merged_df_chrome' is your dataset

# Prepare the future data
future_dates_chrome = pd.date_range(start='2024-01-01', end='2025-12-31', freq='MS')
future_df_chrome = pd.DataFrame({'time': future_dates_chrome})
future_df_chrome['month'] = future_df_chrome['time'].dt.month
future_df_chrome['year'] = future_df_chrome['time'].dt.year

# Set 'sla' and 'adt' in future_df_chrome to the last known values from merged_df_chrome
last_sla = merged_df_chrome['sla'].iloc[-1]
last_adt = merged_df_chrome['adt'].iloc[-1]
future_df_chrome['sla'] = last_sla
future_df_chrome['adt'] = last_adt

# Make predictions for future SLA, ADT, and geocontaminant values using separate models
future_df_chrome['predicted_sla'] = sla_model.predict(future_df_chrome[['sla', 'adt', 'month', 'year']])
future_df_chrome['predicted_adt'] = adt_model.predict(future_df_chrome[['sla', 'adt', 'month', 'year']])
future_df_chrome['predicted_value'] = value_model.predict(future_df_chrome[['sla', 'adt', 'month', 'year']])

# Filter the merged_df_chrome DataFrame to include data from 2024 onwards
merged_df_filtered_chrome = merged_df_chrome[merged_df_chrome['time'] >= '2024-01-01']

# Visualize the predicted trends
fig, ax = plt.subplots(figsize=(12, 8))

# Predicted trends
ax.plot(future_df_chrome['time'], future_df_chrome['predicted_sla'], color='blue', label='Predicted SLA (m)')
ax.plot(future_df_chrome['time'], future_df_chrome['predicted_adt'], color='green', label='Predicted ADT (m)')
ax.plot(future_df_chrome['time'], future_df_chrome['predicted_value'], color='purple', label='Predicted Chromium Level (ug/L)')

ax.set_xlabel('Time')
ax.set_ylabel('Values')
ax.set_title('Predicted Trends vs. Time (2024 onwards)')
ax.legend()
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# In[125]:


future_df_merc


# In[126]:


future_df_lead


# In[136]:


future_df_ars


# In[ ]:




