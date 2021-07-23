
# # Jupyter Notebook for Counting Building Occupancy from Polaris Traffic Simulation Data
# 
# This notebook will load a Polaris SQLlite data file into a Pandas data frame using sqlite3 libraries and count the average number of people in each building in each hour of the simulation.
# 
# For help with Jupyter notebooks
# 
# For help on using sql with Pandas see
# http://www.pererikstrandberg.se/blog/index.cgi?page=PythonDataAnalysisWithSqliteAndPandas
# 
# For help  on data analysis with Pandas see
# http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/Index.ipynb
# 

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


fname = ".\data\detroit-Demand.sqlite"
outfolder = ".\\output\\"

print(f'Connecting to database {fname}')
# Create your connection.  Assumes data is in a parallel subdirectory to this one
cnx = sqlite3.connect('.\data\detroit2-Demand.sqlite')

print("getting location data from SQLite File")
# exract all the beginning locations of the building simulation
beginning_location = pd.read_sql_query("SELECT * FROM Beginning_Location_All", cnx)

print("getting trip data from SQLite File")
trips = pd.read_sql_query("SELECT start, end, origin, destination, person FROM Trip", cnx)
trips["start_hr"] = trips.start // 3600
trips["end_hr"] = trips.end // 3600

print("analyzing data from SQL file")
# create the data frames that have the counts of things grouped by start hr & origin and end hr & destination
departs = trips.groupby(['start_hr', 'origin']).size().reset_index(name='countleave')
arrives = trips.groupby(['end_hr', 'destination']).size().reset_index(name='countarrive')


departm = {}
arrivem = {}

# create an array of hours for general use
hr = list(range(24))

print("extracting departures and arrivals")

# create a dataframe for departurses and arrivals per hour by location
# columns are hours, rows are locations.  Fill in missing data with zeros
for i in hr:
    departm[i] = pd.merge(
        beginning_location, departs[departs.start_hr == i],
        left_on='location', right_on='origin', how='left'
    ).fillna(0)
    arrivem[i] = pd.merge(
        beginning_location, arrives[arrives.end_hr == i],
        left_on='location', right_on='destination', how='left'
    ).fillna(0)


# now create an occupancy for each building at each our by adding arrivals and
# subtracting departures from each building at each hour
occm = {}
occm[0] = departm[0].occupants + arrivem[0].countarrive - departm[0].countleave
for i in range(1, 24):
    occm[i] = occm[i-1] + arrivem[i].countarrive - departm[i].countleave

# convert occupancy dictionary to a dataframe
occupancy = pd.DataFrame(occm)

print("counting up occupancy")

# add columns with the locations and their land use / categorization
# to the occupancy dataframe
occupancy["location"]=beginning_location.location
occupancy['land_use']=beginning_location.land_use


cols=['location', 'land_use']+list(range(23))

# reorder the columns putting location and land use at the beginning
cols=['location', 'land_use'] + hr
occupancy = occupancy[cols]

print("removing zero occupancy locations dataframe")
# Remove locations with no occupancy over 24 hours
occupancy_clean = occupancy[occupancy.iloc[:, 2:].abs().sum(axis=1) != 0]

print("grouping locations by land use type")
# Group locations by land use type
land_uses = occupancy_clean.land_use.unique()
occu_profiles = {}
for land in land_uses:
    profiles = occupancy_clean[occupancy_clean.land_use == land].drop('land_use', axis=1)
    occu_profiles[land] = profiles.set_index('location')


print("plotting occupancy by land use")
# plot Occupancy schedules by land use type
fig, axs = plt.subplots(5, 2, figsize=(16, 20))
axs_flat = [x for y in axs for x in y]
for i, ax in enumerate(axs_flat[:-1]):
    land_type = list(occu_profiles.keys())[i]
    df = occu_profiles[land_type]
    ax.step(hr, df.mean(axis=0), color=[0, 0.45, 0.7], label='average')
    ax.fill_between(
        hr, df.min(axis=0), df.max(axis=0),
        step='pre', color=[0, 0.45, 0.7], alpha=0.5, label='range'
    )
    ax.set_xticks(range(0, 24, 2))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('hour', fontsize=14)
    ax.set_ylabel('occupancy', fontsize=14)
    ax.legend(fontsize=14)
    ax.set_xlim([0,23])
    ax.set_ylim([df.min().min() - 1,df.max().max() + 1])
    ax.set_title(land_type, fontsize=18)
fig.tight_layout()
fig.savefig(outfolder + 'occupancy_by_land_use_type.png', bbox_inches='tight', dpi=600)
plt.close()

print("plotting occupancy histograms")
# plot histogram of maximum occupancy by land use type
fig, axs = plt.subplots(5, 2, figsize=(12, 20))
axs_flat = [x for y in axs for x in y]
for i, ax in enumerate(axs_flat[:-1]):
    land_type = list(occu_profiles.keys())[i]
    df = occu_profiles[land_type]
    ax.hist(
        x=df.max(axis=1).values, color=[0, 0.45, 0.7],
        alpha=0.5, label='all buildings'
    )
    ax.axvline(
        x=df.max(axis=1).mean(), color=[0, 0.45, 0.7],
        linewidth=2, label='average'
    )
    # ax.set_xticks(range(0, 24, 1))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Maximum occpuancy', fontsize=14)
    ax.set_ylabel('Counts', fontsize=14)
    ax.legend(fontsize=14)
    ax.set_title(land_type, fontsize=18)
fig.tight_layout()
fig.savefig( outfolder+'maximum_occupancy_by_land_use_type.png', bbox_inches='tight', dpi=600)
plt.close()

print("developing schedules")

# Export occupancy profile data
for key, value in occu_profiles.items():
    csv_out = outfolder + f'Detroit_occupancy_{key.lower()}.csv'
    value.to_csv(csv_out)

# Extract schedules with 4+ maximum occupancy and 0+ minimum occupancy
occu_profiles_clean = {}
for key, value in occu_profiles.items():
    occu_profiles_clean[key] = value[(value.max(axis=1) >= 4) & (value.min(axis=1) >= 0)]

# Normalize by maximum occupancy
occu_profiles_clean_norm = {}
for key, value in occu_profiles_clean.items():
    occu_profiles_clean_norm[key] = value.div(value.max(axis=1), axis=0)


land_types = ['BUSINESS', 'RESIDENTIAL-MULTI']


print("plotting schedules by land use type ")

# Cleaned occupancy schedules by land use type
for land_type in land_types:
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    df = occu_profiles_clean[land_type]
    ax.step(hr, df.mean(axis=0), color=[0, 0.45, 0.7], label='average')
    ax.fill_between(
        hr, df.min(axis=0), df.max(axis=0),
        step='pre', color=[0, 0.45, 0.7], alpha=0.5, label='range'
    )
    ax.set_xticks(range(0, 24, 2))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('hour', fontsize=14)
    ax.set_ylabel('occupancy', fontsize=14)
    ax.legend(fontsize=14)
    ax.set_xlim([0,23])
    ax.set_ylim([df.min().min() - 1,df.max().max() + 1])
    ax.set_title(land_type, fontsize=18)
    fig.savefig(outfolder + 'occupancy_clean_{}.png'.format(land_type.lower()), bbox_inches='tight', dpi=600)
    plt.close()

print("plotting average occupancy with 905% CI")
# Plot average occupancy profiles with 95% range
for land_type in land_types:
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    df = occu_profiles_clean_norm[land_type]
    ax.step(hr, df.mean(axis=0), color=[0, 0.45, 0.7], label='average')
    ax.fill_between(
        hr, df.quantile(0.025, axis=0), df.quantile(0.975, axis=0),
        step='pre', color=[0, 0.45, 0.7], alpha=0.5, label='95% range'
    )
    ax.set_xticks(range(0, 24, 2))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('hour', fontsize=14)
    ax.set_ylabel('occupancy percentage', fontsize=14)
    ax.legend(fontsize=14)
    ax.set_xlim([0,23])
    ax.set_title(land_type, fontsize=18)
    fig.savefig(outfolder + f'occupancy_clean_norm_{land_type.lower()}.png', bbox_inches='tight', dpi=600)
    plt.close()

print("exporting data to CSVs")

# Export cleaned and normalized data
for land_type in land_types:
    csv_out_clean = outfolder + f'Detroit_occupancy_clean_{land_type.lower()}.csv'
    csv_out_clean_norm = outfolder + f'Detroit_occupancy_clean_norm_{land_type.lower()}.csv'
    occu_profiles_clean[land_type].to_csv(csv_out_clean)
    occu_profiles_clean_norm[land_type].to_csv(csv_out_clean_norm)


occu_profiles_clean['BUSINESS'].max(axis=1).value_counts()

# Reference schedules from prototype building models
ref_sches = {
    'occupancy': {
            'BUSINESS': [
                0, 0, 0, 0, 0, 0, 0.11, 0.21, 1, 1, 1, 1,
                0.53, 1, 1, 1, 1, 0.32, 0.11, 0.11, 0.11, 0.11, 0.05, 0
            ],
            'RESIDENTIAL-MULTI': [
                1, 1, 1, 1, 1, 1, 1, 0.85, 0.39, 0.25, 0.25, 0.25,
                0.25, 0.25, 0.25, 0.25, 0.30, 0.52, 0.87, 0.87, 0.87, 1, 1, 1
            ]
    },
    'light': {
            'BUSINESS': [
                0.18, 0.18, 0.18, 0.18, 0.18, 0.23, 0.23, 0.42, 0.9, 0.9, 0.9, 0.9,
                0.8, 0.9, 0.9, 0.9, 0.9, 0.61, 0.42, 0.42, 0.32, 0.32, 0.23, 0.18
            ],
            'RESIDENTIAL-MULTI': [
                0.011, 0.011, 0.011, 0.011, 0.034, 0.074, 0.079, 0.074, 0.034, 0.023, 0.023, 0.023,
                0.023, 0.023, 0.023, 0.040, 0.079, 0.113, 0.153, 0.181, 0.181, 0.124, 0.068, 0.028
                
            ]
    },
    'plug': {
            'BUSINESS': [
                0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1,
                0.94, 1, 1, 1, 1, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
            ],
            'RESIDENTIAL-MULTI': [
                0.45, 0.41, 0.39, 0.38, 0.38, 0.43, 0.54, 0.65, 0.66, 0.67, 0.69, 0.70,
                0.69, 0.66, 0.65, 0.68, 0.80, 1.00, 1.00, 0.93, 0.89, 0.85, 0.71, 0.58
            ]
    },
}


# Adjust lighting and plug load schedules using KNN regression
def adjust_sche(elec_ref, occu_ref, occu, n=3):
    hours_adjusted = np.asarray(range(1, 25)) / 24 * np.pi
    occu_ref_array = np.asarray(occu_ref)
    elec_ref_array = np.asarray(elec_ref)
    elec = []
    for i in range(24):
        dist = np.sqrt(
            np.sin(abs(hours_adjusted - (i+1) / 24 * np.pi)) ** 2 + (occu_ref_array - occu[i]) ** 2
        )
        try:
            elec.append(elec_ref_array[dist == 0][0])
        except IndexError:
            idxs = np.argsort(dist)[:n]
            elec.append(np.average(elec_ref_array[idxs], weights=1/dist[idxs]).round(3))
    return elec


# Lighting
light_profiles_clean_norm = {}
for land_type in land_types:
    light_sche = occu_profiles_clean_norm[land_type].copy()
    for i in range(occu_profiles_clean_norm[land_type].shape[0]):
        light_sche.iloc[i, :] = adjust_sche(
            ref_sches['light'][land_type],
            ref_sches['occupancy'][land_type],
            occu_profiles_clean_norm[land_type].iloc[i, :]
        )
    light_profiles_clean_norm[land_type] = light_sche


plug_profiles_clean_norm = {}
for land_type in land_types:
    plug_sche = occu_profiles_clean_norm[land_type].copy()
    for i in range(occu_profiles_clean_norm[land_type].shape[0]):
        plug_sche.iloc[i, :] = adjust_sche(
            ref_sches['plug'][land_type],
            ref_sches['occupancy'][land_type],
            occu_profiles_clean_norm[land_type].iloc[i, :]
        )
    plug_profiles_clean_norm[land_type] = plug_sche

# Plot average occupancy profiles with 90% range and reference schedules
for land_type in land_types:
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    df = occu_profiles_clean_norm[land_type]
    ax.step(hr, df.mean(axis=0), color=[0, 0.45, 0.7], label='average')
    ax.fill_between(
        hr, df.quantile(0.05, axis=0), df.quantile(0.95, axis=0),
        step='pre', color=[0, 0.45, 0.7], alpha=0.3, label='90% range'
    )
    ax.step(hr, ref_sches['occupancy'][land_type], color='k', linestyle='--', label='DOE REF')
    ax.set_xticks(range(0, 24, 2))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('hour', fontsize=14)
    ax.set_ylabel('occupancy percentage', fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlim([0,23])
    ax.set_title(land_type, fontsize=18)
    fig.savefig(outfolder + f'occupancy_clean_norm_with_ref_{land_type.lower()}.png', bbox_inches='tight', dpi=600)
    plt.close()


# Plot average lighting profiles with 90% range and reference schedules
for land_type in land_types:
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    df = light_profiles_clean_norm[land_type]
    ax.step(hr, df.mean(axis=0), color=[0, 0.45, 0.7], label='average')
    ax.fill_between(
        hr, df.quantile(0.05, axis=0), df.quantile(0.95, axis=0),
        step='pre', color=[0, 0.45, 0.7], alpha=0.3, label='90% range'
    )
    ax.step(hr, ref_sches['light'][land_type], color='k', linestyle='--', label='DOE REF')
    ax.set_xticks(range(0, 24, 2))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('hour', fontsize=14)
    ax.set_ylabel('lighting percentage', fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlim([0,23])
    ax.set_title(land_type, fontsize=18)
    fig.savefig(outfolder + f'lighting_clean_norm_with_ref_{land_type.lower()}.png', bbox_inches='tight', dpi=600)
    plt.close()


# Plot average plug load profiles with 90% range and reference schedules
for land_type in land_types:
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    df = plug_profiles_clean_norm[land_type]
    ax.step(hr, df.mean(axis=0), color=[0, 0.45, 0.7], label='average')
    ax.fill_between(
        hr, df.quantile(0.05, axis=0), df.quantile(0.95, axis=0),
        step='pre', color=[0, 0.45, 0.7], alpha=0.3, label='90% range'
    )
    ax.step(hr, ref_sches['plug'][land_type], color='k', linestyle='--', label='DOE REF')
    ax.set_xticks(range(0, 24, 2))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('hour', fontsize=14)
    ax.set_ylabel('plug load percentage', fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlim([0,23])
    ax.set_title(land_type, fontsize=18)
    fig.savefig(outfolder + f'plugload_clean_norm_with_ref_{land_type.lower()}.png', bbox_inches='tight', dpi=600)
    plt.close()




