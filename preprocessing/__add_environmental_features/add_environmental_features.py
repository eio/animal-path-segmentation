from cdsapi import Client
from datetime import datetime, timedelta
import pandas as pd
import pygrib
import csv
import os, sys

# Get the absolute path to the directory containing the current script
# and append to sys.path the subdirectory containing the local module to import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
# Local scripts
from utils.consts import TIMESTAMP, LATITUDE, LONGITUDE
from terrain_type_lookups import TERRAIN_TYPE_LOOKUP

# Set up the Copernicus Data Store (CDS) API client
# (requires credentials in `/Users/$USERNAME/.cdsapirc`)
client = Client()

# Input: Labeled data with derived time & movement features
INPUT_CSV = "Cranes_all_features.csv"
# Output: All of the above + ECMWF ERA5 environmental data (via CDS API)
OUTPUT_CSV = "Cranes_and_environment.csv"
# Specify the variables to retrieve with the CDS API
VARIABLES = {
    # pygrib.open()[x].name : $VAR_NAME_CDS_API,
    "2 metre temperature": "2m_temperature",
    "Total precipitation": "total_precipitation",
    "Surface net solar radiation": "surface_solar_radiation",
    "Volumetric soil water layer 1": "volumetric_soil_water_layer_1",
    "Soil type": "soil_type",
    "High vegetation cover": "high_vegetation_cover",
    "Type of high vegetation": "type_of_high_vegetation",
    "Low vegetation cover": "low_vegetation_cover",
    "Type of low vegetation": "type_of_low_vegetation",
}


def round_to_nearest_hour(time_string):
    # Convert the time string to a datetime object
    time_obj = datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S.%f")
    # Round the minutes and seconds to the nearest hour
    if time_obj.minute >= 30:
        # Increment the hour & zero out minutes and seconds
        rounded_time_obj = time_obj.replace(minute=0, second=0) + timedelta(hours=1)
    else:
        # Maintain the hour & zero out minutes and seconds
        rounded_time_obj = time_obj.replace(minute=0, second=0)
    # Format the rounded time as a string without milliseconds
    rounded_time_string = rounded_time_obj.strftime("%Y-%m-%d %H:%M:%S")
    return rounded_time_string


def extract_data(grib_msgs):
    data = {}
    for g in grib_msgs:
        varname = VARIABLES[g.name]
        data[varname] = g.values
        # if "type" in varname:
        #     print(varname + "_STRING", int(g.values))
        #     data[varname + "_STRING"] = TERRAIN_TYPE_LOOKUP[varname][int(g.values)]
    return data


def make_cds_request(timestamp, latitude, longitude, filename_grib):
    payload = {
        "format": "grib",
        "product_type": "reanalysis",
        # Specify a single point by listing coordinate twice
        "area": [latitude, longitude, latitude, longitude],
        # ex. timestamp = "2011-02-16 13:30:44.000"
        "year": timestamp[:4],
        "month": timestamp[5:7],
        "day": timestamp[8:10],
        "time": timestamp[11:19],
        # https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels
        "variable": list(VARIABLES.values()),
    }
    print("PAYLOAD:", payload)
    # Make the API request
    client.retrieve(
        "reanalysis-era5-single-levels",
        payload,
        filename_grib,
    )


def get_environmental_data(df):
    # Iterate through each row of the Dataframe
    for index, row in df.iterrows():
        # Unpack the relevant fields
        timestamp = row[TIMESTAMP]
        # Round timestamp to nearest hour to meet CDS expectation
        timestamp = round_to_nearest_hour(timestamp)
        latitude = str(row[LATITUDE])
        longitude = str(row[LONGITUDE])
        # Define output GRIB filename
        identifier = "{}_{}_{}".format(
            latitude, longitude, timestamp.split(".")[0].replace(" ", "T")
        )
        filename_grib = "grib/{}.grib".format(identifier)
        # Use the Copernicus Data Store (CDS) API to
        # retrieve environmental data and save as a local GRIB file
        make_cds_request(
            timestamp,
            latitude,
            longitude,
            filename_grib,
        )
        # Open the GRIB file
        grib_msgs = pygrib.open(filename_grib)
        # Read the environmental data
        env_data = extract_data(grib_msgs)
        # Update the Dataframe row with new columns and values
        for column, value in env_data.items():
            df.at[index, column] = value
        # Close the GRIB file
        grib_msgs.close()
    # Save the updated Dataframe to local CSV
    df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    # Define the input CSV of events data
    # with expected columns: [latitude, longitude, timestamp]
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {df.shape[0]} events.")
    # Iterate through the provided events data
    # and pull environmental data from Copernicus Data Store
    # at each individual waypoint
    get_environmental_data(df)
