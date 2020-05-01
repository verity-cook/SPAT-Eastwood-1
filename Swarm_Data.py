"""
SPAT-Eastwood-1
Verity Cook and Rebecca Dunkley
17/04/2020

Get data from the Swarm database
"""
import datetime as dt
from viresclient import SwarmRequest # download data from the Swarm database

#%% Downloading Swarm data
def get_data(value, satellite, start, end, step = None):                      
    """
     Input:
        values     - specify datasets to retrieve eg. 'MAG', 'FAC'
        satellite  - 'A', 'B', 'C' or '_' for dual satellite FAC
        start      - datetime, start time for data
        end        - datetime, end time for data
        step       - ISO_8601 duration format 
                     eg. 'PT10M' to sample every 10 minutes
     Output:
        data as pandas DataFrame
    """
    # get magnetic field values in NEC frame
    if value == 'MAG':
        collection = 'SW_OPER_MAG' + satellite + '_LR_1B'
        values = ['B_NEC']
    
    # or get FAC values
    elif value == 'FAC':
        collection = 'SW_OPER_FAC' + satellite + 'TMS_2F'
        values = ['FAC']
                    
    request = SwarmRequest()
   
    # select which collection of data
    request.set_collection(collection) 
    
    # select which variables to retrieve
    request.set_products(measurements  = values,        # measured by satellite of chosen collection
                         models        = ['IGRF'],      # get IGRF model
                         auxiliaries   = ['Kp', 'Dst', 'MLT', 'QDLat', 'QDLon'], # additional parameters, not unique to collection
                         residuals     = False,         # if True then only data-model residuals are returned
                         sampling_step = step)          # sampling rate    
    
    
    # time range to retrive data
    data = request.get_between(start_time = start,
                               end_time   = end)
    
    # transfer data as pandas DataFrame
    df = data.as_dataframe(expand = True) # split B_NEC into three seperate arrays 

    return df

#%% Download and save data
for y in range(2014, 2020):
    # get magnetic field data for the year
    MAG = get_data('MAG', 'A', dt.datetime(y, 1, 1), dt.datetime(y + 1, 1, 1))
    MAG.to_csv('MAG/MAG{}.csv'.format(y))
    print('saved data {}'.format(y))





