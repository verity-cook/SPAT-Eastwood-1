"""
SPAT-Eastwood-1
Verity Cook and Rebecca Dunkley
01/05/2020

Calcuting FAC values from the magnetic field data from Swarm
"""

import numpy as np
import pandas as pd

#%% Swarm Method (Ritter et. al. 2013)

def calc_gamma(v_N, v_E):
    """
    calculate rotation angle
    Input:
        v_N, v_E - x, y components of velocity in NEC frame
    Output:
        gamma, rotation angle
    """
    return - np.arctan((v_N - v_E)/(v_N + v_E))

def NEC_to_VSC(N, E, gamma):
    """
    convert X from NEC to VSC frame
    Input:
        N, E - x, y components in NEC frame
    Output:
        x and y components in VSC frame
    """
    
    # calculate x and y components in VSC frame
    V =   N * np.cos(gamma) + E * np.sin(gamma)
    S = - N * np.sin(gamma) + E * np.cos(gamma)
    
    return V, S

def get_velocity_NEC(data):
    """
    find velocity components in NEC frame 
    Input: 
        data - pandas DataFrame from Swarm
    """
    # geocentric poision
    r = data.Radius    
    lat = data.Latitude  * np.pi/180 # in radians
    lon = data.Longitude * np.pi/180
    
    # get N and E components
    N = r * lat
    E = r * np.cos(lat) * lon
    
    # get velocity (dt = 1)
    v_N = N.diff() # dN/dt
    v_E = E.diff() # dE/dt   
    
    return v_N, v_E

 
def single_sat_FAC(data):
    """
    calculate single-satellite FAC densities
    Input: 
        data - pandas DataFrame from Swarm
    Output:
        orginal DataFrame with FAC added
        
    """
    # mean field
    B_MF_N = data.B_NEC_IGRF_N
    B_MF_E = data.B_NEC_IGRF_E
    B_MF_C = data.B_NEC_IGRF_C
    
    # residual magnetic fields
    B_N = data.B_NEC_N - B_MF_N
    B_E = data.B_NEC_E - B_MF_E
    
    # velocity in NEC frame
    v_N, v_E = get_velocity_NEC(data)
    
    gamma = calc_gamma(v_N, v_E)
    
    v_V, v_S = NEC_to_VSC(v_N, v_E, gamma) # velocity in VSC frame
    B_V, B_S = NEC_to_VSC(B_N, B_E, gamma) # magnetic field in VSC frame
    
    # find difference in neibouring data values
    dB_V = B_V.diff()
    dB_S = B_S.diff()
    
    dt = 1 # time between data points
    mu_0 = 4E-7 * np.pi # permeability of free space
    j_r = - (1/(2 * mu_0 * dt * 1000)) * ((dB_S/v_V) - (dB_V/v_S))
    
    # mean field inclination angle
    I = np.arctan(B_MF_C/np.sqrt(B_MF_N**2 + B_MF_E**2))

    # calculate single satellite FAC
    j_FAC = - j_r/np.sin(I)
    data_new = data.copy()
    data_new['I'] = I
    data_new['FAC_SS'] = j_FAC
    
    # if absolute inclination angle is less than 30 deg
    data_new.loc[data_new.I.abs() < 30 * np.pi/180,'FAC_SS'] = np.nan
    return data_new
    

    
    
    

    



