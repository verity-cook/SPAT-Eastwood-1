"""
SPAT-Eastwood-1
Verity Cook and Rebecca Dunkley

The functions we used to analyse the ESA Swarm data. 
"""

# import libraries 
import numpy as np
import pandas as pd
import datetime as dt
from scipy import signal, fftpack

from Calculating_FACs import single_sat_FAC
#%% Read Saved Data
def read_storm(file_path):
    """
    Creates DataFrame with datetime index from file
    Input:
        file_path - string of file path including '.csv'
    Output:
        DataFrame containing storm data
    """
    storm = pd.read_csv(file_path, index_col = 0)
    storm.index = pd.DatetimeIndex(storm.index)
    return storm

#%% Filtering the data
    
# low-pass filter to remove high frequency structures, suppress small scale FACS
def butterworth(cutoff, fs = 1, order = 2, btype = 'low'):
    """
    Butterworth filter
    Input:
        cuttoff - frequency -3dB cutoff
        fs      - frequency of sampling (1 sample per second = 1 Hz)
        order   - order of filter
        btype   - 'high', 'low' or 'band'
    Output:
        numerator and denominator arrays of IIR filter
        
    """
    nyq = fs/2 # nyquist frequency
    freq = cutoff/nyq
    b, a = signal.butter(order, freq, btype = btype)
    return b, a

# filter the data
def filter_data(data, period, order = 2, btype = 'low'):
    """
    Input:
        data   - data to be filtered
        period - cutoff period
        order  - order of filter
        btype  - 'high' or 'low', type of filter 
    Output:
        filtered data as array
    """
    cutoff = 1/period   # frequency = 1/period
    b, a = butterworth(cutoff, order = order)
    filtered = signal.lfilter(b, a, data) # apply filter to data
    return pd.Series(filtered, data.index)

#%% Finding Geomagnetic Storms 
def find_storm(Dst, threshold = -30):
    """
    Find start and end times of storm regions withtin the data
    Input:
        Dst       - pandas Series containing Dst values
        threshold - value of Dst below which it is considered a 
                    geomangetic storm
    Output:
        DataFrame containing: storm sudden commencement, start of storm (when 
        Dst falls below the threshold) and end of storm (when Dst rises 
        above threshold)
    """
    # get times for which the Dst is below the thresold
    storm_Dst = Dst[Dst < threshold]
    storm_time = storm_Dst.index
    
    # find start and end times of storm regions
    start = [storm_time[0]]
    end = []
    for t0, t1 in zip(storm_time, storm_time[1:]):
        # if neighouring data points are more than 24 hours apart
        if (t1 - t0).total_seconds() > 24 * 60 * 60:
            end.append(t0)
            start.append(t1)
    end = end + [storm_time[-1]]
    
    # find time of storm sudden commencement (SSC)
    # Dst in the 4 hours before start of each storm
    before_storm = [s - dt.timedelta(hours = 4) for s in start]
    Dst_before = [Dst[b:s] for b, s in zip(before_storm, start)]
    
    # find time (index) of maximum in Dst
    SSC = [d.idxmax() for d in Dst_before]
    
    return pd.DataFrame({'SSC': SSC, 'start': start, 'end':end})
    
#%% Sampling Rate 
    
def sampling(data, rate):
    """
    Input:
        data - Dataframe contining magnetic field and location data
        rate - sampling rate
    Output:
        maximum FAC calculated for that sampling rate
    """
    FAC = []
    for r in range(rate): # stagger the start
        data_rate = data.iloc[r::rate] # slice data
        FAC_data = single_sat_FAC(data_rate) # calculate FAC
        FAC.append(FAC_data.FAC_SS.abs().max()) # find maxmimum 
    return FAC




def sampling_example(data, rate):
    """
    Input:
        data - Dataframe contining magnetic field and location data
        rate - sampling rate
    Output:
        maximum FAC calculated for that sampling rate
    """
    FACs = {}
    for r in range(rate): # stagger the start
        data_rate = data.iloc[r::rate] # slice data
        FAC_data = single_sat_FAC(data_rate) # calculate FAC
        FACs['{}_{}'.format(r, rate)] = FAC_data.FAC_SS.values
        
    return FACs

#%% Comparing FAC to Storm Indicies

# Correlation
def correlation_single(FAC, storm_idx, interval):
    """
    Correlation for a single storm
    Input:
        FAC       - FAC values for single storm
        storm_idx - Dst or Kp values
        interval  - in hours, interval for which the value changes 
                   (1 for Dst, 3 for Kp)
    Output:
        Dst or Kp values with corresponding average,standard deviation, 
        minium and maximum of FAC values
    """
    # get index value over interval 
    idx_interval = storm_idx.groupby(pd.Grouper(freq='{}Min'.format(interval * 60))).mean()
    
    # get FACs over interval 
    FAC_abs = FAC.abs() # absolute value
    FAC_interval   = FAC_abs.groupby(pd.Grouper(freq='{}Min'.format(interval * 60)))
    
    # get FAC statistics  
    FAC_stats = [FAC_interval.mean(), 
                 FAC_interval.std(), 
                 FAC_interval.max(), 
                 FAC_interval.min()]
    
    return idx_interval, FAC_stats

def correlation_all(storms):
    """
    Correlation for all storms
    Input:
        FAC       - FAC values for all storms
        storm_idx - Dst or Kp values
        interval  - in hours, interval for which the value changes 
                   (1 for Dst, 3 for Kp)
    Output:
        Dst or Kp values with corresponding average,standard deviation, 
        minium and maximum of FAC values
    """
    Dsts = []
    FAC_ave_Dst = []
    FAC_std_Dst = []
    FAC_max_Dst = []
    FAC_min_Dst = []
    
    Kps  = []
    FAC_ave_Kp = []
    FAC_std_Kp = []
    FAC_max_Kp = []
    FAC_min_Kp = []
    for s in storms:
        storm_i = storms[s]
        
        Dst_i = storm_i.Dst
        Kp_i  = storm_i.Kp
        FAC_i = storm_i.FAC
        
        d, FAC_d = correlation_single(FAC_i, Dst_i, 1)
        k, FAC_k = correlation_single(FAC_i, Kp_i,  3)
        
        Dsts = Dsts + list(d)
        FAC_ave_Dst = FAC_ave_Dst + list(FAC_d[0])
        FAC_std_Dst = FAC_std_Dst + list(FAC_d[1])
        FAC_max_Dst = FAC_max_Dst + list(FAC_d[2])
        FAC_min_Dst = FAC_min_Dst + list(FAC_d[3])

        Kps = Kps + list(k)
        FAC_ave_Kp = FAC_ave_Kp + list(FAC_k[0])
        FAC_std_Kp = FAC_std_Kp + list(FAC_k[1])
        FAC_max_Kp = FAC_max_Kp + list(FAC_k[2])
        FAC_min_Kp = FAC_min_Kp + list(FAC_k[3])    
        print(s)

    stats_Dst = {}
    stats_Dst['Dst'] = Dsts
    stats_Dst['FAC_ave']= FAC_ave_Dst       
    stats_Dst['FAC_std']= FAC_std_Dst       
    stats_Dst['FAC_max']= FAC_max_Dst       
    stats_Dst['FAC_min']= FAC_min_Dst       

    stats_Kp = {}
    stats_Kp['Kp'] = Kps
    stats_Kp['FAC_ave']= FAC_ave_Kp       
    stats_Kp['FAC_std']= FAC_std_Kp       
    stats_Kp['FAC_max']= FAC_max_Kp       
    stats_Kp['FAC_min']= FAC_min_Kp         
    
    stats_Dst = pd.DataFrame(stats_Dst)
    stats_Kp = pd.DataFrame(stats_Kp)
    
    bins_d = np.arange(-220,100, 5)
    labels_d = np.arange(-217.5, 95.5, 5)
    stats_Dst['binned'] = pd.cut(stats_Dst['Dst'], bins=bins_d, labels=labels_d)
   
    bins_k = np.arange(0,10,0.5)
    labels_k = np.arange(0.25,9.75,0.5)
    stats_Kp['binned'] = pd.cut(stats_Kp['Kp'], bins=bins_k, labels=labels_k)
    
    FAC_ave_d = stats_Dst.groupby('binned')['FAC_max'].mean()
    FAC_std_d = stats_Dst.groupby('binned')['FAC_max'].std()
    
    FAC_ave_k = stats_Kp.groupby('binned')['FAC_max'].mean()
    FAC_std_k = stats_Kp.groupby('binned')['FAC_max'].std()
    
    return [labels_d, FAC_ave_d, FAC_std_d], [labels_k, FAC_ave_k, FAC_std_k]

# Time of maximum
def time_of_max(storms):
    """
    Input:
        storms - Dataframe for each storm
        
    Output:
        DataFrame containing time of SSC, absolute max FAC, min Dst, max Kp 
        and the actual values of the absolute max FAC, min Dst and max Kp 
    """
    time_SSC = []
    time_FAC = []
    time_Dst = []
    time_Kp  = []
    max_FAC = []
    max_Dst = []
    max_Kp = []
    for s in storms:
        storm = storms[s]
        time_SSC.append(storm.index[0])
        time_FAC.append(storm.FAC.abs().idxmax()) # absolute maximum FAC
        time_Dst.append(storm.Dst.idxmin())       # miniumum Dst
        time_Kp.append(storm.Kp.idxmax())         # maximum Kp
        max_FAC.append(storm.FAC.abs().max())
        max_Dst.append(storm.Dst.min())
        max_Kp.append(storm.Kp.max())
    diff_FAC = [(t_f - t0).total_seconds() for t_f, t0 in zip(time_FAC, time_SSC)]
    diff_Dst = [(t_d - t0).total_seconds() for t_d, t0 in zip(time_Dst, time_SSC)]
    diff_Kp  = [(t_k - t0).total_seconds() for t_k, t0 in zip(time_Kp,  time_SSC)]
    
    times = {}
    times['t_SSC'] = time_SSC
    times['t_FAC'] = time_FAC
    times['t_Dst'] = time_Dst
    times['t_Kp']  = time_Kp 
    times['diff_FAC'] = diff_FAC
    times['diff_Dst'] = diff_Dst
    times['diff_Kp']  = diff_Kp
    times['FAC'] = max_FAC
    times['Dst'] = max_Dst
    times['Kp'] = max_Kp
    return pd.DataFrame(times)
     
#%% Location

# Local Time Longitude
def LTL(Lon):
    """
    Calculate caluclate local time longitude
    Input:
        Lon - geographical longitude
    Output:
        local time longitude
    """
    ut = Lon.index # universal time
    
    # number of seconds since midnight
    t = ut.hour * 3600 + ut.minute * 60 + ut.second + ut.microsecond / 1000000.0
    LTLon = Lon + (t/86400) * 360 # calcualte local time longitude
    return LTLon

def convert_for_plot(Lat, Lon):
    lat_ = 90 - abs(Lat)
    lon_ = Lon%360
    lon_ = np.deg2rad(Lon)
    return lat_, lon_

def Largest_North_South(storms, n = 1):
    """
    Get position and value of largest n FAC during storm and split data into 
    northern and southern hemisphere
    Input:
        storms - DataFrame containing Swarm data
        n      - number of largest FAC values to take
    Output:
        a DataFrame for each of the northern and southern hemispheres
    """
    max_time = []
    max_FAC = []
    Lat = []
    Lon = []
    for s in storms:
        storm = storms[s]
        largest = storm.nlargest(n, 'FAC') # rows for largest n FAC
        smallest = storm.nsmallest(n, 'FAC')
        max_time = max_time + list(largest.index) +  list(smallest.index)
        max_FAC = max_FAC + list(largest.FAC) + list(smallest.FAC)
        Lat = Lat + list(largest.Latitude) + list(smallest.Latitude)
        LoniL = largest.Longitude
        LoniL = LTL(LoniL) # convert longitudes to local time longitudes
        LoniS = smallest.Longitude
        LoniS = LTL(LoniS)
        Lon = Lon + list(LoniL) + list(LoniS)
    
    data = pd.DataFrame({'FAC': max_FAC, 'Lat': Lat, 'LTLon': Lon}, 
                        index = pd.DatetimeIndex(max_time))
    
    N_data = data[data.Lat >= 0]
    S_data = data[data.Lat < 0]

    LatN, LonN = convert_for_plot(N_data.Lat, N_data.LTLon)
    LatS, LonS = convert_for_plot(S_data.Lat, S_data.LTLon)
    
    N = N_data.copy()
    N['Lat'] = LatN
    N['LTLon'] = LonN
    
    S = S_data.copy()
    S['Lat'] = LatS
    S['LTLon'] = LonS
    return N, S

#%% Cumulative Distribution of FACs
    
def cumulative_dist(storm, j0 = 0.5):
    """
    Input:
        storm - Dataframe containing Swarm data
        j0    - FAC threshold 
    Output:
        FAC values and their rank, power-law index and it's error
    """
    FAC = storm.FAC.abs()
    j = FAC[FAC > j0] # only consider FAC values above threshold
    
    # group FACs
    start = [j.index[0]]
    end = []    
    for t0, t1 in zip(j.index, j.index[1:]):
        if (t1 - t0).total_seconds() > 20: # 20 seconds for same FAC event
            end.append(t0)
            start.append(t1)
    end.append(j.index[-1])
    
    # fing peaks
    j_peaks = [j[s:e].max() for s, e in zip(start, end)]
    N = len(j_peaks) # number of data points
   
    j_sort = np.array(sorted(j_peaks, reverse = True))
    n = np.arange(1, len(j_sort) + 1)
    
    delta = N/sum(np.log(j_sort/j0)) 
    sigma_delta = (delta - 1) * N**(-0.5)
 
    return pd.DataFrame({'j':j_sort,'n': n, 'delta': delta, 'sigma_delta':sigma_delta})
    
#%% Power Spectrum of FACs
def power_spectrum(data, f_s = 1):
    """
    Input:
        data - signal
        f_s  - sampling rate of the signal
    Output:
        frequencies and power specturm
    """
    data = data[np.isnan(data) == False] # get rid of nan values
    X = fftpack.fft(data) # calculate fourier transform
    Ps = np.abs(X)**2 # find power spectrum
    f = fftpack.fftfreq(len(data)) * f_s # find frequencies
    
    power = pd.DataFrame({'f':np.array(f), 'Ps':np.array(Ps)})
    
    # remove negative data points 
    power = power[power.f > 0]
    
    return power

def lower_fit(storm):
    power = power_spectrum(storm.FAC)
    
    turb = power[power.f > 0.003]
    
    bins_f = np.logspace(np.log(0.003), np.log(0.5), 100, base = np.e)
    labels_f = np.logspace(np.log(0.003), np.log(0.5), 99, base = np.e)
    
    turb_min = turb.copy() 
    turb_min['binned'] = pd.cut(turb_min['f'], bins=bins_f, labels = labels_f )
    turb_min = turb_min.sort_values('Ps').groupby('binned', as_index=False).first()
    f_min = turb_min.f
    Ps_min = turb_min.Ps
    
    a, b = np.polyfit(np.log(f_min), np.log(Ps_min), 1)
    yhat = -5/3 * np.log(f_min) + b                     
    f_log = np.log(f_min)
    ybar = np.sum(f_log)/len(f_log)          
    ssreg = np.sum((yhat-ybar)**2)  
    sstot = np.sum((f_log - ybar)**2)
    R_squared = ssreg / sstot
    return yhat, f_log, f_min, Ps_min, power, R_squared


















