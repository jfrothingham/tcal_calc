import numpy as np
from astropy.io import fits
import astropy

import dysh
from dysh.fits.gbtfitsload import GBTFITSLoad


import glob
import os,sys
import subprocess

from astropy import units as u
from astropy import constants as ac

import matplotlib.pyplot as plt

import traceback
import logging

from tcal_calc.utils import *
import time


Rx_dict = {
    'C' : [2,2,2,2,4999.0],
    'L' : [1,1,2,2,1439.0],
    'P8': [2,1,2,2,799.0]
}



def parse_getForecastValues(stdout):
    step1 = str(stdout).split('\\n')
    step2=[]
    for ii,bob in enumerate(step1):
        try:
            step2.append(float(bob[-6:]))
        except:
            continue
    step3 = np.array(step2)
    return step3

def dumb(arr):
    out=''
    for i in range(len(arr)):
        out += f'{arr[i]} '
    return out[:-1]

def select_regions(psscan):
    #plt.plot(psscan.timeaverage())
    plt.ion()
    psscan.timeaverage().plot()
    
    includes=[]
    try:
        while True:
            start=time.time()
            pts = plt.ginput(1)
            vline = pts[0][0]
            includes.append(vline)
            plt.axvline(x=vline,c='k',linewidth=0.5)
            plt.pause(0.05)
            plt.show()
            if (time.time()-start > 60):
                break
    except:
        print(includes)
        plt.close()
        
    if len(includes) > 0:
        includes = freq_to_chan(includes,psscan)
    
        #make mask
        mask = np.ones(psscan.timeaverage().data.shape[0])
        for i in range(len(includes)//2):
            mask[includes[2*i]:includes[(2*i)+1]] = 0
    else:
        mask = np.zeros(psscan.timeaverage().data.shape[0])
    return mask


def freq_to_chan(includes,psscan):
    #get channel number from Hz
    spectrum = psscan.timeaverage()
    
    freqs= (np.array(includes,dtype=np.float64)/1e6)*u.MHz
    
    bw = np.abs(spectrum.frequency[-1] - spectrum.frequency[0]).to(u.MHz)
    v0 = np.min(spectrum.frequency).to(u.MHz)
    vmax = np.max(spectrum.frequency).to(u.MHz)
    nchan = spectrum.data.shape[0]
    
    #-1 if lsb, +1 if usb
    bw_dir = np.sign(spectrum.frequency[-1] - spectrum.frequency[0])
    
    if bw_dir:
        chans = ((freqs - v0) / bw) * nchan
    else:
        chans = ((vmax - freqs) / bw) * nchan
        
    #check for odd number of clicks
    if chans.size % 2 != 0:
        print('Warning: odd number of clicks, ignoring the last click')
        chans = chans[:-1]
        print(chans.size)
        
    #round and return
    return np.rint(chans).astype(np.int32)




def tcal_calc(sdf,onscan,offscan,mask = None,ifnum=0,plnum=0,fdnum=0,fileout='garbage.txt'):
    # access cal indices from tpscan
    tp_on = sdf.gettp(scan=onscan,plnum=plnum,ifnum=ifnum,fdnum=fdnum)[0]
    tp_off = sdf.gettp(scan=offscan,plnum=plnum,ifnum=ifnum,fdnum=fdnum)[0]

    #get the 4 sig/cal states
    #onsource_calon_indices = sdf.calonoff_rows(onscan,ifnum=ifnum,plnum=plnum,fdnum=fdnum)['ON']
    onsource_calon_indices = tp_on._calrows['ON']
    onsource_calon_chunk = sdf.rawspectra(0,0)[onsource_calon_indices]
    onsource_calon_data = np.mean(onsource_calon_chunk,axis=0)

    #onsource_caloff_indices = sdf.calonoff_rows(onscan,ifnum=ifnum,plnum=plnum,fdnum=fdnum)['OFF']
    onsource_caloff_indices = tp_on._calrows['OFF']
    onsource_caloff_chunk = sdf.rawspectra(0,0)[onsource_caloff_indices]
    onsource_caloff_data = np.mean(onsource_caloff_chunk,axis=0)

    #offsource_calon_indices = sdf.calonoff_rows(offscan,ifnum=ifnum,plnum=plnum,fdnum=fdnum)['ON']
    offsource_calon_indices = tp_off._calrows['ON']
    offsource_calon_chunk = sdf.rawspectra(0,0)[offsource_calon_indices]
    offsource_calon_data = np.mean(offsource_calon_chunk,axis=0)

    #offsource_caloff_indices = sdf.calonoff_rows(offscan,ifnum=ifnum,plnum=plnum,fdnum=fdnum)['OFF']
    offsource_caloff_indices = tp_off._calrows['OFF']
    offsource_caloff_chunk = sdf.rawspectra(0,0)[offsource_caloff_indices]
    offsource_caloff_data = np.mean(offsource_caloff_chunk,axis=0)

    #get a tpscan for metadata
    tpscan  = sdf.gettp(scan=onscan,plnum=plnum,ifnum=ifnum,fdnum=fdnum)
    
    num_chan = len(offsource_caloff_data)
    #need to get frequencies
    #is there a way to do this without the time average?
    freqs = np.flip(np.array( tpscan.timeaverage().frequency.to(u.MHz) ))
    
    if mask is not None:
        onsource_calon_data[mask==1] = np.nan
        onsource_caloff_data[mask==1] = np.nan
        offsource_calon_data[mask==1] = np.nan
        offsource_caloff_data[mask==1] = np.nan
        freqs[mask==1] = np.nan

    onscan_idx = sdf.get_summary().SCAN.eq(onscan).idxmax()
    offscan_idx = sdf.get_summary().SCAN.eq(offscan).idxmax()

    #need to get source and elevation from the scan?
    #fluxS_vctr = getFluxCalib(sdfAsum.iloc[onscan_idx]['OBJECT'],freqs)
    ApEff = getApEff(sdf.get_summary().iloc[onscan_idx]['ELEVATION'], freqs)

    #get MJD of observation for getForecastValues
    tp_spec = tpscan.timeaverage()
    
    
    #uncalibrated Tcal calculations (cal for the cal god)

    #Find mean/deviation of inner 80%
    calcounts_onsource=onsource_calon_data-onsource_caloff_data
    calcounts_offsource=offsource_calon_data-offsource_caloff_data
    sourcecounts_calon=onsource_calon_data-offsource_calon_data
    sourcecounts_caloff=onsource_caloff_data-offsource_caloff_data

    calonsource=calcounts_onsource/sourcecounts_calon
    caloffsource=calcounts_offsource/sourcecounts_caloff


    start_idx = int(0.1*num_chan)
    end_idx = int(0.9*num_chan)

    meancalonsource=np.nanmean(calonsource[start_idx:end_idx])
    meancaloffsource=np.nanmean(caloffsource[start_idx:end_idx])
    sdcalonsource=np.nanstd(calonsource[start_idx:end_idx])
    sdcaloffsource=np.nanstd(caloffsource[start_idx:end_idx])
    diffmean=np.abs(((meancalonsource-meancaloffsource)/meancaloffsource)*100.0)
    diffstd=np.abs(((sdcalonsource-sdcaloffsource)/sdcaloffsource)*100.0)
    Tcal=(calonsource+caloffsource)/2.0
    
    
    #check that data is ok
    if (diffmean > 1.0) or (diffstd > 5.0):
        print(f"Difference in means of Off/On scans:          {diffmean}")
        print(f"Difference in std deviations of Off/On scans: {diffstd}")
        print("Differences between On/Off cal measurements are large, this is probably not a good dataset for determining Tcals")
    else:
        print('data is OK to calibrate')
    
    Tsys_calon=offsource_calon_data/sourcecounts_calon
    Tsys_caloff=offsource_caloff_data/sourcecounts_caloff

    #uncalibrated vector Tcal/Tsys:

    #need to switch these to use sort vector from freqs
    #can't always guarantee lower sideband
    Tcal = np.flip(Tcal)
    Tsys_caloff = np.flip(Tsys_caloff)
    
    sdfsum = sdf.get_summary()
    
    #Calibrating the calibrations!
    #get source flux, but in Ta not Jy
    source = tp_spec.meta['OBJECT']
    fluxT_Vctr = compute_sed(freqs*u.MHz,'Perley-Butler 2017',source,units='K')


    #S_to_T = ApEff / 0.352
    #fluxT_Vctr = fluxS_Vctr * S_to_T * (u.K/u.Jy)


    TCal_Cal=Tcal*fluxT_Vctr
    TSys_Cal=Tsys_caloff*fluxT_Vctr
    print(f'Tsys: {np.nanmean(TSys_Cal[start_idx:end_idx])}')

    #AveEl is average elevation between the on/off scans
    AveEl = 0.5*(sdfsum.iloc[onscan_idx]['ELEVATIO'] + sdfsum.iloc[offscan_idx]['ELEVATIO'])
    AM=AirMass(AveEl)

    #string version of coarse frequencies in GHz for opacity corrections (1 MHz resolution)
    freqFC = (np.arange( np.round(np.nanmax(freqs)-np.nanmin(freqs)) ) + np.nanmin(freqs)) / 1000
    
    #print('gabagool!')
    mjd = astropy.time.Time(tp_spec.meta['DATE-OBS']).mjd

    arg = f'/users/rmaddale/bin/getForecastValues -type Opacity -freqList {dumb(freqFC)} -timeList {mjd}'
    #print(arg)
    Taucoarse = subprocess.run(arg.split(),stdout=subprocess.PIPE).stdout

    Taucoarse = parse_getForecastValues(Taucoarse)
    Tau = np.interp(freqs, freqFC*1000, Taucoarse)
    
    arg = f'/users/rmaddale/bin/getForecastValues -type AtmTsys -freqList {dumb(freqFC)} -elev {AveEl} -timeList {mjd}'
    #print(arg)
    Tsky_coarse = subprocess.run(arg.split(),stdout=subprocess.PIPE).stdout


    Tsky_coarse = parse_getForecastValues(Tsky_coarse)
    Tsky = np.interp(freqs, freqFC*1000, Tsky_coarse) * u.K


    Tbg = 2.73*u.K
    Tspill = 3.0*u.K

    Trx = TSys_Cal-Tsky-Tbg-Tspill

    #write_values to ascii
    
    return freqs, TCal_Cal, TSys_Cal, Trx






def tcal_master(session, offscan, plnum=0, fdnum=0,plot=True):

    fileout=f'Tcals_{session[5:]}.txt'
    print(fileout)
    #set up derived params
    band = session[session.rfind('_')+1:]
    #number of IFs to plot
    num_ifs_tot = Rx_dict[band][0]*Rx_dict[band][1]
    #number of sep IF configs in astrid script
    num_configs = Rx_dict[band][1]
    #number of separate sdfits files to load
    num_sdfs = Rx_dict[band][0]
    #number of scans to skip to get same pol type/diode level 
    num_scans_to_skip = 2*Rx_dict[band][2]*Rx_dict[band][3]

    inpath=f'/home/sdfits/{session}'
    inpath = '/home/scratch/esmith/TRCO_230718_C.raw.vegas/'
    infiles = glob.glob(inpath+'*fits')
    print(infiles)
    if len(infiles) != num_sdfs:
        print('num_sdfs mismatch: ',len(infiles),num_sdfs)
        
    out_specs = []
    out_freqs = []
    
    
    for i in range(num_sdfs):
        sdf = GBTFITSLoad(infiles[i])
        for j in range(num_configs):
            
            this_onscan = offscan + (num_scans_to_skip*j) + 1
            this_offscan = offscan + (num_scans_to_skip*j)
            try:
                print('==========================')
                print(f'ON scan: {this_onscan} / OFF scan: {this_offscan}',i)
                print(sdf,this_onscan,this_offscan,i,plnum,fdnum)
                
                #get inclusion regions for fitting (channels)
                mask = select_regions(sdf.getps(scan=this_onscan,plnum=plnum,fdnum=fdnum,ifnum=i))
                
                
                freqs, Tcal_Cal, Tsys_cal, Trx = tcal_calc(sdf,this_onscan,this_offscan,mask=mask,ifnum=i,plnum=plnum,fdnum=fdnum)
                print(freqs[0])
                out_specs.append(Tcal_Cal)
                out_freqs.append(freqs)
            except Exception as e:
                print(f'Data for scans {this_onscan}/{this_offscan} expected but not found - skipping...')
                print(e)
                
    if plot:
        for i in range(len(out_specs)):
            plt.plot(out_freqs[i],out_specs[i])

    return out_freqs,out_specs














