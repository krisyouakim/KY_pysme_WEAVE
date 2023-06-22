from astropy.io import fits
from astropy.table import Table 
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_spectra(hdulist, index, line_mask, segm_mask, 
                                 full_line_mask, new_segm_mask,
                                 w_range, balmer_zoom=False, arm='R'): 
    
    cname = hdulist[2].data['CNAME'][index]
    spec_ind = np.where(hdulist[4].data['CNAME'] == cname)
    RVS_ind = np.where(hdulist[5].data['CNAME'] == cname)

#    spectra_rvs = hdulist[4].data['LAMBDA_RR_R']

    # get radial velocity of star
    rv_star = hdulist[2].data['VRAD'][index]
    
    wav_RVS = hdulist[5].data['LAMBDA_RVS_%s' %arm][RVS_ind][0]
    flux_RVS = hdulist[5].data['FLUX_RVS_%s' %arm][RVS_ind][0]


    # compute radial velocity correction ---> lambda = lamda_obs/(1-Vr/c)
    rv_corr = (1 + rv_star/(3.0*10**5))

    print(cname)
    print(hdulist[4].data['CNAME'][spec_ind])
    print(hdulist[5].data['CNAME'][RVS_ind])

    fig = plt.figure(figsize=(25,20))

    for i in range(len(w_range)-1):

        ax = fig.add_subplot(int((len(w_range)-1) / 2) 
                                  + int((len(w_range)-1) % 2), 
                                      2 , i + 1)
        
    
#        if balmer_zoom:
#    
#            if arm == 'B':
#            #H_gamma
#                ax.set_xlim([4335,4345])
#                ax.axvline(x=4340.72, color='red', linestyle='--')
#    
#            if arm == 'G':
#            #H_beta
#                ax.set_xlim([4855,4866])
#                ax.axvline(x=4861.35, color='red', linestyle='--')
#    
#            if arm == 'R':
#            #H_alpha
#                ax.set_xlim([6555,6569])
#                ax.axvline(x=6562.79, color='red', linestyle='--')
    
    #    ax.plot(wav, flux, color='k')
    #    ax.plot(wav / rv_corr, flux, color='orange')
        ax.plot(wav_RVS / rv_corr, flux_RVS, color='m')
    
        for start, end in zip(full_line_mask['start'].values, 
                                 full_line_mask['end'].values):
            ax.axvspan(start, end, alpha=0.1, color='blue', zorder=1)

        for start, end in zip(line_mask['start'].values, line_mask['end'].values):
            ax.axvspan(start, end, alpha=0.1, color='red', zorder=2)
    
#        for seg_start, seg_end in zip(segm_mask[segm_mask.source==1]['start'], 
#                                        segm_mask[segm_mask.source ==1]['end']):
#            ax.axvline(seg_start, linestyle='--')
#            ax.axvline(seg_end, linestyle='-', linewidth=2)
#    
#        for seg_start, seg_end in zip(segm_mask[segm_mask.source==2]['start'], 
#                                        segm_mask[segm_mask.source ==2]['end']):
#            ax.axvline(seg_start, linestyle='--', color='r')
#            ax.axvline(seg_end, linestyle='-', color='r', linewidth=2)
    
        for seg_start, seg_end in zip(new_segm_mask[new_segm_mask.source==1]['start'], 
                                        new_segm_mask[new_segm_mask.source ==1]['end']):
            ax.axvline(seg_start, linestyle='--')
            ax.axvline(seg_end, linestyle='-', linewidth=4)
    
        for seg_start, seg_end in zip(new_segm_mask[new_segm_mask.source==2]['start'], 
                                        new_segm_mask[new_segm_mask.source ==2]['end']):
            ax.axvline(seg_start, linestyle='--', color='r')
            ax.axvline(seg_end, linestyle='-', color='r', linewidth=4)

        if i % 2 == 0:
            ax.set_ylabel(r'flux', size = 20)

        if i > (len(w_range) - 4):
            ax.set_xlabel(r'wavelength ($\AA$)', size = 20)
    
    #    ax.set_xlim(w_range[i],w_range[i+1])
        ax.set_xlim(w_range[i],w_range[i+1])
    
#        ax.tick_params(axis="x", labelsize=20)
#        ax.tick_params(axis="y", labelsize=20)
    
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.93, 
                            wspace=0.05, hspace=0.1)

if __name__ == '__main__':

#    path = "/crex/proj/snic2020-16-23/shared/WAS/aps/v2.0/"
#    path = "/home/kryo6156/work/data/WEAVE/golden_sample/CCG_NGC6791_F1W1/HRB/"

    # CCG_NGC6791_HR Blue Red 
#    hdulist = fits.open('DATA/stacked_1002082__stacked_1002081_exp_b_APS.fits')
    # CCG_NGC6791_HR Green Red
#    hdulist = fits.open('DATA/stacked_1002118__stacked_1002117_exp_b_APS.fits')
    # gums_229.66_b43.5_HR Green Red
    hdulist = fits.open('DATA/stacked_1003030__stacked_1003029_exp_b_APS.fits')

    line_mask = pd.read_csv('DATA/LINELIST/new_ges_WEAVE_G_R_Sp.dat',
                                delimiter='\t',
                                names=['peak', 'start', 'end'], 
                                usecols=[0,1,2],
                                comment=';')

    segm_mask = pd.read_csv('DATA/LINELIST/ges_WEAVE_G_R_Segm.csv', 
#                                delimiter='\t',
                                names=['start', 'end', 'source'], 
                                usecols=[0,1,3],
                                comment=';')

    full_line_mask = pd.read_csv('DATA/LINELIST/full_ges_WEAVE_G_R_Sp.dat',
                                delimiter='\t',
                                names=['peak', 'start', 'end'], 
                                usecols=[0,1,2],
                                comment=';')

    new_segm_mask = pd.read_csv('DATA/LINELIST/new_ges_WEAVE_G_R_Segm.csv', 
#                                delimiter='\t',
                                names=['start', 'end', 'source'], 
                                usecols=[0,1,3],
                                comment=';')

#    #Calculate the min and max wavelengths of the spectra
#    wav_val = hdulist[4].data['LAMBDA_STELLAR'][0] 

    arm = 'G' # ('G' | 'R')

    if arm == 'R':
        w_range = np.arange(5980,6950, 100)

    if arm == 'G':
        w_range = np.arange(4770,5500, 100)

    print('data_loaded')

#    res_hist(path)
    plot_spectra(hdulist, 750, line_mask, segm_mask, 
                              full_line_mask, new_segm_mask,
                              w_range, False, arm)
    plt.show()
