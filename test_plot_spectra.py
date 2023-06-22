from astropy.io import fits
from astropy.table import Table 
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle

#test = glob.glob('/crex/proj/snic2020-16-23/shared/WAS/aps/v2.0/20160903/*.fits')
#print(test)

def res_hist(path):

    # load in list of files   
#    os.chdir(os.path.join(PATH, 'earth-analytics'))
    night_list = np.loadtxt('nights.txt', dtype='str')
    res_list = []

    # loop through all files
    for nights in night_list:
        for spec in glob.glob(path + nights + '/*.fits'):
            hdulist = fits.open(spec)

            #store value for print(hdulist[0].header['MODE'])
            res_list.append(hdulist[0].header['MODE'])  
            hdulist.close()


    print(res_list)
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)

#    ax.hist(res_list, len(res_list[res_list=='HIGH_RES')
    ax.hist(res_list)
    
#    print(hdulist[0].header['MODE'])
#    print(hdulist[0].header['CAMERA'])

def plot_spectra(hdulist, sme_spec, synth_inp, index, balmer_zoom=False, arm='R'): 
#                     wav_min=np.min(wav_val)-1, wav_max=np.max(wav_val)+1):

#    print(synth_inp.keys())
#    print(synth_inp['vrad'])
#    print(synth_inp['wob'])
#    print(synth_inp['wind'])
#    print(synth_inp['vrad_flag'])
#    print(len(synth_inp['sob']))

    fig = plt.figure(figsize=(25,10))
    ax = fig.add_subplot(111)

    ind_1 = index
    ind_2 = 2
    
    cname_1 = hdulist[2].data['CNAME'][ind_1]
    spec_ind_1 = np.where(hdulist[4].data['CNAME'] == cname_1)
    RVS_ind_1 = np.where(hdulist[5].data['CNAME'] == cname_1)

    cname_2 = hdulist[2].data['CNAME'][ind_2]
    spec_ind_2 = np.where(hdulist[4].data['CNAME'] == cname_2)
    RVS_ind_2 = np.where(hdulist[5].data['CNAME'] == cname_2)

#    spectra_rvs = hdulist[4].data['LAMBDA_RR_R']

    # get radial velocity of star
    rv_star_1 = hdulist[2].data['VRAD'][ind_1]
    rv_star_2 = hdulist[2].data['VRAD'][ind_2]
    
    wav_1 = hdulist[4].data['LAMBDA_RR_%s' %arm][spec_ind_1][0]
    flux_1 = hdulist[4].data['FLUX_RR_%s' %arm][spec_ind_1][0]
    
    wav_2 = hdulist[4].data['LAMBDA_RR_%s' %arm][spec_ind_2][0]
    flux_2 = hdulist[4].data['FLUX_RR_%s' %arm][spec_ind_2][0]

    wav_RVS_1 = hdulist[5].data['LAMBDA_RVS_%s' %arm][RVS_ind_1][0]
    flux_RVS_1 = hdulist[5].data['FLUX_RVS_%s' %arm][RVS_ind_1][0]

    wav_RVS_2 = hdulist[5].data['LAMBDA_RVS_%s' %arm][RVS_ind_2][0]
    flux_RVS_2 = hdulist[5].data['FLUX_RVS_%s' %arm][RVS_ind_2][0]

#    wav_FER = hdulist[5].data['LAMBDA_FR_%s' %arm][RVS_ind][0]
#    flux_FER = hdulist[5].data['FLUX_FR_%s' %arm][RVS_ind][0]

    print(cname_1)
    print(hdulist[4].data['CNAME'][spec_ind_1])
    print(hdulist[5].data['CNAME'][RVS_ind_1]) 
    print(cname_2)
    print(hdulist[4].data['CNAME'][spec_ind_2])
    print(hdulist[5].data['CNAME'][RVS_ind_2])

    # compute radial velocity correction ---> lambda = lamda_obs/(1-Vr/c)
    rv_corr_1 = (1 + rv_star_1/(3.0*10**5))
    rv_corr_2 = (1 + rv_star_2/(3.0*10**5))
#    print(wav[0])
#    print(wav[-1])

    if balmer_zoom:

        if arm == 'B':
        #H_gamma
            ax.set_xlim([4335,4345])
            ax.axvline(x=4340.72, color='red', linestyle='--')

        if arm == 'G':
        #H_beta
            ax.set_xlim([4855,4866])
            ax.axvline(x=4861.35, color='red', linestyle='--')

        if arm == 'R':
        #H_alpha
            ax.set_xlim([6555,6569])
            ax.axvline(x=6562.79, color='red', linestyle='--')

#    ax.plot(wav, flux, color='k')
#    ax.plot(wav / rv_corr, flux, color='orange')
#    ax.plot(wav_RVS / rv_corr, flux_RVS, color='m')
    ax.plot(wav_RVS_1/rv_corr_1, flux_RVS_1, color='k')
#    ax.plot(wav_RVS_2/rv_corr_2, flux_RVS_2, color='y')
#    ax.plot(wav_FER, flux_FER, color='b')

    # Plot SME result (this is normalized)
#    ax.plot(sme_spec['wave'].flatten(), sme_spec['synth'].flatten(), 
#                color='y')
#    ax.plot(sme_spec['wave'].flatten()/(1 - 11.8/(3.0*10**5)), sme_spec['flux'].flatten(), 
#                color='m')
    # Plot the latest synth input
#    wave_flattened = [val for sublist in synth_inp['wave'] for val in sublist]
#    spec_flattened = [val for sublist in synth_inp['spec'] for val in sublist]
#    ax.plot(wave_flattened / (1 + synth_inp['vrad'][0]/(3.0*10**5)), 
#                synth_inp['sob'], 
#                color='y')
#    ax.plot(wave_flattened, spec_flattened, 
#                color='k')

    ax.set_xlabel(r'wavelength ($\AA$)', size = 20)
    ax.set_ylabel(r'flux', size = 20)

    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
#    ax.set_ylim([0,1.3])

    plt.subplots_adjust(left=0.074, bottom=0.12, right=0.96, top=0.88, 
                            wspace=0.2, hspace=0.2)

if __name__ == '__main__':

#    path = "/crex/proj/snic2020-16-23/shared/WAS/aps/v2.0/"
#    path = "/home/kryo6156/work/data/WEAVE/golden_sample/CCG_NGC6791_F1W1/HRB/"

    # CCG_NGC6791_HR Blue Red 
#    hdulist = fits.open('DATA/stacked_1002082__stacked_1002081_exp_b_APS.fits')
    # CCG_NGC6791_HR Green Red
#    hdulist = fits.open('DATA/stacked_1002118__stacked_1002117_exp_b_APS.fits')
    # gums_229.66_b43.5_HR Green Red
    hdulist = fits.open('DATA/stacked_1003030__stacked_1003029_exp_b_APS.fits')
#    sme_spec = pickle.load(open('OUTPUT/SPECTRA/conv_fix_2_WVE_09532626+0819028_WEAVE_HR_G_R_Sp_SME_spectra.pkl', 'rb')) 
#    sme_spec = pickle.load(open('OUTPUT/SPECTRA/gums_229/conv_fix_767_WVE_09530987+0736056_WEAVE_HR_G_R_Sp_SME_spectra.pkl', 'rb')) 
    sme_spec = pickle.load(open('OUTPUT/SPECTRA/gums_229/conv_fix_235_WVE_09564797+0715403_WEAVE_HR_G_R_Sp_SME_spectra.pkl', 'rb')) 
#    synth_inp = pickle.load(open('OUTPUT/latest_synth_input.pkl', 'rb')) 
    synth_inp = None
    
#    #Calculate the min and max wavelengths of the spectra
#    wav_val = hdulist[4].data['LAMBDA_STELLAR'][0] 
#    

#    res_hist(path)
    plot_spectra(hdulist, sme_spec, synth_inp, 387, False, arm='G') 
    plt.show()
