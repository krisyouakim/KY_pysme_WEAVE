"""File which contains funcitons that can be called to test the pipeline, such
as plotting the spectrum"""

import matplotlib.pyplot as plt
import numpy as np 

from astropy.io import fits

def plot_spectra(wav, flux, wav_range=(3650, 9600)):

    """
    Function to plot the spectrum in a given wavelength range
    Parameters                                                                  
    -------                                                                     
        wav: array_like
            Wavelengths of spectra                                        
        flux: array_like
            Flux of spectra                                              
        wav_range: tuple
            wavelength range to be plotted (min, max)
    """

    fig = plt.figure(figsize=(25,10))                                           
    ax = fig.add_subplot(111)                                                   
                                                                                 
#     spectra_rvs = hdulist[hdu_ind].data                                         
#                                                                                 
#     # get radial velocity of star                                               
#     rv_star = hdulist[8].data['VRAD_RVS']                                       
#                                                                                 
#     wav = hdulist[hdu_ind].data['LAMBDA_STELLAR'][index]                        
#     flux = spectra_rvs['FLUX_STELLAR'][0,:]                                     
#                                                                                 
#     # compute radial velocity correction ---> lambda = lamda_obs/(1-Vr/c)       
#     rv_corr = (1 + rv_star[index]/(3.0*10**5))                                  
#     print(rv_corr)                                                              
#     print(rv_star[index]/(3.0*10**5))                                           
#     print(wav[0])                                                               
#                                                                                 
#     ax.plot(wav / rv_corr, flux, color='k')                                     

    ax.plot(wav, flux, color='k')                                     
    ax.set_xlabel(r'wavelength ($\AA$)', size = 20)                             
    ax.set_ylabel(r'flux', size = 20)                                           

    ax.set_xlim(wav_range)

    # vertical lines at rest wav of CaHK lines
    ax.axvline(3934.78)
    ax.axvline(3969.59)
                                                                                 
    ax.tick_params(axis="x", labelsize=20)                                      
    ax.tick_params(axis="y", labelsize=20)                                      
                                                                                 
    plt.subplots_adjust(left=0.074, bottom=0.12, right=0.96, top=0.88,          
                             wspace=0.2, hspace=0.2)

    plt.show()

if __name__ == "__main__":

    np.random.seed(12345)
    
    path = "/home/kryo6156/work/data/WEAVE/" 
    hdulist = fits.open(path + 'stacked_1003138_1003137.aps.fits')
    hdu_ind = 4

    # get radial velocity of star                                               
    rv_star = hdulist[8].data['VRAD_RVS']                                       

    fig = plt.figure(figsize=(10,18))

#    for i,ind in enumerate(np.random.choice(range(
#                              len(hdulist[hdu_ind].data['LAMBDA_STELLAR']))
#                              ,10)):

    indices = np.where(rv_star > 3.75)
    print(indices)
    print(indices[0][0])

    for i,ind in enumerate(indices[0]):

        wav = hdulist[hdu_ind].data['LAMBDA_STELLAR'][ind]
        flux = hdulist[hdu_ind].data['FLUX_STELLAR'][ind]

        print(i, ind, rv_star[ind])
   
        # compute radial velocity correction ---> lambda = lamda_obs/(1-Vr/c)       
        rv_corr = (1 - rv_star[ind]/(3.0*10**5))                                  
        
        print(wav)
        print(flux)
        ax = fig.add_subplot(10,1,i+1)
        ax.plot(wav / rv_corr, flux, color='k')
#        ax.plot(wav, flux, color='k')

        if i+1 == 5:
            ax.set_ylabel(r'flux', size = 20)                                           

        elif i+1 == 10:
            ax.set_xlabel(r'wavelength ($\AA$)', size = 20)                             

        ax.set_xlim((6200, 6400))
    
        # vertical lines at rest wav of CaHK lines
        ax.axvline(3934.78)
        ax.axvline(3969.59)

    plt.subplots_adjust(left=0.1, bottom=0.08, right=0.96, top=0.95,          
                             wspace=0.2, hspace=0.2)

    plt.show()
