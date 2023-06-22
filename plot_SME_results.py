import matplotlib.pyplot as plt
import numpy as np
import pickle

path = '/crex/proj/snic2020-16-23/shared/KY_pysme_WEAVE/OUTPUT/SPECTRA/'


def find_mask_bounds(mask, wav):
    # This function finds the beginnning and end points of the regions of the
    # segement mask
    
    # shift the indices by one
    shift_mask = np.insert(mask,0,0)
    sub_mask = np.append(mask, 0) - shift_mask

    range_ind = np.vstack([np.where(sub_mask == 1)[0]-1, 
                               np.where(sub_mask == -1)[0]-1]).T
#    range_ind = np.vstack([np.where(sub_mask > 0)[0]-1, 
#                               np.where(sub_mask < 0)[0]-1]).T

    return wav[range_ind]


#def plot_spectra(spec, var, segm): 
def plot_spectra(spec, var, run_type, plot_seg): 
#                     wav_min=np.min(wav_val)-1, wav_max=np.max(wav_val)+1):

    # Set the segments to be plotted based on input params
    if run_type == 'G_R':
        if plot_seg == 'green':
            plot_range = range(0,12)
        elif plot_seg == 'red':
            plot_range = range(12,23)
    
    elif run_type == 'test':
        plot_range = range(0,2)

    if run_type == 'G_R':
        fig = plt.figure(figsize=(24,18))

    elif run_type == 'test':
        fig = plt.figure(figsize=(20,5))

#    for segm in range(1,len(var['radial_velocity_global']-1)):
#    for i,segm in enumerate(range(13,25)):
    for i,segm in enumerate(plot_range):
#    for i,segm in enumerate(range(len(spec['wave']))):
        if run_type == 'G_R':
            ax = fig.add_subplot(4,3,i+1)

        elif run_type == 'test':
            ax = fig.add_subplot(1,2,i+1)
        
        wav = spec['wave'][segm]
        flux = spec['flux'][segm]
    
        synth_flux = spec['synth'][segm]
        print(segm, len(wav), len(flux), len(synth_flux))
#        print(var['radial_velocity_global'][segm])
    
        # compute radial velocity correction ---> lambda = lamda_obs/(1-Vr/c)
        sme_rv_corr = (1 + var['radial_velocity_global'][segm]/(3.0*10**5))
    #    rv_corr = (1 + -36.82/(3.0*10**5))
        rv_corr = (1 + 80.90/(3.0*10**5))
    #    rv_corr = (1 + -33.99/(3.0*10**5))
        
        # plot data spectrum
    #    ax.plot(wav / rv_corr, flux, color='m')
        ax.plot(wav / sme_rv_corr, flux, color='k', zorder=3, lw=3)
    #    ax.plot(wav / rv_corr, flux, color='k', zorder=3, lw=3)
    #    ax.plot(wav, flux, color='k', zorder=3, lw=3)
        # plot synthetic spectrum
    #    ax.plot(wav / rv_corr, synth_flux, color='b')
        ax.plot(wav / sme_rv_corr, synth_flux, color='orange', zorder=3, lw=3)
    #    ax.plot(wav, synth_flux, color='orange', zorder=3, lw=3)
    
        # plot data spectrum
    #    ax.plot(wav, flux, color='k')
    #    # plot synthetic spectrum
    #    ax.plot(wav, synth_flux, color='orange')
    
        # get begining and end points of mask ranges and colour these regions red
    #    mask_bounds = find_mask_bounds(spec['mask'][segm], wav / sme_rv_corr) 
        mask_bounds = find_mask_bounds(spec['mask'][segm], wav) 
        for reg in mask_bounds:
            ax.axvspan(reg[0], reg[1], alpha=0.1, color='red', zorder=1) 
    
        if i+1 > 9: 
            ax.set_xlabel(r'wavelength ($\AA$)', size = 20)
        
        if i % 3 == 0:
            ax.set_ylabel(r'flux', size = 20)
    
        if i == 1:
            plt.title('Teff %d --- Grav %.1f --- FeH %.2f' 
                        %(var['effective_temperature'], 
                          var['gravity'], var['metallicity']),
                            size=25)
    
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)


    plt.subplots_adjust(left=0.07, bottom=0.08, right=0.96, top=0.94, 
                            wspace=0.2, hspace=0.2)

if __name__ == '__main__':

#    path = "/crex/proj/snic2020-16-23/shared/WAS/aps/v2.0/"
    path = "/crex/proj/snic2020-16-23/shared/KY_pysme_WEAVE/OUTPUT/"

    obj_num = '09530987+0736056'
    obj_ind = 767
    mode = 'SME'# options are ('SME' | 'Balmer')
    run_type = 'G_R' # options are ('G_R' | 'test')
    plot_seg = 'red' # options are ('green' | 'red')

#    spec = pickle.load(open(path + \
#            'SPECTRA/conv_whole_%d_WVE_%s_' %(obj_ind, obj_num) \
#                    + 'WEAVE_HR_G_R_gums_229_%s_spectra.pkl' %mode, 'rb'))
#    
#    var = pickle.load(open(path + \
#            'VARIABLES/conv_whole_%d_WVE_%s_' %(obj_ind, obj_num) \
#                    + 'WEAVE_HR_G_R_gums_229_%s_variables.pkl' %mode, 'rb'))

    spec = pickle.load(open(path + \
            'SPECTRA/gums_229/conv_fix_%d_WVE_%s_' %(obj_ind, obj_num) \
          + 'WEAVE_HR_%s_Sp_%s_spectra.pkl' %(run_type, mode), 'rb'))
    
    var = pickle.load(open(path + \
            'VARIABLES/gums_229/conv_fix_%d_WVE_%s_' %(obj_ind, obj_num) \
          + 'WEAVE_HR_%s_Sp_%s_variables.pkl' %(run_type, mode), 'rb'))

    print(var['effective_temperature'], var['gravity'], var['metallicity'],
            var['radial_velocity_global'])
    print(np.mean(var['fit_results']['residuals']))

#    plot_spectra(spec, var, 2)
    plot_spectra(spec, var, run_type, plot_seg)
    plt.show()
