import os
import numpy as np
import pandas as pd
import pickle

from astropy.io import fits

#with fits.open('PYSME-v0.3.fits') as hdul1:
##    with fits.open('pysme_WEAVE.fits') as hdul2:
#    print(hdul1[1].columns[:5])
#
#raise

path_1 = 'OUTPUT/VARIABLES/NGC_6791/'
path_2= 'OUTPUT/VARIABLES/NGC_6939/'
path_3 = 'OUTPUT/VARIABLES/gums_229/'
# Load in fits template
#template = fits.open('PYSME-v0.3.fits')
#print(template.info())
#print(template[1].header)
#template.close()
#raise
# Load in files recursively
files_1 = os.listdir(path_1)
conv_ind_1 = np.array([x.find('conv') for x in files_1])
balmer_ind_1 = np.array([x.find('Balmer') for x in files_1])
test_ind_1 = np.array([x.find('test') for x in files_1])

files_2 = os.listdir(path_2)
conv_ind_2 = np.array([x.find('conv') for x in files_2])
balmer_ind_2 = np.array([x.find('Balmer') for x in files_2])
test_ind_2 = np.array([x.find('test') for x in files_2])

files_3 = os.listdir(path_3)
conv_ind_3 = np.array([x.find('conv') for x in files_3])
balmer_ind_3 = np.array([x.find('Balmer') for x in files_3])
test_ind_3 = np.array([x.find('test') for x in files_3])

run_files_1 = np.array(files_1)[(balmer_ind_1 == -1) 
                                    & (test_ind_1 == -1) 
                                    & (conv_ind_1 != -1)]

run_files_2 = np.array(files_2)[(balmer_ind_2 == -1) 
                                    & (test_ind_2 == -1) 
                                    & (conv_ind_2 != -1)]

run_files_3 = np.array(files_3)[(balmer_ind_3 == -1) 
                                    & (test_ind_3 == -1) 
                                    & (conv_ind_3 != -1)]

# Load in files
dictlist_1 = [pickle.load(open(path_1 + x,'rb')) for x in run_files_1]
dictlist_2 = [pickle.load(open(path_2 + x,'rb')) for x in run_files_2]
dictlist_3 = [pickle.load(open(path_3 + x,'rb')) for x in run_files_3]

#dictlist_1.update({'WPROV':'stacked_1002118.fit__stacked_1002117.fit'})
#dictlist_2.update({'WPROV':'stacked_1002154.fit__stacked_1002153.fit'})
#dictlist_3.update({'WPROV':'stacked_1003030.fit__stacked_1003029.fit'})

dictlist = dictlist_1 + dictlist_2 + dictlist_3

# Extract variables
# make arrays
teff = [x['effective_temperature'] for x in dictlist]
teff_err = [np.sqrt(abs(x['fit_results']['covariance'][0][0])) for x in dictlist]
logg = [x['gravity'] for x in dictlist]
logg_err = [np.sqrt(abs(x['fit_results']['covariance'][1][1])) for x in dictlist]
feh = [x['metallicity'] for x in dictlist]
feh_err = [np.sqrt(abs(x['fit_results']['covariance'][2][2])) for x in dictlist]
vsini = [x['rotational_velocity'] for x in dictlist]
vsini_err = [np.sqrt(abs(x['fit_results']['covariance'][3][3])) for x in dictlist]
CNAME = [x['CNAME'] for x in dictlist]
WPROV = \
   ['stacked_1002118.fit__stacked_1002117_exp_b_APS.fits' for x in dictlist_1] \
 + ['stacked_1002154.fit__stacked_1002153_exp_b_APS.fits' for x in dictlist_2] \
 + ['stacked_1003030.fit__stacked_1003029_exp_b_APS.fits' for x in dictlist_3]
res_mean = [np.mean(x['fit_results']['residuals']) for x in dictlist]
chisq = [x['fit_results']['chisq'] for x in dictlist]

# append to fits file
with fits.open('PYSME-v0.3.fits') as hdul1:
#    with fits.open('pysme_WEAVE.fits') as hdul2:
    nrows1 = hdul1[1].data.shape[0]
    nrows2 = len(teff)
    nrows = nrows1 + nrows2
    primary_hdu = fits.PrimaryHDU(data=hdul1[0].data, header=hdul1[0].header)
    bintable_hdu = fits.BinTableHDU.from_columns(hdul1[1].columns,
                                                 name=hdul1[1].name,
                                                 nrows=nrows)
    bintable_hdu.data['PYSME_TEFF'][nrows1:] = teff
    bintable_hdu.data['PYSME_TEFF_ERR'][nrows1:] = teff_err
    bintable_hdu.data['PYSME_LOGG'][nrows1:] = logg 
    bintable_hdu.data['PYSME_LOGG_ERR'][nrows1:] = logg_err
    bintable_hdu.data['PYSME_FEH'][nrows1:] = feh
    bintable_hdu.data['PYSME_FEH_ERR'][nrows1:] = feh_err
    bintable_hdu.data['PYSME_MICRO'][nrows1:] = vsini
    bintable_hdu.data['PYSME_MICRO_ERR'][nrows1:] = vsini_err
    bintable_hdu.data['CNAME'][nrows1:] = CNAME 
    bintable_hdu.data['WPROV'][nrows1:] = WPROV 
    bintable_hdu.data['PYSME_VBROAD'][nrows1:] = res_mean 
    bintable_hdu.data['PYSME_VBROAD_ERR'][nrows1:] = chisq
    hdu = fits.HDUList([primary_hdu, bintable_hdu])

#    for col in hdu[1].columns

#print(hdu[1].data.columns)
print(hdu[1].data[0:10])
hdu.writeto('debug_pysme_WEAVE_gs.fits', overwrite=True)
#print(hdu[1].header)
