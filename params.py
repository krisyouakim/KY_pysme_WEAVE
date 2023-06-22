# Parameter file to be used as input for the WEAVE pySME code
# Please fill in all parameters

# path to the data file containing the WEAVE spectrum
path = '/proj/snic2020-16-23/shared/KY_pysme_WEAVE/DATA/'
# Which field (pointing) the object is in (gums_229 | NGC_6791 | NGC_6939)
field = 'gums_229'
if field == 'NGC_6791':
    #CCG_NGC6791_HRG
    # File containting the blue and red wavelength regions
    spec_file_blue = 'stacked_1002082__stacked_1002081_exp_b_APS.fits'
    # File containting the green wavelength region
    spec_file_green = 'stacked_1002118__stacked_1002117_exp_b_APS.fits'
    # L1 stack or superstack file with the extra header information
    head_file = 'stack_1002081.fit'
elif field == 'NGC_6939':
    # NGC_6939_HRG
    # File containting the blue and red wavelength regions
    spec_file_green = 'stacked_1002154__stacked_1002153_exp_b_APS.fits'
    # L1 stack or superstack file with the extra header information
    head_file = 'stack_1002154.fit'
elif field == 'gums_229':
    # gums_229.66_b43.5_HRG
    # File containting the blue and red wavelength regions
    spec_file_blue = 'stacked_1002958__stacked_1002957_exp_b_APS.fits'
    # File containting the green wavelength region
    spec_file_green = 'stacked_1003030__stacked_1003029_exp_b_APS.fits'
    # L1 stack or superstack file with the extra header information
    head_file = 'stack_1003030.fit'
field_for_obs = 'free'
field_end = 'free'
# Which segment of HR data to use, must be one of (for testing us 'test', which
# uses small portion of the spectrum, with corresponding reduced line lists
# and line masks so the code can be debugged)
# ('R' | 'B' | 'G' | 'G_R' | 'stitched' | 'test')
wav_segment = 'G_R'
#wav_segment = 'test'
#wav_segment = 'test_green'
# number of iterations - nominaly four SME calls : 
# normalise, iterate, normalise, iterate
iterations = [1, 2, 1, 20]
#iterations = [1, 5]
atmosphere_grid_file = 'marcs2014.sav'
#line_list = 'reduced_WEAVE_HR_hfs.lin'
line_list = 'WEAVE_HR_hfs.lin'
segment_mask = 'new_ges_WEAVE_G_R_Segm.dat'
line_mask = 'new_ges_WEAVE_G_R_Sp.dat'
#segment_mask = 'test_green_ges_WEAVE_Segm.dat'
#line_mask = 'test_green_ges_WEAVE_Sp.dat'
#segment_mask = 'WEAVE_HR_R_Segm.dat'
#line_mask = 'WEAVE_HR_R_Sp.dat'
#segment_mask = 'test_WEAVE_HR_Segm.dat'
#line_mask = 'test_WEAVE_HR_Sp.dat'
cont_mask = 'WEAVE_HR_Cont.dat'
# NLTE file names. [0] must be H and [1] Fe, or change it in pysme_run_sme
atmosphere_abundance_grid = ['', '']
depthmin = 0
# Determines how the radial velocity is fitted
# ('none' | 'fix' | 'each' | 'whole')
vrad_flag = 'fix'
# Sets where the stdout and stderr is printed
# ('out_file' | 'term')
print_to = 'term'
