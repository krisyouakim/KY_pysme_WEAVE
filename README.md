# pysme_WEAVE

This code runs the python version of Spectroscopy Made Easy (pySME) for WEAVE
spectra.

THINGS TO FIX EVENTUALLY
1. Do more tests on the extinction function in pysme_gaia_dist_extinc.py
    -check to see the discrepancy between the values from gaia source catalogue
    and those from the Green et al 2019 dustmaps for stars at different 
    temperatures
    -Put in checks to make sure the queried gaia parameters are correct

2. Suppress warnings about config files from dustmaps package 
    (line 205 in pysme_gaia_dist_extinc.py)

3. Double check that the resolution is treated properly in the interpolation 
    function on line 223 in pysme_WEAVE.py

4. I think the spectra are already radial velocity corrected. Double check this
    to make sure, otherwise revisit line 359 in pysme_makestruct.py

5. Improve wavelengths_flux_inside_segments function on line 544 of 
    pysme_makestruct.py. Function works but can be cleaned up, although it 
    probably won't make a diff for runtime. Only do this if there is time.

6. Improve segmentation routine in pre_normalise function, line 711 in 
    pysme_makestruct.py

7. Turn on NLTE functionality once you have grids for this 
    (line 100 of pysme_run_sme.py)

8. Include line broadening profile. The file is given in the L1 product header,
    hdulist[0].header['PROFILES'], but they are not currently provided in the
    golden sample

9. Clean up the Temp_Balmer_spectra load and delete thing...this should be done
    a better way
