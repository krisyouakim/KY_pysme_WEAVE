"""
This file contains functions to query distance and extinction catalogues for
the WEAVE target star and are called from the pysme_WEAVE.py file.
"""
import numpy as np

import dustmaps
import astropy.units as units
import warnings

from astropy.coordinates import SkyCoord
from astropy.io import fits
from scipy.interpolate import interp1d
from astroquery.gaia import Gaia


def get_cbj_dist(head_file, head_ind):
    """
    Query distance for object computed with gaia edr3 from CBJ catalogue
    (Bailer-Jones C.~A.~L., Rybizki J., Fouesneau M., Demleitner M.,
    Andrae R., 2020, arXiv, arXiv:2012.05220 CBJ catalogue. First check for 
    photogeometric distance, and if it doesn't exist, then get the geometric 
    distance

    Parameters
    -------
        head_file: fits_hdu
            The data file containing extra information about the spectrum
        head_ind: Int
            The index corresponding to the object id
    Returns
    -------
        CBJ_dict: dictionary
             The relevant parameters from CBJ catalogue
    """
    # uncomment to see list of available tables at Gaia Tap+
    #tables = Gaia.load_tables(only_names=True)
    #for table in (tables):
    #    print(table.get_qualified_name())

    # uncomment to see a description of a specific table
    #tables = Gaia.load_tables()
    #table = Gaia.load_table('external.gaiaedr3_distance')
    #print(f"table = {table}")

    # Get gaia_id from the WEAVE file and search the catalogue for a matching 
    # id
    gaia_id = int(head_file[6].data['TARGID'][head_ind])
    # alternate gaia ids that do exist in the catalogue (for testing)
#    gaia_id=4144473463028744832
#    gaia_id=189578280197517824

    job = Gaia.launch_job("select * from external.gaiaedr3_distance \
                           where source_id = %d" %gaia_id)

    r = job.get_results()

    #create a dictionary of the distance parameters
    cbj_dict = {}

    #if a photogeometric distance is available, use that one, otherwise use
    #the purely geometric solution (flag ending in 99 means no reliabel gaia
    # photometry)
    if len(r) == 0:
        print("No object in CBJ catalogue with this gaia id.", \
              "Can not get distance.")
        cbj_dict['r_med'] = '--'
        cbj_dict['r_lo'] = '--'
        cbj_dict['r_hi'] = '--'

    elif (str(r['r_med_photogeo'].data[0]) == '--' or
          int(r['flag'].data[0][-2:]) == 99):

        print("Object found in CBJ catalogue with gaia id: %s" %gaia_id, \
              "Using geometric distance.")
        
        cbj_dict['r_med'] = float(r['r_med_geo'].data[0])
        cbj_dict['r_lo'] = float(r['r_lo_geo'].data[0])
        cbj_dict['r_hi'] = float(r['r_hi_geo'].data[0])

    else:
        print("Object found in CBJ catalogue with gaia id: %s" %gaia_id, \
              "Using photogeometric distance.")
        cbj_dict['r_med'] = float(r['r_med_photogeo'].data[0])
        cbj_dict['r_lo'] = float(r['r_lo_photogeo'].data[0])
        cbj_dict['r_hi'] = float(r['r_hi_photogeo'].data[0])

    return cbj_dict

def get_extinction(cbj_dict, head_file, head_ind):
    """
    Get extinction values for WEAVE star from Gaia. First check to see if there
    is a StarHorse entry. Then check the extinction maps from Green et al. 
    2019, and finally just use the value provided by the Gaia catalogue.

    Parameters
    -------
        cbj_dict: dict
            Dictionary containing distances from CBJ catalogue
        head_file: fits_hdu
            The data file containing all the information about the spectrum
        head_ind: Int
            The index corresponding to the object id
    Returns
    -------
        extinction_dict: dictionary
             The relevant parameters from CBJ catalogue
    """

    gaia_id = int(head_file[6].data['TARGID'][head_ind])
    # alternate gaia ids that do exist in the catalogue (for testing)
#    gaia_id=4144473463028744832
#    gaia_id=189578280197517824
    gaia_G_mag = float(head_file[6].data['MAG_GG'][head_ind])
    gaia_BP_RP = (float(head_file[6].data['MAG_BP'][head_ind]) 
                   - float(head_file[6].data['MAG_RP'][head_ind]))
    print(gaia_id)

    targ_ra = float(head_file[6].data['TARGRA'][head_ind])
    targ_dec = float(head_file[6].data['TARGDEC'][head_ind])

    #create a dictionary of the distance parameters
    extinction_dict = {}

    job = Gaia.launch_job("select top 10 phot_g_mean_mag, a_g_val,\
                           a_g_percentile_lower, a_g_percentile_upper,\
                           lum_val, lum_percentile_lower,\
                           lum_percentile_upper, bp_rp,\
                           e_bp_min_rp_val, e_bp_min_rp_percentile_lower,\
                           e_bp_min_rp_percentile_upper,\
                           parallax, parallax_error, ra, dec\
                           from gaiadr2.gaia_source \
                           where source_id = %d" %gaia_id)

    r = job.get_results()

    #in the ab
    if len(r) == 0:
        print("No object in gaia DR2 source catalogue with this gaia id.", \
              "Can not get line-of-sight extinction paramter a_g.")
        extinction_dict['gaia_id'] = '--'
        extinction_dict['plx'] = '--'
        extinction_dict['e_plx'] = '--'
        extinction_dict['a_g'] = '--'
        extinction_dict['a_g_lower'] = '--'
        extinction_dict['a_g_upper'] = '--'
        extinction_dict['lbol'] = '--'
        extinction_dict['lbol_lower'] = '--'
        extinction_dict['lbol_upper'] = '--'
        extinction_dict['bp_rp'] = '--'
        extinction_dict['bprp_red'] = '--'
        extinction_dict['bprp_red_lower'] = '--'
        extinction_dict['bprp_red_upper'] = '--'


    else:
        #quick check on the magnitude to see that the query got the right star 
        if float(r['phot_g_mean_mag'].data[0]) - gaia_G_mag > 0.5:
            print("incorrect object matched in gaia DR2 source catalogue,", \
                  "can not trust extinction parameter so returning -- instead")
            extinction_dict['gaia_id'] = '--'
            extinction_dict['plx'] = '--'
            extinction_dict['e_plx'] = '--'
            extinction_dict['a_g'] = '--'
            extinction_dict['a_g_lower'] = '--'
            extinction_dict['a_g_upper'] = '--'
            extinction_dict['lbol'] = '--'
            extinction_dict['lbol_lower'] = '--'
            extinction_dict['lbol_upper'] = '--'
            extinction_dict['bp_rp'] = '--'
            extinction_dict['bprp_red'] = '--'
            extinction_dict['bprp_red_lower'] = '--'
            extinction_dict['bprp_red_upper'] = '--'

        else:
            print('using gaiadr2.gaia_source for a_g')
            extinction_dict['gaia_id'] = gaia_id
            extinction_dict['plx'] = float(r['parallax'].data[0])
            extinction_dict['e_plx'] = float(r['parallax_error'].data[0])
            extinction_dict['a_g'] = float(r['a_g_val'].data[0])
            extinction_dict['a_g_lower'] = \
                                     float(r['a_g_percentile_lower'].data[0])
            extinction_dict['a_g_upper'] = \
                                     float(r['a_g_percentile_upper'].data[0])
            extinction_dict['lbol'] = float(r['lum_val'])
            extinction_dict['lbol_lower'] = float(r['lum_percentile_lower'])
            extinction_dict['lbol_upper'] = float(r['lum_percentile_upper'])
            extinction_dict['bp_rp'] = float(r['bp_rp'])
            extinction_dict['bprp_red'] = float(r['e_bp_min_rp_val'])
            extinction_dict['bprp_red_lower'] = \
                                       float(r['e_bp_min_rp_percentile_lower'])
            extinction_dict['bprp_red_upper'] = \
                                       float(r['e_bp_min_rp_percentile_upper'])

    ### if star does not have extinction available in Gaia DR2 then get it 
    # from 3D dustmaps using CBJ distances

    #get E(B-V) from Green et al 2019 dust maps for more details see: 
    # https://dustmaps.readthedocs.io/en/latest/

    try:
        from dustmaps.bayestar import BayestarWebQuery
        bayestar_found = True

    except ImportError:
        bayestar_found = False

    if bayestar_found == True and cbj_dict['r_med'] != '--':
        coords = SkyCoord(targ_ra*units.deg, targ_dec*units.deg, 
                             distance=cbj_dict['r_med']*units.pc )
        
        bswq = BayestarWebQuery(version='bayestar2019')

        ebv = bswq(coords, mode='median')
        ebv_lower, ebv_upper = bswq(coords, mode='percentile', pct=[16., 84.]) 

    else:
        ebv = '--'
        ebv_lower = '--'
        ebv_upper = '--'

    extinction_dict['ebv'] = ebv
    extinction_dict['ebv_lower'] = ebv_lower
    extinction_dict['ebv_upper'] = ebv_upper

    # Gaia extinction coefficients as a function of BP-RP colours computed
    # according to Casagrande et al 2020 (see Figure 1).
    if extinction_dict['a_g'] == '--' and ebv != '--':
        print('using bayestar for a_g')
        BP_RP_0 = gaia_BP_RP - 1.339 * ebv
        R_G = 3.068 - 0.504 * BP_RP_0 + 0.53 * BP_RP_0**2
        extinction_dict['a_g'] = ebv * R_G
        print(extinction_dict['a_g'])
        # a_g computed using relations described in Arentsen et al 2020 
        # (PIGSII) as described in footnote on page 4
#        extinction_dict['a_g'] = ebv * (2.742 + 0.35) * 0.718


    return extinction_dict
