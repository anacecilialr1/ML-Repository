from astroquery.gaia import Gaia
import pandas as pd
import numpy as np

def query(mode):
    """
    Retrieves a labeled sample of Gaia DR3 sources

    Parameters:
    -----------
    set : str
        Either 'training' or 'testing'. Determines the size of the sample
        - 'training': returns 15,000 elements per class (star, galaxy, quasar)
        - 'testing': returns 150,000 elements per class, offset by random_index for no overlap

    Returns:
    --------
    combined : pandas.DataFrame
        A merged dataframe of stars, galaxies, and quasars with relevant astrometric and photometric features,
        including a 'classification' label column (based on the maximum class probability) and additional features:
            - 'uwe' : unit weight error
            - 'relvarg': relative G-band flux uncertainty
            - 'sinb': sine of galactic latitude

    Note that only high-confidence sources are selected (class probability > 0.999).

    """
    ## Check if the imput parameter is a string
    if not isinstance(mode, str):
      raise TypeError("Expected string, got %s" % (type(set),))

    ## Retrieves the training sample of 15000 elements per class
    if mode == 'training':

        # Querry to retrieve the elements classified as galaxies
        query_galaxy = "SELECT TOP 15000 dr3.source_id, dr3.random_index, dr3.ra, dr3.dec, dr3.b, dr3.parallax, dr3.pm, dr3.phot_g_mean_mag, dr3.bp_g, dr3.g_rp,\
                        dr3.astrometric_chi2_al, dr3.astrometric_n_good_obs_al, dr3.phot_g_n_obs, dr3.phot_g_mean_flux_over_error,\
                        p.classprob_dsc_allosmod_quasar AS quasar, p.classprob_dsc_allosmod_galaxy AS galaxy,\
                        p.classprob_dsc_allosmod_star AS star\
                FROM gaiadr3.gaia_source as dr3\
                LEFT JOIN gaiadr3.astrophysical_parameters AS p USING (source_id)\
                WHERE dr3.parallax is not null\
                AND dr3.pm is not null\
                AND dr3.phot_g_mean_mag is not null\
                AND dr3.bp_g is not null\
                AND dr3.g_rp is not null\
                AND dr3.astrometric_chi2_al is not null\
                AND dr3.astrometric_n_good_obs_al is not null\
                AND dr3.phot_g_n_obs is not null\
                AND dr3.phot_g_mean_flux_over_error is not null\
                AND p.classprob_dsc_allosmod_galaxy > 0.999\
                AND dr3.phot_g_mean_mag > 14.5\
                AND 0.3 + 1.1*dr3.bp_g - 0.29*POWER((dr3.bp_g),2) < dr3.g_rp\
                ORDER BY dr3.random_index"

        # Querry to retrieve the elements classified as stars
        query_star = "SELECT TOP 15000 dr3.source_id, dr3.random_index, dr3.ra, dr3.dec, dr3.b, dr3.parallax, dr3.pm, dr3.phot_g_mean_mag, dr3.bp_g, dr3.g_rp,\
                        dr3.astrometric_chi2_al, dr3.astrometric_n_good_obs_al, dr3.phot_g_n_obs, dr3.phot_g_mean_flux_over_error,\
                        p.classprob_dsc_allosmod_quasar AS quasar, p.classprob_dsc_allosmod_galaxy AS galaxy,\
                        p.classprob_dsc_allosmod_star AS star\
                FROM gaiadr3.gaia_source as dr3\
                LEFT JOIN gaiadr3.astrophysical_parameters AS p USING (source_id)\
                WHERE dr3.parallax is not null\
                AND dr3.pm is not null\
                AND dr3.phot_g_mean_mag is not null\
                AND dr3.bp_g is not null\
                AND dr3.g_rp is not null\
                AND dr3.astrometric_chi2_al is not null\
                AND dr3.astrometric_n_good_obs_al is not null\
                AND dr3.phot_g_n_obs is not null\
                AND dr3.phot_g_mean_flux_over_error is not null\
                AND p.classprob_dsc_allosmod_star > 0.999\
                AND dr3.phot_g_mean_mag > 14.5\
                AND 0.3 + 1.1*dr3.bp_g - 0.29*POWER((dr3.bp_g),2) < dr3.g_rp\
                ORDER BY dr3.random_index"
        
        # Querry to retrieve the elements classified as quasars
        query_quasar =  "SELECT TOP 15000 dr3.source_id, dr3.random_index, dr3.ra, dr3.dec, dr3.b, dr3.parallax, dr3.pm, dr3.phot_g_mean_mag, dr3.bp_g, dr3.g_rp,\
                        dr3.astrometric_chi2_al, dr3.astrometric_n_good_obs_al, dr3.phot_g_n_obs, dr3.phot_g_mean_flux_over_error,\
                        p.classprob_dsc_allosmod_quasar AS quasar, p.classprob_dsc_allosmod_galaxy AS galaxy,\
                        p.classprob_dsc_allosmod_star AS star\
                FROM gaiadr3.gaia_source as dr3\
                LEFT JOIN gaiadr3.astrophysical_parameters AS p USING (source_id)\
                WHERE dr3.parallax is not null\
                AND dr3.pm is not null\
                AND dr3.phot_g_mean_mag is not null\
                AND dr3.bp_g is not null\
                AND dr3.g_rp is not null\
                AND dr3.astrometric_chi2_al is not null\
                AND dr3.astrometric_n_good_obs_al is not null\
                AND dr3.phot_g_n_obs is not null\
                AND dr3.phot_g_mean_flux_over_error is not null\
                AND p.classprob_dsc_allosmod_quasar > 0.999\
                AND dr3.phot_g_mean_mag > 14.5\
                AND 0.3 + 1.1*dr3.bp_g - 0.29*POWER((dr3.bp_g),2) < dr3.g_rp\
                ORDER BY dr3.random_index"
        print('Retrieving training dataset')

    ## Retrieves the testing sample of 15000 elements per class        
    elif mode == 'testing':
        
        # Querry to retrieve the elements classified as galaxies
        query_galaxy = "SELECT TOP 150000 dr3.source_id, dr3.random_index, dr3.ra, dr3.dec, dr3.b, dr3.parallax, dr3.pm, dr3.phot_g_mean_mag, dr3.bp_g, dr3.g_rp,\
                        dr3.astrometric_chi2_al, dr3.astrometric_n_good_obs_al, dr3.phot_g_n_obs, dr3.phot_g_mean_flux_over_error,\
                        p.classprob_dsc_allosmod_quasar AS quasar, p.classprob_dsc_allosmod_galaxy AS galaxy,\
                        p.classprob_dsc_allosmod_star AS star\
                FROM gaiadr3.gaia_source as dr3\
                LEFT JOIN gaiadr3.astrophysical_parameters AS p USING (source_id)\
                WHERE dr3.parallax is not null\
                AND dr3.random_index > 159213675\
                AND dr3.pm is not null\
                AND dr3.phot_g_mean_mag is not null\
                AND dr3.bp_g is not null\
                AND dr3.g_rp is not null\
                AND dr3.astrometric_chi2_al is not null\
                AND dr3.astrometric_n_good_obs_al is not null\
                AND dr3.phot_g_n_obs is not null\
                AND dr3.phot_g_mean_flux_over_error is not null\
                AND p.classprob_dsc_allosmod_galaxy > 0.999\
                AND dr3.phot_g_mean_mag > 14.5\
                AND 0.3 + 1.1*dr3.bp_g - 0.29*POWER((dr3.bp_g),2) < dr3.g_rp\
                ORDER BY dr3.random_index"

        # Querry to retrieve the elements classified as stars
        query_star = "SELECT TOP 150000 dr3.source_id, dr3.random_index, dr3.ra, dr3.dec, dr3.b, dr3.parallax, dr3.pm, dr3.phot_g_mean_mag, dr3.bp_g, dr3.g_rp,\
                        dr3.astrometric_chi2_al, dr3.astrometric_n_good_obs_al, dr3.phot_g_n_obs, dr3.phot_g_mean_flux_over_error,\
                        p.classprob_dsc_allosmod_quasar AS quasar, p.classprob_dsc_allosmod_galaxy AS galaxy,\
                        p.classprob_dsc_allosmod_star AS star\
                FROM gaiadr3.gaia_source as dr3\
                LEFT JOIN gaiadr3.astrophysical_parameters AS p USING (source_id)\
                WHERE dr3.parallax is not null\
                AND dr3.pm is not null\
                AND dr3.phot_g_mean_mag is not null\
                AND dr3.bp_g is not null\
                AND dr3.random_index > 20773\
                AND dr3.g_rp is not null\
                AND dr3.astrometric_chi2_al is not null\
                AND dr3.astrometric_n_good_obs_al is not null\
                AND dr3.phot_g_n_obs is not null\
                AND dr3.phot_g_mean_flux_over_error is not null\
                AND p.classprob_dsc_allosmod_star > 0.999\
                AND dr3.phot_g_mean_mag > 14.5\
                AND 0.3 + 1.1*dr3.bp_g - 0.29*POWER((dr3.bp_g),2) < dr3.g_rp\
                ORDER BY dr3.random_index"

        # Querry to retrieve the elements classified as quasars
        query_quasar =  "SELECT TOP 150000 dr3.source_id, dr3.random_index, dr3.ra, dr3.dec, dr3.b, dr3.parallax, dr3.pm, dr3.phot_g_mean_mag, dr3.bp_g, dr3.g_rp,\
                        dr3.astrometric_chi2_al, dr3.astrometric_n_good_obs_al, dr3.phot_g_n_obs, dr3.phot_g_mean_flux_over_error,\
                        p.classprob_dsc_allosmod_quasar AS quasar, p.classprob_dsc_allosmod_galaxy AS galaxy,\
                        p.classprob_dsc_allosmod_star AS star\
                FROM gaiadr3.gaia_source as dr3\
                LEFT JOIN gaiadr3.astrophysical_parameters AS p USING (source_id)\
                WHERE dr3.parallax is not null\
                AND dr3.random_index > 43678626\
                AND dr3.pm is not null\
                AND dr3.phot_g_mean_mag is not null\
                AND dr3.bp_g is not null\
                AND dr3.g_rp is not null\
                AND dr3.astrometric_chi2_al is not null\
                AND dr3.astrometric_n_good_obs_al is not null\
                AND dr3.phot_g_n_obs is not null\
                AND dr3.phot_g_mean_flux_over_error is not null\
                AND p.classprob_dsc_allosmod_quasar > 0.999\
                AND dr3.phot_g_mean_mag > 14.5\
                AND 0.3 + 1.1*dr3.bp_g - 0.29*POWER((dr3.bp_g),2) < dr3.g_rp\
                ORDER BY dr3.random_index"
        print('Retrieving training dataset')
    else: 
      raise ValueError("Select a querry for testing or for training")

    # Launches ADQL queries to the Gaia DR3 archive
    jobq = Gaia.launch_job_async(query_quasar)
    jobg = Gaia.launch_job_async(query_galaxy)
    jobs = Gaia.launch_job_async(query_star)

    gtable = jobg.get_results()
    stable = jobs.get_results()
    qtable = jobq.get_results()

    # Using pandas to compute new relevant columns
    datag = gtable.to_pandas()
    datas = stable.to_pandas()
    dataq = qtable.to_pandas()
    
    # Creates label column based on the maximum class probability
    datag['classification'] = datag[['quasar', 'galaxy', 'star']].idxmax(axis=1, skipna = True)
    datas['classification'] = datas[['quasar', 'galaxy', 'star']].idxmax(axis=1, skipna = True)
    dataq['classification'] = dataq[['quasar', 'galaxy', 'star']].idxmax(axis=1, skipna = True)

    combined = pd.concat([datag, datas, dataq], ignore_index=True)

    # Prints the value counts of each class
    print(combined['classification'].value_counts())

    # Computes the rest of the features
    combined['uwe'] = np.sqrt(combined['astrometric_chi2_al']/(combined['astrometric_n_good_obs_al']-5))
    combined['relvarg'] = np.sqrt(combined['phot_g_mean_flux_over_error'])/combined['phot_g_mean_mag']
    combined['sinb']  = np.sin(combined['b']* np.pi / 180)

    return combined

    
    

