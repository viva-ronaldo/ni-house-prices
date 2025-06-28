#Method:
#- Read all LPS 2005 price data; remove bad data, Mixed types, and some very large props
#- Get Q4 2004 to current quarter price changes (LPS values are probably from Q4 2004)
#  - Estimate the change from Q4 2004 to Q1 2005 as NIHPI doesn't go back past Q1 2005
#  - Inflate LPS prices to current quarter with NIHPI by LGD
#  - Adjust the factor by property type, apartment vs the rest
#  - ONS Northern Ireland prices gives 2005-2022Q3 changes as 1.57 - this seems to be the NIHPI
#    They use a regression model on actual sales prices to normalise them https://www.gov.uk/government/publications/about-the-uk-house-price-index/quality-and-methodology#methods-used-to-produce-the-uk-hpi 
#    e.g. Scotland values are 2.05 overall and 1.81 flats; Wales 1.77, 1.35
#    so it is agreed that flats have risen by less (mostly diff in last two years)
#  - Now updated to current quarter Q1 2024: nw factors are 1.66 houses and 1.35 flats
#- Create model target price per square metre, and features
#- Fit model for ppsqm with postcode, home type, yard, garage, garden, other outbuilding, and multiple area features;
#  a similar model by LSOA; then two slightly simpler ones to get the calculator coefficients for houses and flats separately
#- Summarise data by postcode or LSOA to make output files

import numpy as np
import pandas as pd
import geopandas as gpd
import gc, glob, re

from process_LPS_functions import (
    read_nihpi_api_nationwide,
    read_nihpi_api_lgd,
    load_and_filter_LPS_houses_data,
    convert_desc_to_home_type,
    calculate_glm_coefs,
    calculate_glm_coefs_by_LSOA,
    summarise_coefs,
    save_colour_bands_file,
    combine_coefs_for_calculator,
    save_nearest_five_postcode_files,
)

use_nihpi_api = False  # API is not updating as of Q3 2024
recent_price_quarter = 'Q4 2024'  # use latest quarter if use_nihpi_api = True, otherwise use this quarter in pre-downloaded data
refresh_nearest_five_postcodes = False  # make true if the LPS data ever change (new postcodes added)

min_properties_per_postcode = 10
smoothing_alpha = 3000
palette_all_props, palette_LSOAs = 'PuOr', 'PuOr'

Q42004_TO_Q12005_FACTOR = 1.02  # doesn't change

house_price_data_dir = '/media/shared_storage/data/ni-house-prices_data'
nihpi_nw_file = f"{house_price_data_dir}/other_data/ni-hpi-by-property-type-q1-2005---{recent_price_quarter.replace(' ','-').lower()}.csv"
nihpi_lgd_file = f"{house_price_data_dir}/other_data/standardised-price-and-index-by-lgd-q1-2005---{recent_price_quarter.replace(' ','-').lower()}.csv"

recent_price_quarter_str = f'{recent_price_quarter[-4:]}_{recent_price_quarter[:2]}'  # e.g. 2024_Q4
ppsqm_postcode_glm_coefs_filename = f'ppsqm_nge10_glmmethod_smoothed_alpha{smoothing_alpha}_{recent_price_quarter_str}.csv'
ppsqm_lsoa_glm_coefs_filename = f'ppsqm_nge10_LSOAglmmethod_{recent_price_quarter_str}.json'
postcode_glm_colour_bands_filename = f'colour_bands_ppsqm_nge10_glmmethod_{recent_price_quarter_str}.csv'
lsoa_glm_colour_bands_filename = f'colour_bands_ppsqm_nge10_LSOAglmmethod_{recent_price_quarter_str}.csv'
calculator_coefs_houses_filename = f'calculator_coefs_houses_smoothed_alpha{smoothing_alpha}_{recent_price_quarter_str}.csv'
calculator_coefs_flats_filename = f'calculator_coefs_flats_smoothed_alpha{smoothing_alpha}_{recent_price_quarter_str}.csv'
processed_lps_data_path = f'{house_price_data_dir}/other_data/processed_LPS_valuations_{recent_price_quarter_str}.csv'

# Read in data

# HPI scaling factors
if use_nihpi_api:
    # ODNI API works, though it was not obvious that it did until I made an account
    # Nationwide by Apartment, House
    price_changes_nw = read_nihpi_api_nationwide()
    # By LGD
    price_changes_lgd = read_nihpi_api_lgd()
    assert price_changes_nw.index[-1] == price_changes_lgd.index[-1], 'NW and LGD APIs are not aligned'

    print(f'Most recent price quarter from NIHPI APIs (nw and lgd) is {price_changes_nw.index[-1]}')
    recent_price_quarter = price_changes_nw.index[-1]
else:
    price_changes_nw = pd.read_csv(nihpi_nw_file).set_index('Quarter_Year')
    price_changes_lgd = pd.read_csv(nihpi_lgd_file).set_index('Quarter_Year')
    price_changes_lgd = price_changes_lgd[[c for c in price_changes_lgd.columns if '_HPI' in c]]

# LGD geometries
lgd_geom = gpd.read_file(f'{house_price_data_dir}/other_data/LAD_MAY_2022_UK_BSC_V3/').to_crs("EPSG:4326")   #convert from something to lat lon geometry

# LSOA geometries
soas = (
    gpd.read_file(f'{house_price_data_dir}/other_data/soa2001_ODNI.json')
    .set_crs('EPSG:29902', allow_override=True)
    .to_crs('EPSG:4326')
)
soas['SOA_LABEL'] = soas.SOA_LABEL.apply(lambda l: l.replace('_',' '))

# --- Prepare houses data ---

lps_files = glob.glob(f'{house_price_data_dir}/LPS_data/lps_valuations_*_*.csv')
postcodes_file = f'{house_price_data_dir}/other_data/full_postcode_latlons_from_doogal.csv'
houses = load_and_filter_LPS_houses_data(lps_files, postcodes_file)

#Get price change factors from 2005 to current

print(f'\nScaling prices from Q4 2004 to {recent_price_quarter}')

# Nationwide
hpi_apts_q12005, hpi_allprops_q12005 = price_changes_nw.loc['Q1 2005', ['NI_Apartment_Price_Index', 'NI_Residential_Property_Price_Index']]
hpi_apts_latest, hpi_allprops_latest = price_changes_nw.loc[recent_price_quarter, ['NI_Apartment_Price_Index', 'NI_Residential_Property_Price_Index']]

price_change_2005_to_current_nw_apts = hpi_apts_latest / (hpi_apts_q12005 / Q42004_TO_Q12005_FACTOR)
price_change_2005_to_current_nw_allprops = hpi_allprops_latest / (hpi_allprops_q12005 / Q42004_TO_Q12005_FACTOR)
# Apartments make up 11% of the market (from multiple quarterly reports) so get the houses component as 8/9 of the total
price_change_2005_to_current_nw_houses = (9 * price_change_2005_to_current_nw_allprops - price_change_2005_to_current_nw_apts) / 8
print(f'NIHPI factors countrywide for house and apartment are {price_change_2005_to_current_nw_houses:.2f} and {price_change_2005_to_current_nw_apts:.2f}')
# Set these factors in df but they will only be used for gap filling the LGD factors
houses['value_current_by_nw'] = np.where(
    houses.home_type == 'House',
    houses.value * price_change_2005_to_current_nw_houses,
    houses.value * price_change_2005_to_current_nw_apts)
houses['price_per_sq_m_current_by_nw'] = houses.value_current_by_nw / houses.area_m2

# By LGD
price_changes_2005_to_current_lgd = price_changes_lgd.loc[recent_price_quarter] / (price_changes_lgd.loc['Q1 2005'] / Q42004_TO_Q12005_FACTOR)
price_changes_2005_to_current_lgd.index = [
    c[:-4]
    .replace('_', ' ')
    .replace('Armagh City', 'Armagh City,')
    .replace('Newry', 'Newry,')
    for c in price_changes_2005_to_current_lgd.index
]
price_changes_2005_to_current_lgd.index.name = 'LGD'
print('NIHPI factors by LGD for all property types are:\n', price_changes_2005_to_current_lgd)

# Get the current value calculated by LGD and property type and join on
ghouses = (
    gpd.GeoDataFrame(houses, geometry=gpd.points_from_xy(houses.Longitude, houses.Latitude))
    .set_crs("EPSG:4326")
    [['address', 'postcode', 'value', 'home_type', 'geometry', 'Longitude', 'Latitude']]
)

houses = houses.join(
    (ghouses
        .assign(id_tmp = lambda df: df.index)
        .sjoin(lgd_geom[['LAD22NM', 'geometry']], how='inner')
        .merge(price_changes_2005_to_current_lgd.to_frame(name='value_factor_by_lgd'),
            left_on='LAD22NM', right_on='LGD', how='left')
        .rename(columns={'LAD22NM': 'lgd'})
        .set_index('id_tmp').rename_axis('id')
        .assign(value_current_by_lgd = lambda df: np.where(
            df.home_type == 'House',
            df.value * df.value_factor_by_lgd * (price_change_2005_to_current_nw_houses / price_change_2005_to_current_nw_allprops),
            df.value * df.value_factor_by_lgd * (price_change_2005_to_current_nw_apts / price_change_2005_to_current_nw_allprops))
            # apply the overall scaling by LGD, then move houses up a bit and apts down a bit using the nationwide ratios
        )
        [['lgd', 'value_current_by_lgd']]
    ),
    how='left'
)
houses['price_per_sq_m_current_by_lgd'] = houses.value_current_by_lgd / houses.area_m2

print(f'Filling in {houses.price_per_sq_m_current_by_lgd.isnull().mean()*100:.1f}% current values with nw due to missing LGD value')
houses['value_current_by_lgd'].fillna(houses.value_current_by_nw, inplace=True)
houses['price_per_sq_m_current_by_lgd'].fillna(houses.price_per_sq_m_current_by_nw, inplace=True)

print('Finally, prices from 2005 to now have been scaled by:\n\n', houses.groupby('home_type').agg(
    mean_price_2005 = ('price_per_sq_m', np.mean), 
    mean_price_current = ('price_per_sq_m_current_by_lgd', np.mean)))

del lgd_geom, price_changes_2005_to_current_lgd

#Define terms for models
houses['home_type_House'] = np.where(houses.home_type == 'House', 1, 0)
houses['home_type_Flat'] = np.where(houses.home_type == 'Flat', 1, 0)
#
houses['has_garden'] = houses.desc.str.contains('garden')
houses['has_yard'] = houses.desc.str.contains('yard')
#Most of the time has_garage is Yes when outbuilding is mentioned, but not always; these could be other buildings
houses['has_other_outbuilding'] = houses.apply(lambda row: 'Yes' if (re.search('outbuild', row['desc']) \
    and row['has_garage']=='No') else 'No', axis=1)
#
houses['has_garage_01'] = np.where(houses.has_garage == 'Yes', 1, 0)
houses['has_other_outbuilding_01'] = np.where(houses.has_other_outbuilding == 'Yes', 1, 0)
houses['has_garden_01'] = np.where(houses.has_garden, 1, 0)
houses['has_yard_01'] = np.where(houses.has_yard, 1, 0)
#
houses['area_squared'] = houses.area_m2**2
houses['area_lt_60_01'] = np.where(houses.area_m2 < 60, 1, 0)
houses['area_lt_100_01'] = np.where(houses.area_m2 < 100, 1, 0)

#Trim df a bit
houses = houses.drop(columns=['address', 'lgd', 'desc', 'type',
    'price_per_sq_m',
    'has_garage', 'has_other_outbuilding', 'has_garden', 'has_yard',
    'price_per_sq_m_current_by_nw', 'value_current_by_nw'])

#Save out for analysis, without geometry
houses.drop(columns=['geometry']).to_csv(processed_lps_data_path, index=False)

# ---- Fit models ----

#All data, postcodes
print('\nFitting models for all properties')

all_home_types_formula = 'price_per_sq_m_current_by_lgd ~ 0 + ' + \
    'has_garage_01 + has_garden_01 + has_yard_01 + has_other_outbuilding_01 + ' + \
    'home_type + home_type:area_m2 + home_type:area_squared + home_type:area_lt_60_01'

pc_coefs_normd, _ = calculate_glm_coefs(houses, all_home_types_formula, min_properties_per_postcode=min_properties_per_postcode, normalise_pc_coefs=True)

gc.collect()

# Can get SE or conf_int on each postcode coefficient (average of the 3 fits)
#  but how to combine other term SEs in calculator model?
# Or get RMSE training error from each fit as (((this_glm.resid_response)**2).mean())**0.5)
#  and take median, or combine all residuals. Could inflate a bit for test error.
#  Values of ~150 ppsqm are typical, i.e. about 10%
# Or could store SEs for all coefs and in the calculator, sample each coef in a Monte Carlo method to get prediction range.

# Save coefficients by postcode for map

summ_coefs = summarise_coefs(houses, pc_coefs_normd, soas, cmap=palette_all_props, alpha=smoothing_alpha)
summ_coefs.to_csv(f'static/{ppsqm_postcode_glm_coefs_filename}', index=False)
save_colour_bands_file(summ_coefs, f'static/{postcode_glm_colour_bands_filename}')  # colour bands for plot

print('Saved map coefficients for all properties')

del pc_coefs_normd, summ_coefs
gc.collect()

# The same for LSOA
print('\nFitting GLM by LSOA')

lsoa_coefs = calculate_glm_coefs_by_LSOA(houses, all_home_types_formula)

gc.collect()

summ_soas_coefs = summarise_coefs(houses, lsoa_coefs, soas, region_type='LSOA', cmap=palette_LSOAs, alpha=None)
summ_soas_coefs = gpd.GeoDataFrame(summ_soas_coefs)[
    ['LSOA Code', 'n_properties', 'ppsqm_delta', 'html_colour', 'popup_text', 'geometry']
]
#simplify shapes to give a smaller file
summ_soas_coefs['geometry'] = summ_soas_coefs.geometry.simplify(0.0002)
summ_soas_coefs.to_file(f'static/{ppsqm_lsoa_glm_coefs_filename}', driver='GeoJSON')
save_colour_bands_file(summ_soas_coefs, f'static/{lsoa_glm_colour_bands_filename}', unit_variable='LSOA Code')

print('Saved map coefficients for all properties by LSOA')

del lsoa_coefs, summ_soas_coefs
gc.collect()

# There is an uncertainty on all the postcode and LSOA coefs of ~50-100,
#   i.e. we get a different result each time depending on the batch shuffling.
#   Some of this will be handled by the spatial smoothing, and the rest is
#   acceptable given that the range of coef values is ~-1000 to +2000, so
#   most postcodes' rankings will be fairly stable even with +/-100 on coef value.


# ---- Calculator coefficients: fit slightly different models ----

print('Fitting calculator GLM on houses')

calculator_formula_houses = 'price_per_sq_m_current_by_lgd ~ 0 + ' + \
    'has_garage_01 + has_garden_01 + ' + \
    ' + area_m2 + area_squared + area_lt_60_01'

pc_coefs_calculator_houses, non_pc_coefs_calculator_houses = calculate_glm_coefs(
    houses[houses.home_type == 'House'],
    calculator_formula_houses,
    min_properties_per_postcode=min_properties_per_postcode,
    normalise_pc_coefs=False
)

comb_coefs_calculator_houses = combine_coefs_for_calculator(
    pc_coefs_calculator_houses,
    non_pc_coefs_calculator_houses,
    houses[houses.home_type=='House'],
    soas,
    alpha=smoothing_alpha)
comb_coefs_calculator_houses.round(5).to_csv(f'static/{calculator_coefs_houses_filename}', index=False)

print('Saved calculator coefficients for houses')

print('Fitting calculator GLM on flats')

calculator_formula_flats = 'price_per_sq_m_current_by_lgd ~ 0 + ' + \
    ' + area_m2 + area_squared + area_lt_60_01'

pc_coefs_calculator_flats, non_pc_coefs_calculator_flats = calculate_glm_coefs(
    houses[houses.home_type == 'Flat'],
    calculator_formula_flats,
    min_properties_per_postcode=min_properties_per_postcode,
    normalise_pc_coefs=False
)

comb_coefs_calculator_flats = combine_coefs_for_calculator(
    pc_coefs_calculator_flats,
    non_pc_coefs_calculator_flats,
    houses[houses.home_type=='Flat'],
    soas,
    alpha=smoothing_alpha)
comb_coefs_calculator_flats.round(5).to_csv(f'static/{calculator_coefs_flats_filename}', index=False)

print('Saved calculator coefficients for flats')

# These have the standard error which is really related to number of training observations
#   and not the variation within a postcode, but it gives a reasonable way
#   to generate a range for each postcode.
# For short postcodes, fitting a model and getting SE would not work, because
#   the SEs could be small, meaning a short postcode *average* effect is well fitted
#   but doesn't say anything about variation within the short postcode,
#   which can be large. So keep the old method for this, of combining the 
#   short postcodes and taking quantiles.
# Combine these two methods to assign generic value_lower and value_upper to all postcode terms.

# Nearest short postcodes, for the calculator output
# Don't need to rerun this unless input LPS data changes (new postcodes)
if refresh_nearest_five_postcodes:
    print('Getting nearest five postcodes')

    save_nearest_five_postcode_files(
        ghouses,
        comb_coefs_calculator_houses,
        comb_coefs_calculator_flats,
        'static/postcodeshort_nearest_five.csv',
        'static/postcodelongtoshort_nearest_five.csv')

    print('Saved postcode nearest fives long and short')

print('Done')
