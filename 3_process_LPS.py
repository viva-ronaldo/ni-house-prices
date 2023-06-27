#Method:
#- Read all LPS 2005 price data; remove bad data, Mixed types, and some very large props
#- Get Q1 2005 to Q2 2022 price changes (LPS values are from 1 Jan 2005)
#  - First countrywide from NI House Price Index (ODNI); this gives 1.51 for 'residential'
#    (could also use 1.26 for apartments, but can't use 1.62 detached, 1.56 semi-d, 1.43 terrace)
#  - Second use by LGD, also from NIHPI (ODNI); this gives factors 1.41-1.60 for 11 areas
#  - LGD scalings are used for 99% of cases; rest use nw
#  - Now updated to Q3 2022: nw factors 1.57 overall and 1.28 flats
#  - Nationwide HPI gives 2005-2022Q3 change as 1.67 (1.60 for Q4), so maybe should shift NIHPI up a bit (1.67/1.57 = 1.06)
#  - ONS Northern Ireland prices gives 2005-2022Q3 changes as 1.57 - this seems to be the NIHPI
#    They use a regression model on actual sales prices to normalise them https://www.gov.uk/government/publications/about-the-uk-house-price-index/quality-and-methodology#methods-used-to-produce-the-uk-hpi 
#    e.g. Scotland values are 2.05 overall and 1.81 flats; Wales 1.77, 1.35
#    so it is agreed that flats have risen by less (mostly diff in last two years)
#- Fit model for ppsqm_2022 with postcode, home type, yard, garage, garden, other outbuilding features
#    (would like to have also house type breakdown and num bedrooms in particular); and another model by LSOA
#- Lots of summarising data by postcode or LSOA to make output files

#TODO add mean ppsqm for flat and house+garage+garden to website
#TODO check postcode dropdown in calculator
#TODO check worker true error in papa.parse for table

import numpy as np
import pandas as pd
import re, glob, copy, random, time, gc
import geopandas as gpd
from matplotlib import colors, cm
from statsmodels.api import GLM, add_constant
import psutil

min_properties_per_postcode = 10
min_property_area, max_property_area = 2, 800
smoothing_alpha = 3000

recent_price_quarter = 'Q3 2022'
nihpi_nw_file = f"other_data/ni-hpi-by-property-type-q1-2005---{recent_price_quarter.replace(' ','-').lower()}.csv"
nihpi_lgd_file = f"other_data/standardised-price-and-index-by-lgd-q1-2005---{recent_price_quarter.replace(' ','-').lower()}.csv"

test_files = glob.glob('LPS_data/lps_valuations_*_*.csv')

houses = pd.concat([pd.read_csv(f) for f in test_files])
print(f'Read {houses.shape[0]} rows')

houses = houses[houses.postcode.notnull()]
print(f'{houses.shape[0]} after dropping missing postcodes')

#
#houses = houses[houses.postcode.isin(houses.postcode.sample(500).values)].copy()
#houses = houses[houses.postcode.str.contains('BT5 ')].copy()
#print(f'TMP filtered to {len(houses)} rows')
#

houses['desc'] = houses.desc.fillna('')
houses = houses[~houses.desc.str.contains('presbytery|convent|sheltered|shletered|mobile home|shop|premises|self-catering unit|'+
                                          'public house|showhouse|wardens house|community|unfinished|vacant|freestanding caravan|'+
                                          'gate lodge|gatelodge|church|wardens|vacany|room|parochial house|'+
                                          'mobile|single level self contained|caravan|carvan')].copy()
houses = houses[~houses.desc.isin(['','outbuilding','hut','houses (huts) garden','hut garden',
                                   'storage','manse','cabin','garage','store','bed & breakfast',
                                   'prefab','log cabin'])]
print(f'{houses.shape[0]} after dropping empty or non-standard descs')

#Add lon, lat from Doogal postcodes 
postcodes = pd.read_csv('other_data/full_postcode_latlons_from_doogal.csv', 
    usecols=['Postcode','Latitude','Longitude','LSOA Code'])
houses = houses.merge(postcodes, left_on='postcode', right_on='Postcode')

#Some postcode lon lat are 0
print(f"Dropping {(houses.Longitude==0).sum()} lon=lat=0 cases of {houses.shape[0]} total")
houses = houses[houses.Longitude != 0]

print(f"Dropping {(houses.type=='Mixed').sum()} type=Mixed cases")
houses = houses[houses.type=='Domestic']

#19000 is used in some suspicious cases; may be in bad disrepair or some other situation
print(f'Dropping {(houses.value==19000).sum()} cases of £19,000')
houses = houses[houses.value!=19000]

#Drop very big ones which would stretch a linear ppsqm fit or be not normal houses
print(f'Dropping {(houses.area_m2>max_property_area).sum()} cases bigger than {max_property_area}m^2')
houses = houses[houses.area_m2<=max_property_area]

print(f'Dropping {len(houses[houses.area_m2 <= min_property_area])} records with < {min_property_area} m^2 area')
houses = houses[houses.area_m2 > min_property_area]

houses = houses.set_index('id')

#Some look like typos for area
area_corrections = [
    (201786, 41),
    (204122, 61),
    (204152, 61),
    (204122, 53),
    (204152, 53),
    (260850, 110), #map shows 110 but prop pages show 11
    (260849, 110),
    (260848, 110),
    (260846, 110),
    (291239, 93), #not 963
    (251067, 81), #not 810
    (242378, 106), #not 1006
    (188692, 78.9), #probably, not 789
    (181623, 45.8), #not 458
    (246832, 86), #ish, not 23
    (144572, 118.15), #this is the value in the map view
    (153232, 94), #probably two 47s
    (153232, 94),
    (153177, 97), #probably two 48.5s
    (159505, 96), #probably two 48s
    (2000077, 47), #probably, not 17 (over 2 floors) as more expensive than a 43m2
    (99782, 99), #not 19
    (109693, 98), #probably two 49s
    (89430, 92), #probably two 46s
    (89043, 68), #probably two 34s
    (88844, 94) #probably two 47s
]
for t in area_corrections:
    #check, in case running on a subset, as loc adds a row otherwise
    if t[0] in houses.index:
        houses.loc[t[0], 'area_m2'] = t[1]

value_corrections = [
    (243125, 1_000_000), #change 64 from 100k; 62 nearby is 1m
    (207581, 190_000), #probably, not 19,000
    (194274, 95_000) #probably, not 19,500
]
for t in value_corrections:
    if t[0] in houses.index:
        houses.loc[t[0], 'value'] = t[1]


houses['price_per_sq_m'] = houses.value/houses.area_m2

print(f'''Dropping {len(houses[(houses.value <= 10000) & (houses.price_per_sq_m < 100)])} 
    cases with value a few thousand and ppsqm < £100/m^2''')
houses = houses[~((houses.value <= 10000) & (houses.price_per_sq_m < 100))].copy()

print(f'  and {houses[houses.price_per_sq_m <= 200].shape[0]} rows with ppsqm < £200/m^2')
houses = houses[houses.price_per_sq_m > 200].copy()

#A few more look dodgy in size and/or value, or may be a non-house
bad_ids = [
    217637,207728,227086,249727,279559,241819,283490,283752,
    244679,166273,290979,164369,177058,145552,147965,195652,
    124847,138233,116652,145536, 84213,113324, 89525, 66628,
     64802, 85170,200077, 63280
    ]
print(f'Dropping {len(bad_ids)} ids found to be dodgy')
houses = houses[~houses.index.isin([bad_ids])].copy()

#Assign home type
def convert_desc_to_home_type(desc_string):
    if 'house' in desc_string:
        return 'House'
    elif desc_string[:2] in ['h ','h(','hg','ho','h+','h&','h/','h.','h0','hy','h<','h,','hs','h[','h{']:
        return 'House'
    elif desc_string in ['bungalow','cottage','h','ouse outbuilding garden','shouse outbuildings garden',
                         'hag','hag o',"h's o g",'hp & g','1st & 2nd floor house','use outbuilding garden']:
        return 'House'
    elif re.search('maisonette|farmhouse|detached rural dwelling|chalet|'+
        'cottage|townhouse|bungalow|converted barn|converted outbuilding|'+
        'huose|hiuse|huse|nouse|hiouse|hpuse', desc_string):
        return 'House'
    elif re.search('flat|flt|falt|flaat|fla t|glat|flct|flst|bedsit|penthouse|'+
                   'apartment|appartment|apart|apt|aparment|app|a\[artment|'+
                   'aparatment|apar|aoartment|aartment|aprtment|apatrtment|apratment|'+
                   'flast |flzt |flar |room |rm |rooms |fl 1st|flay |fla\(1st\)|fla gd|'+
                   'fat|flag|first \(first floor\)|lat \(1st floor\)|f;at \(1st floor\)', desc_string):
        return 'Flat'
    elif desc_string in ['<1st>','app','1lat (1st)','1f app','app(gf)']:
        return 'Flat'
    else:
        return 'Other'

houses['home_type'] = houses.desc.apply(convert_desc_to_home_type)

#Drop the few Others
print(f"Dropping {(houses.home_type=='Other').sum()} cases not identified as House or Flat")
houses = houses[houses.home_type.isin(['House','Flat'])].copy()


#Get price change factors from 2005 to 2022

#ODNI API works though it is not obvious until I make an account
# import requests
# res = requests.get('https://admin.opendatani.gov.uk/api/3/action/datastore_search?resource_id=dc7af407-bcb5-4820-81c0-a5e0dd7cbcb9')
# nihpi_df = pd.DataFrame(res.json()['result']['records'])
# nihpi_types_dict = {x['id']: x['type'] for x in res.json()['result']['fields']}
# for c in nihpi_df.columns:
#     if nihpi_types_dict[c] == 'numeric':
#         nihpi_df[c] = nihpi_df[c].astype(float)
# nihpi_df = nihpi_df[['Quarter_Year'] + sorted([c for c in nihpi_df.columns if c != 'Quarter_Year'])]

print(f'\nScaling prices from Q1 2005 to {recent_price_quarter}')
#Nationwide
price_change_nw = pd.read_csv(nihpi_nw_file).set_index('Quarter_Year')
price_change_2005_2022_nw_apts = price_change_nw.loc[recent_price_quarter]['NI_Apartment_Price_Index'] / \
    price_change_nw.loc['Q1 2005']['NI_Apartment_Price_Index']
price_change_2005_2022_nw_all = price_change_nw.loc[recent_price_quarter]['NI_Residential_Property_Price_Index'] / \
    price_change_nw.loc['Q1 2005']['NI_Residential_Property_Price_Index']
#Assume that detached, semi-detached, terrace, and apartment each make up a quarter of the 'all' value; 
#  adjust it to remove the apartment bit
price_change_2005_2022_nw_houses = (4*price_change_2005_2022_nw_all - price_change_2005_2022_nw_apts)/3
print(f'Countrywide NIHPI factors for house and apartment are {price_change_2005_2022_nw_houses:.2f} and {price_change_2005_2022_nw_apts:.2f}')
houses['value_2022_by_nw'] = houses.apply(lambda row: 
    row.value*price_change_2005_2022_nw_houses if row.home_type=='House' else row.value*price_change_2005_2022_nw_apts, axis=1)
houses['price_per_sq_m_2022_by_nw'] = houses.value_2022_by_nw/houses.area_m2

#By LGD
#TODO change to use different term for flats
price_change_lgd = pd.read_csv(nihpi_lgd_file).set_index('Quarter_Year')
price_changes_2005_2022_lgd = price_change_lgd.loc[recent_price_quarter] / price_change_lgd.loc['Q1 2005']
price_changes_2005_2022_lgd = price_changes_2005_2022_lgd.loc[[c for c in price_changes_2005_2022_lgd.index if '_HPI' in c]]
price_changes_2005_2022_lgd.index = [c[:-4].replace('_',' ').replace('Armagh City','Armagh City,')\
    .replace('Newry','Newry,') for c in price_changes_2005_2022_lgd.index]
price_changes_2005_2022_lgd.index.name = 'LGD'
print('NIHPI factors by LGD are:\n', price_changes_2005_2022_lgd)
lgd_geom = gpd.read_file('other_data/LAD_MAY_2022_UK_BSC_V3/').to_crs("EPSG:4326")   #convert from something to lat lon geometry
ghouses = gpd.GeoDataFrame(houses, geometry=gpd.points_from_xy(houses.Longitude, houses.Latitude))\
    .set_crs("EPSG:4326")\
    [['address','postcode','value','home_type','geometry','Longitude','Latitude']]
#Join on the 2022 value calculated via LGD
houses = houses.join(
    (ghouses
        .assign(id_tmp = lambda df: df.index)
        .sjoin(lgd_geom[['LAD22NM','geometry']], how='inner')
        .merge(price_changes_2005_2022_lgd.to_frame(name='value_factor_by_lgd'), 
            left_on='LAD22NM', right_on='LGD', how='left')
        .rename(columns={'LAD22NM': 'lgd'})
        .set_index('id_tmp').rename_axis('id')
        .assign(value_2022_by_lgd = lambda df: df.value * df.value_factor_by_lgd)
        .assign(value_2022_by_lgd = lambda df: df.apply(lambda row: row.value_2022_by_lgd * (price_change_2005_2022_nw_houses/price_change_2005_2022_nw_all) \
            if row.home_type=='House' else \
            row.value_2022_by_lgd * (price_change_2005_2022_nw_apts/price_change_2005_2022_nw_all), axis=1))  #move houses up a bit and apts down a bit
        [['lgd','value_2022_by_lgd']]
    ),
    how='left'
)
houses['price_per_sq_m_2022_by_lgd'] = houses.value_2022_by_lgd/houses.area_m2

print(f'Filling in {houses.price_per_sq_m_2022_by_lgd.isnull().mean()*100:.1f}% 2022 values with nw due to missing LGD value')
houses['value_2022_by_lgd'].fillna(houses.value_2022_by_nw, inplace=True)
houses['price_per_sq_m_2022_by_lgd'].fillna(houses.price_per_sq_m_2022_by_nw, inplace=True)

print('Finally prices from 2005 to 2022 have been scaled by:\n', houses.groupby('home_type').agg(
    mean_price_2005 = ('price_per_sq_m', np.mean), 
    mean_price_2022 = ('price_per_sq_m_2022_by_lgd', np.mean)))

del lgd_geom, price_changes_2005_2022_lgd

#Define terms for models
houses['has_garden'] = houses.desc.str.contains('garden')
houses['has_yard'] = houses.desc.str.contains('yard')
#Most of the time has_garage is Yes when outbuilding is mentioned, but not always; these could be other buildings
houses['has_other_outbuilding'] = houses.apply(lambda row: 'Yes' if (re.search('outbuild', row['desc']) \
    and row['has_garage']=='No') else 'No', axis=1)
#
houses['has_garage_01'] = houses.has_garage.apply(lambda g: 1 if g=='Yes' else 0)
houses['has_other_outbuilding_01'] = houses.has_other_outbuilding.apply(lambda g: 1 if g=='Yes' else 0)
houses['has_garden_01'] = houses.has_garden.apply(lambda g: 1 if g else 0)
houses['has_yard_01'] = houses.has_yard.apply(lambda g: 1 if g else 0)

#Trim df a bit
houses = houses.drop(columns=['address','postcode','lgd','desc','type',
    'has_garage','has_other_outbuilding','has_garden','has_yard',
    'value_2022_by_nw','price_per_sq_m_2022_by_nw',
    'price_per_sq_m'])

#Save out for analysis, without geometry
houses.drop(columns=['geometry']).to_csv('other_data/processed_LPS_valuations.csv', index=False)


#Fit models
def calculate_glm_coefs(df, include_home_type_term=True, min_properties_per_postcode=min_properties_per_postcode):
    #houses_mod = df.copy()
    if include_home_type_term:
        houses_mod = pd.concat([df, pd.get_dummies(df['home_type'], prefix='home_type')], axis=1)
    else:
        houses_mod = df

    #This gets too big to do in one go
    #houses_mod = pd.concat([houses_mod, pd.get_dummies(houses_mod['Postcode'], prefix='postcode')], axis=1)
    #postcode_cols = [c for c in houses_mod.columns if 'postcode_' in c]
    postcode_has_min_rows = (houses_mod.Postcode.value_counts() >= min_properties_per_postcode)
    postcode_cols = ['postcode_'+c for c in houses_mod.Postcode.unique() if c in postcode_has_min_rows.index[np.where(postcode_has_min_rows)]]
    print(f'Using {len(postcode_cols)} postcode terms with >= {min_properties_per_postcode} rows')

    #Fit three times and average the coefficients
    step = 300
    glm_res = []
    for e in [1,2,3]:
        #Need to jumble up the postcodes to avoid fitting only one area at once
        jumbled_postcode_cols = copy.copy(postcode_cols)
        random.shuffle(jumbled_postcode_cols)

        postcode_glms = []
        for i in range(0,len(postcode_cols),step):
            postcode_cols_batch = jumbled_postcode_cols[i:min(i+step,len(postcode_cols))]
            postcodes_batch = [p.split('_')[1] for p in postcode_cols_batch]
            
            #train_batch = houses_mod[houses_mod.Postcode.isin(postcodes_batch)]
            #Can leave out the unused postcode cols
            #train_batch = train_batch[[c for c in houses_mod.columns if 'postcode_' not in c or c in postcode_cols_batch]]
            
            tmp_houses_mod = houses_mod[houses_mod.Postcode.isin(postcodes_batch)]
            train_batch = pd.concat([tmp_houses_mod, pd.get_dummies(tmp_houses_mod['Postcode'], prefix='postcode')], axis=1)
            del tmp_houses_mod
            
            try:
                features = ['has_garage_01','has_garden_01','has_yard_01','has_other_outbuilding_01']
                if include_home_type_term:
                    features += ['home_type_House','home_type_Flat'] 
                features += postcode_cols_batch
                this_glm = GLM(train_batch['price_per_sq_m_2022_by_lgd'], 
                    train_batch[features]).fit()
                #No need for constant, as everything is House or Flat so base price can go into those
                #Don't know why but I don't need to leave out one factor level here
                postcode_glms.append(this_glm)
            except np.linalg.LinAlgError:
                print('This batch GLM did not converge; skipping')
            print(i, len(postcode_glms), np.round(psutil.Process().memory_info().rss / (1024 * 1024),0))
            if psutil.Process().memory_info().rss / (1024 * 1024) > 4000:
                gc.collect()

        glm_res.append(postcode_glms)

    del houses_mod

    pc_coefs = pd.concat([g.params for g in glm_res[0]]).filter(like='postcode').to_frame(name='coef_1')\
        .join(pd.concat([g.params for g in glm_res[1]]).filter(like='postcode').to_frame(name='coef_2'))\
        .join(pd.concat([g.params for g in glm_res[2]]).filter(like='postcode').to_frame(name='coef_3'))
    pc_coefs['pc_coef_avg'] = (pc_coefs.coef_1 + pc_coefs.coef_2 + pc_coefs.coef_3)/3

    pc_coefs['postcode'] = [s.split('_')[-1] for s in pc_coefs.index]
    pc_coefs = pc_coefs.reset_index()[['postcode','pc_coef_avg']]

    #Save non-postcode coefs for calculator
    #ravel doesn't work if sublists are unequal length, but concatenate does work here
    non_pc_coefs = (pd.concat([g.params for g in np.concatenate(glm_res)])
        .filter(regex='const|area|garage|garden|yard|outbuilding|home_type')
        .to_frame(name='value')
        .assign(coef = lambda df: df.index).reset_index(drop=True)
        .groupby('coef', as_index=False)
        .agg(value_mean = ('value', lambda v: np.round(np.mean(v),4)))
    )

    return pc_coefs, non_pc_coefs

#All data
print('\nFitting models for all properties')
pc_coefs, non_pc_coefs = calculate_glm_coefs(houses)
#We need one of these with pc coefs normalised to zero, for the map, and one non-normalised for the calculator
pc_coefs_normd = pc_coefs.copy()
pc_coefs_normd['pc_coef_avg'] -= pc_coefs_normd.pc_coef_avg.mean()
#Houses only
print('\nFitting models for houses')
pc_coefs_houses_normd, _ = calculate_glm_coefs(houses[houses.home_type=='House'], include_home_type_term=False)
pc_coefs_houses_normd['pc_coef_avg'] -= pc_coefs_houses_normd.pc_coef_avg.mean()
#Flats only
print('\nFitting models for flats')
pc_coefs_flats_normd, _ = calculate_glm_coefs(houses[houses.home_type=='Flat'], include_home_type_term=False)
pc_coefs_flats_normd['pc_coef_avg'] -= pc_coefs_flats_normd.pc_coef_avg.mean()


def calculate_glm_coefs_by_LSOA(df):
    #houses_mod = df.copy()
    houses_mod = pd.concat([df, pd.get_dummies(df['home_type'], prefix='home_type')], axis=1)

    lsoa_has_50_rows = (houses_mod['LSOA Code'].value_counts() >= 50)
    lsoa_cols = ['lsoa_'+c for c in houses_mod['LSOA Code'].unique() if c in lsoa_has_50_rows.index[np.where(lsoa_has_50_rows)]]
    print(f'Using {len(lsoa_cols)} LSOA terms with >= 50 rows')

    step = 150
    jumbled_lsoa_cols = copy.copy(lsoa_cols)
    random.shuffle(jumbled_lsoa_cols)
    all_lsoa_coefs = []
    for i in range(0,len(lsoa_cols),step):
        print(i)
        lsoa_cols_batch = jumbled_lsoa_cols[i:min(i+step,len(lsoa_cols))]
        lsoa_batch = [p.split('_')[1] for p in lsoa_cols_batch]
        
        #train_batch = houses_mod[houses_mod.Postcode.isin(postcodes_batch)]
        #Can leave out the unused postcode cols
        #train_batch = train_batch[[c for c in houses_mod.columns if 'postcode_' not in c or c in postcode_cols_batch]]
        
        tmp_houses_mod = houses_mod[houses_mod['LSOA Code'].isin(lsoa_batch)]
        train_batch = pd.concat([tmp_houses_mod, pd.get_dummies(tmp_houses_mod['LSOA Code'], prefix='lsoa')], axis=1)
        del tmp_houses_mod

        #houses_mod = houses_mod[houses_mod['LSOA Code'].isin(lsoa_has_50_rows.index[np.where(lsoa_has_50_rows)])]
        #train_batch = pd.concat([houses_mod, pd.get_dummies(houses_mod['LSOA Code'], prefix='lsoa')], axis=1)

        features = ['has_garage_01','has_garden_01','has_yard_01','has_other_outbuilding_01']
        features += ['home_type_House','home_type_Flat'] 
        features += lsoa_cols_batch[:-1]
        #For some reason this but not the other GLMs fails if all cat columns
        #  are included, even if no intercept is used, so need to exclude one
        this_glm = GLM(train_batch['price_per_sq_m_2022_by_lgd'], 
            train_batch[features]).fit()

        lsoa_coefs = this_glm.params.filter(like='lsoa').to_frame(name='lsoa_coef')
    
        #I think the excluded LSOA has a zero coefficient (if no intercept)
        print(f'Setting {lsoa_cols_batch[-1]} coef to 0 before normalising')
        lsoa_coefs.loc[lsoa_cols_batch[-1]] = 0.

        all_lsoa_coefs.append(lsoa_coefs)

    lsoa_coefs = pd.concat(all_lsoa_coefs).drop_duplicates()

    #These aren't necessarily centred on zero; re-centre here
    #  (OK to change their relationship with the non-lsoa coefs
    #   because this will only be used for the tooltip, not the calculator)
    lsoa_coefs['lsoa_coef'] -= lsoa_coefs.lsoa_coef.mean()
    
    #print(f'{lsoa_cols[-1]} coef is now {lsoa_coefs.loc[lsoa_cols[-1]].lsoa_coef:.1f}')
    #print(f"Its ppsqm is £{houses_mod[houses_mod['LSOA Code']==lsoa_cols[-1].split('_')[-1]].price_per_sq_m_2022_by_lgd.mean():.0f}")

    lsoa_coefs['LSOA Code'] = [s.split('_')[-1] for s in lsoa_coefs.index]
    lsoa_coefs = lsoa_coefs.reset_index()[['LSOA Code','lsoa_coef']]

    return lsoa_coefs

gc.collect()
print('\nFitting GLM by LSOA')
lsoa_coefs = calculate_glm_coefs_by_LSOA(houses)

def convert_price_to_colour(this_ppsqm, all_ppsqm, cmap='BrBG'):
    breaks = np.nanpercentile(all_ppsqm, [i*100/10 for i in range(0,11)])
    whichbin = np.argmax(this_ppsqm <= np.array(breaks))
    #this gives equal in 1-10 and one value in 0
    #cmap_ind = (whichbin/10 - 0.05)
    #whichbin/10 is 0.1 to 1.0; adjust to 0.05-0.95; then shrink to 0.14-0.86 to avoid plotting extreme colours
    #cmap_ind = (cmap_ind-0.5)*0.8 + 0.5
    #and adjust to avoid the middle ~white values - changes it to 0.14-0.46 and 0.54-0.86
    #cmap_ind = 0.14+0.8*(cmap_ind-0.14) if cmap_ind <= 0.5 else 1-0.8*(1-cmap_ind-0.14)-0.14    
    #return colors.to_hex(cm.get_cmap(cmap, 10)(cmap_ind))  

    #Make 14 colours (no 'middle' white) but don't use the edge 1s and middle 2
    fifteen_map = cm.get_cmap(cmap, 14)  #the index range for this is 0-11 inclusive
    cols = [fifteen_map(x) for x in [1,2,3,4,5,8,9,10,11,12]]  #list of length 10
    return colors.to_hex(cols[max(whichbin-1,0)])  #index the list with 0-9 inclusive

#Save 2022 value summaries by different methods
#Previously used 'simple method' of getting mean ppsqm by postcode for all/houses/flats,
#  and a 50x50 quantile grid using pd.qcut on lon and lat

#Get the LSOA names to join to the codes, and the geometries for the geojson output
soas = gpd.read_file('other_data/soa2001_ODNI.json')\
    .set_crs('EPSG:29902', allow_override=True)\
    .to_crs('EPSG:4326')
soas['SOA_LABEL'] = soas.SOA_LABEL.apply(lambda l: l.replace('_',' '))

def run_one_smooth_pass(summ_coefs, alpha=1000):
    '''
    summ_coefs: df containing Postcode, longitude, latitude, ppsqm_delta (to be smoothed)
                  and other columns that will be returned as is
    alpha: higher for less smoothing; alpha=3000 with min distance 0.002 and 3 passes seems good
    '''
    short_postcodes_list = summ_coefs.Postcode.apply(lambda x: x.split(' ')[0]).unique()

    #do by short postcode to avoid a massive cartesian product join
    new_summ_coefs = []
    for s_p in short_postcodes_list:
        one_s_p = summ_coefs[summ_coefs.Postcode.str.contains(s_p+' ')].copy()
        one_s_p['dum'] = 'dum'
        one_s_p_sq = one_s_p.merge(one_s_p, on='dum', how='outer')
        #Rough distance from coords as euclidean will be fine at this scale
        one_s_p_sq['dist'] = np.sqrt((one_s_p_sq.longitude_x - one_s_p_sq.longitude_y)**2 + (one_s_p_sq.latitude_x - one_s_p_sq.latitude_y)**2)
        one_s_p_sq = one_s_p_sq.loc[one_s_p_sq.dist <= 0.002]  #only allow close points to affect each other
        one_s_p_grped = (one_s_p_sq
            .assign(weight = lambda df: np.exp(-alpha*df.dist) * df.n_y/df.n_x,  #inv prop to distance and prop to number of properties, keeping weight=1 for the self row
                wt_mean_val_y = lambda df: df.mean_val_y * df.weight,
                wt_mean_price_per_sq_m_y = lambda df: df.mean_price_per_sq_m_y * df.weight,
                wt_ppsqm_delta_y = lambda df: df.ppsqm_delta_y * df.weight)
            .groupby('Postcode_x'))
        new_values = (one_s_p_grped.wt_ppsqm_delta_y.sum() / one_s_p_grped.weight.sum()).round(1)\
            .to_frame().reset_index()\
            .rename(columns={'Postcode_x': 'Postcode', 0: 'ppsqm_delta'})
        new_values['mean_val'] = (one_s_p_grped.wt_mean_val_y.sum() / one_s_p_grped.weight.sum()).round(0).values
        new_values['mean_price_per_sq_m'] = (one_s_p_grped.wt_mean_price_per_sq_m_y.sum() / one_s_p_grped.weight.sum()).round(0).values
        new_summ_coefs.append(new_values)
    new_summ_coefs = pd.concat(new_summ_coefs)

    new_summ_coefs = summ_coefs[['Postcode'] + [c for c in summ_coefs.columns if c not in new_summ_coefs.columns]]\
        .merge(new_summ_coefs, on='Postcode', how='left')

    return new_summ_coefs

def summarise_coefs(houses, pc_coefs, soas, suffix='', cmap='BrBG', alpha=None):
    summ_coefs = (houses.groupby(['Postcode','LSOA Code'], as_index=False).agg(
        n = ('value', len),
        longitude = ('Longitude', np.max),
        latitude = ('Latitude', np.max),
        mean_val = ('value_2022_by_lgd', lambda v: round(np.mean(v))),
        mean_size = ('area_m2', lambda a: round(np.mean(a))),
        mean_price_per_sq_m = ('price_per_sq_m_2022_by_lgd', lambda p: round(np.mean(p)))
        )
        .merge(pc_coefs, left_on='Postcode', right_on='postcode', how='inner')
        .rename(columns={'pc_coef_avg': 'ppsqm_delta'})
    )
    summ_coefs['ppsqm_delta'] = np.round(summ_coefs.ppsqm_delta, 1)

    #Do smoothing here; leave mean_price_per_sq_m as is but smooth ppsqm_delta
    if alpha is not None:
        for _ in [1,2,3]:
            summ_coefs = run_one_smooth_pass(summ_coefs, alpha=alpha)
    
    high_low_popup_colours = {
        'BrBG': ('#7f4909','#015c53'), 
        'RdBu': ('#a11228','#1b5a9b'), 
        'PRGn_r': ('#156c31','#6a2076')
    }[cmap]
    high_low_strings = (f"<span style='color: {high_low_popup_colours[1]}'><b>above</b></span>",
        f"<span style='color: {high_low_popup_colours[0]}'><b>below</b></span>")
    summ_coefs = (summ_coefs
        .merge(soas, left_on='LSOA Code', right_on='SOA_CODE', how='inner')
        .assign(popup_text = lambda df: df.apply(
            lambda row: f'<b>{row.Postcode}</b><br/>{row.n:g} properties<br/>'+\
                f'Mean value £{row.mean_val:,.0f}<br/>'+\
                f"<b>£{abs(row.ppsqm_delta):,.0f}</b> per sq. m {high_low_strings[0] if row.ppsqm_delta > 0 else high_low_strings[1]} average", axis=1))
        .assign(html_colour = lambda df: df.apply(lambda row: convert_price_to_colour(row.ppsqm_delta, summ_coefs.ppsqm_delta.values, cmap=cmap), axis=1))
        [['Postcode','SOA_LABEL','n','longitude','latitude',
          'mean_val','mean_size','mean_price_per_sq_m','ppsqm_delta',
          'html_colour','popup_text']]
    )
    return summ_coefs

def save_colour_bands_file(summ_coefs_df, output_filename):
    (summ_coefs_df.groupby('html_colour', as_index=False).agg(
        price_lower = ('ppsqm_delta', lambda x: int(np.min(x))),
        price_upper = ('ppsqm_delta', lambda x: int(np.max(x))),
        n_postcodes = ('Postcode', len))
        .sort_values('price_lower')
        .to_csv(output_filename, index=False)
    )

#Map data from GLM by postcode - all data
#summ_coefs = summarise_coefs(houses, pc_coefs_normd, soas, cmap='BrBG', alpha=None)
#summ_coefs.to_csv(f'static/ppsqm_nge10_glmmethod.csv', index=False)
#summ_coefs = summarise_coefs(houses, pc_coefs_normd, soas, cmap='BrBG', alpha=1000)
#summ_coefs.to_csv(f'static/ppsqm_nge10_glmmethod_smoothed_alpha1000.csv', index=False)
summ_coefs = summarise_coefs(houses, pc_coefs_normd, soas, cmap='BrBG', alpha=smoothing_alpha)
summ_coefs.to_csv(f'static/ppsqm_nge10_glmmethod_smoothed_alpha{smoothing_alpha}.csv', index=False)
#
#colour bands for plot
save_colour_bands_file(summ_coefs, 'static/colour_bands_ppsqm_nge10_glmmethod.csv')

#Map data from GLM by postcode - houses only
#summ_coefs_houses = summarise_coefs(houses[houses.home_type=='House'], pc_coefs_houses_normd, soas, suffix='_houses', cmap='RdBu')
#summ_coefs_houses.to_csv(f'static/ppsqm_nge10_glmmethod_houses.csv', index=False)
summ_coefs_houses = summarise_coefs(houses[houses.home_type=='House'], pc_coefs_houses_normd, soas, suffix='_houses', cmap='RdBu', alpha=smoothing_alpha)
summ_coefs_houses.to_csv(f'static/ppsqm_nge10_glmmethod_smoothed_alpha{smoothing_alpha}_houses.csv', index=False)
save_colour_bands_file(summ_coefs_houses, 'static/colour_bands_ppsqm_nge10_glmmethod_houses.csv')

#Map data from GLM by postcode - flats only
#summ_coefs_flats = summarise_coefs(houses[houses.home_type=='Flat'], pc_coefs_flats_normd, soas, suffix='_flats', cmap='PRGn_r')
#summ_coefs_flats.to_csv(f'static/ppsqm_nge10_glmmethod_flats.csv', index=False)
summ_coefs_flats = summarise_coefs(houses[houses.home_type=='Flat'], pc_coefs_flats_normd, soas, suffix='_flats', cmap='PRGn_r', alpha=smoothing_alpha)
summ_coefs_flats.to_csv(f'static/ppsqm_nge10_glmmethod_smoothed_alpha{smoothing_alpha}_flats.csv', index=False)
save_colour_bands_file(summ_coefs_flats, 'static/colour_bands_ppsqm_nge10_glmmethod_flats.csv')

#Calculator coefficients from GLM by short postcode
def summ_values_by_postcode(summ_coefs_for_calculator, postcode_column):
    res = (summ_coefs_for_calculator.groupby(postcode_column, as_index=False).agg(
        value_mean = ('ppsqm_delta', lambda v: np.round(np.mean(v),4)),
        value_pc10 = ('ppsqm_delta', lambda x: np.round(np.quantile(x, [0.1]),4)),
        value_pc90 = ('ppsqm_delta', lambda x: np.round(np.quantile(x, [0.9]),4))
        )
        .rename(columns={postcode_column: 'coef'})
    )
    return res

summ_coefs_for_calculator = summarise_coefs(houses, pc_coefs, soas, cmap='BrBG', alpha=smoothing_alpha)  #using the unnormalised coefficients
summ_coefs_for_calculator['Postcode_short'] = summ_coefs_for_calculator.Postcode.apply(lambda p: p.split(' ')[0])
# summ_coefs_short = (summ_coefs_for_calculator.groupby('Postcode_short', as_index=False).agg(
#     value_mean = ('ppsqm_delta', lambda v: np.round(np.mean(v),4)),
#     value_pc10 = ('ppsqm_delta', lambda x: np.round(np.quantile(x, [0.1]),4)),
#     value_pc90 = ('ppsqm_delta', lambda x: np.round(np.quantile(x, [0.9]),4))
#     )
#     .rename(columns={'Postcode_short': 'coef'})
# )
# summ_coefs_long = (summ_coefs_for_calculator.groupby('Postcode', as_index=False).agg(
#     value_mean = ('ppsqm_delta', lambda v: np.round(np.mean(v),4)),
#     value_pc10 = ('ppsqm_delta', lambda x: np.round(np.quantile(x, [0.1]),4)),
#     value_pc90 = ('ppsqm_delta', lambda x: np.round(np.quantile(x, [0.9]),4))
#     )
#     .rename(columns={'Postcode': 'coef'})
# )
summ_coefs_short = summ_values_by_postcode(summ_coefs_for_calculator, 'Postcode_short')
summ_coefs_long = summ_values_by_postcode(summ_coefs_for_calculator, 'Postcode')
summ_coefs_comb = pd.concat([non_pc_coefs, summ_coefs_short, summ_coefs_long])  #this is used again later
summ_coefs_comb.to_csv(f'static/calculator_coefs_smoothed_alpha{smoothing_alpha}.csv', index=False)


#Map data from GLM by LSOA
summ_soas_coefs = (houses.groupby('LSOA Code', as_index=False).agg(
    n = ('value', len),
    longitude = ('Longitude', np.max),
    latitude = ('Latitude', np.max),
    mean_val = ('value_2022_by_lgd', np.mean),
    mean_size = ('area_m2', np.mean)
    )
    .merge(lsoa_coefs, on='LSOA Code', how='inner')
    .rename(columns={'lsoa_coef': 'ppsqm_delta'})
)
high_low_strings = ("<span style='color: #015c53'><b>above</b></span>", "<span style='color: #7f4909'><b>below</b></span>")
summ_soas_coefs = (summ_soas_coefs
    .merge(soas, left_on='LSOA Code', right_on='SOA_CODE', how='inner')
    .assign(popup_text = lambda df: df.apply(
        lambda row: f'<b>{row.SOA_LABEL}</b><br/>{row.n:g} properties<br/>'+\
            f'Mean value £{row.mean_val:,.0f}<br/>'+\
            f"<b>£{abs(row.ppsqm_delta):,.0f}</b> per sq. m {high_low_strings[0] if row.ppsqm_delta > 0 else high_low_strings[1]} average", axis=1))
    .assign(html_colour = lambda df: df.apply(lambda row: 
        convert_price_to_colour(row.ppsqm_delta, summ_soas_coefs.ppsqm_delta.values), axis=1))
    .query("n >= 10")
)
summ_soas_coefs = gpd.GeoDataFrame(summ_soas_coefs)[['LSOA Code','n','ppsqm_delta',
                                                     'html_colour','popup_text','geometry']]
#simplify shapes to give a smaller file
summ_soas_coefs['geometry'] = summ_soas_coefs.geometry.simplify(0.0002)
summ_soas_coefs.to_file('static/ppsqm_nge10_LSOAglmmethod.json', driver='GeoJSON')
#colour bands for plot
(summ_soas_coefs.groupby('html_colour', as_index=False).agg(
    price_lower = ('ppsqm_delta', lambda x: int(np.min(x))),
    price_upper = ('ppsqm_delta', lambda x: int(np.max(x))),
    n_lsoas = ('LSOA Code', len))
    .sort_values('price_lower')
    .to_csv(f'static/colour_bands_ppsqm_nge10_LSOAglmmethod.csv', index=False)
)

#Nearest short postcodes, for the calculator output
short_postcodes = (ghouses
                   .assign(postcode_short = lambda df: df.postcode.apply(lambda s: s.split(' ')[0]))
                   .groupby('postcode_short', as_index=False)[['Longitude','Latitude']].mean()
                   .assign(dum = 1)
                  )
(short_postcodes.merge(short_postcodes, on='dum', suffixes=['_a','_b'])
 .assign(distance_degrees = lambda df: df.apply(lambda row: ((row['Longitude_a']-row['Longitude_b'])**2 + (row['Latitude_a']-row['Latitude_b'])**2)**0.5, axis=1))
 .query('postcode_short_a != postcode_short_b')
 .sort_values('distance_degrees')
 .groupby('postcode_short_a', as_index=False)
 .agg(nearest_pcs_1 = ('postcode_short_b', lambda p: p[0:1]),
      nearest_pcs_2 = ('postcode_short_b', lambda p: p[1:2]),
      nearest_pcs_3 = ('postcode_short_b', lambda p: p[2:3]),
      nearest_pcs_4 = ('postcode_short_b', lambda p: p[3:4]),
      nearest_pcs_5 = ('postcode_short_b', lambda p: p[4:5]))
 .rename(columns={'postcode_short_a': 'postcode'})
 .query('postcode in @summ_coefs_comb.coef')  #This is for the calculator so only include postcodes that got GLM coefs
 .to_csv('static/postcodeshort_nearest_five.csv', index=False)
)

#Nearest short to long postcoddes
(ghouses.drop_duplicates(subset=['postcode']).assign(dum = 1)
 .merge(short_postcodes, on='dum', suffixes=['_a','_b'])
 .assign(distance_degrees = lambda df: df.apply(lambda row: ((row['Longitude_a']-row['Longitude_b'])**2 + (row['Latitude_a']-row['Latitude_b'])**2)**0.5, axis=1))
 .rename(columns={'postcode_short': 'postcode_short_b'})
 .assign(postcode_short_a = lambda df: df.postcode.apply(lambda s: s.split(' ')[0]))
 .query('postcode_short_a != postcode_short_b')
 .sort_values('distance_degrees')
 .groupby('postcode', as_index=False)
 .agg(nearest_pcs_1 = ('postcode_short_b', lambda p: p[0:1]),
      nearest_pcs_2 = ('postcode_short_b', lambda p: p[1:2]),
      nearest_pcs_3 = ('postcode_short_b', lambda p: p[2:3]),
      nearest_pcs_4 = ('postcode_short_b', lambda p: p[3:4]),
      nearest_pcs_5 = ('postcode_short_b', lambda p: p[4:5]))
 #.rename(columns={'Postcode': 'postcode'})
 .query('postcode in @summ_coefs_comb.coef')
 .to_csv('static/postcodelongtoshort_nearest_five.csv', index=False)
)

