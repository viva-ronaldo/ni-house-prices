#Mean ppsqm is done as mean(ppsqm), not mean(value)/mean(area)
#  which seems more appropriate to avoid one large house dominating
#Try median?

import numpy as np
import pandas as pd
import re, glob, copy, random
import geopandas as gpd
from matplotlib import colors, cm
from statsmodels.api import GLM, add_constant

test_files = glob.glob('LPS_data/test_*_*.csv')

houses = pd.concat([pd.read_csv(f) for f in test_files])
print(f'Read {houses.shape[0]} rows')

houses['desc'] = houses.desc.fillna('')
houses = houses[~houses.desc.str.contains('presbytery|convent|sheltered|mobile home|shop|premises|public house|showhouse|wardens house|community|unfinished')].copy()
houses = houses[~houses.desc.isin(['','outbuilding','hut','house (caravan)','caravan','h (caravan)& o','houses (huts) garden'])]
print(f'{houses.shape[0]} after dropping empty or non-standard descs')

#Add lon, lat from Doogal postcodes 
postcodes = pd.read_csv('other_data/full_postcode_latlons_from_doogal.csv', 
    usecols=['Postcode','Latitude','Longitude','LSOA Code'])
houses = houses.merge(postcodes, left_on='postcode', right_on='Postcode')

print(f"Dropping {(houses.type=='Mixed').sum()} Mixed cases from {houses.shape[0]} total")
houses = houses[houses.type=='Domestic'].copy()

#19000 is used in some suspicious cases; may be in bad disrepair or some other situation
print(f'Dropping {(houses.value==19000).sum()} cases of £19,000')
houses = houses[houses.value!=19000].copy()

#Drop very big ones which would stretch a linear ppsqm fit or be not normal houses
print(f'Dropping {(houses.area_m2>800).sum()} cases bigger than 800m^2')
houses = houses[houses.area_m2<=800].copy()

houses = houses.set_index('id')

#
print(f'Dropping {len(houses[houses.area_m2 <= 2])} records with 0-2 m^2 area')
houses = houses[houses.area_m2 > 2].copy()

#Some look like typos for area
houses['area_m2'] = houses.apply(lambda row: 41 if row.name==201786 else row['area_m2'], axis=1)
houses['area_m2'] = houses.apply(lambda row: 61 if row.name==204122 else row['area_m2'], axis=1)
houses['area_m2'] = houses.apply(lambda row: 61 if row.name==204152 else row['area_m2'], axis=1)
houses['area_m2'] = houses.apply(lambda row: 53 if row.name==204122 else row['area_m2'], axis=1)
houses['area_m2'] = houses.apply(lambda row: 53 if row.name==204152 else row['area_m2'], axis=1)
houses['area_m2'] = houses.apply(lambda row: 110 if row.name in [260850,260849,260848,260846] else row['area_m2'], axis=1)  #map shows 110 but prop pages show 11
houses['area_m2'] = houses.apply(lambda row: 93 if row.name==291239 else row['area_m2'], axis=1)  #not 963
houses['area_m2'] = houses.apply(lambda row: 81 if row.name==251067 else row['area_m2'], axis=1)  #not 810
houses['area_m2'] = houses.apply(lambda row: 106 if row.name==242378 else row['area_m2'], axis=1)  #not 1006
houses['area_m2'] = houses.apply(lambda row: 78.9 if row.name==188692 else row['area_m2'], axis=1)  #probably, not 789
houses['area_m2'] = houses.apply(lambda row: 45.8 if row.name==181623 else row['area_m2'], axis=1)  #not 458
houses['area_m2'] = houses.apply(lambda row: 86 if row.name==246832 else row['area_m2'], axis=1)  #ish, not 23
houses['area_m2'] = houses.apply(lambda row: 118.15 if row.name==144572 else row['area_m2'], axis=1)  #this is the value in the map view
houses['area_m2'] = houses.apply(lambda row: 94 if row.name==153232 else row['area_m2'], axis=1)  #probably two 47s
houses['area_m2'] = houses.apply(lambda row: 97 if row.name==153177 else row['area_m2'], axis=1)  #probably two 48.5s
houses['area_m2'] = houses.apply(lambda row: 96 if row.name==159505 else row['area_m2'], axis=1)  #probably two 48s
houses['area_m2'] = houses.apply(lambda row: 47 if row.name==2000077 else row['area_m2'], axis=1)  #probably, not 17 (over 2 floors) as more expensive than a 43m2
houses['area_m2'] = houses.apply(lambda row: 99 if row.name==99782 else row['area_m2'], axis=1)  #not 19
houses['area_m2'] = houses.apply(lambda row: 98 if row.name==109693 else row['area_m2'], axis=1)  #probably two 49s
houses['area_m2'] = houses.apply(lambda row: 92 if row.name==89430 else row['area_m2'], axis=1)  #probably two 46s
houses['area_m2'] = houses.apply(lambda row: 68 if row.name==89043 else row['area_m2'], axis=1)  #probably two 34s
houses['area_m2'] = houses.apply(lambda row: 94 if row.name==88844 else row['area_m2'], axis=1)  #probably two 47s

houses['value'] = houses.apply(lambda row: 1000000 if row.name==243125 else row['value'], axis=1)  #change 64 from 100k; 62 nearby is 1m
houses['value'] = houses.apply(lambda row: 190000 if row.name==207581 else row['value'], axis=1)  #probably, not 19,000
houses['value'] = houses.apply(lambda row: 95000 if row.name==194274 else row['value'], axis=1)  #probably, not 19,500

houses['price_per_sq_m'] = houses.value/houses.area_m2

print(f'Dropping {len(houses[(houses.value <= 10000) & (houses.price_per_sq_m < 100)])} cases with value a few thousand and ppsqm < £100/m^2')
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
    if desc_string[:5] == 'house':
        return 'House'
    elif desc_string[:2] in ['h ','h(','hg','ho','h+','h&','h/','h.','h0','hy','h<','h,']:
        return 'House'
    elif desc_string in ['bungalow','cottage','h']:
        return 'House'
    elif re.search('flat|flt|bedsit|penthouse|apartment|apart|flast |flzt |flar |room |rm |rooms |fl 1st|flay ', desc_string):
        return 'Flat'
    elif desc_string=='<1st>':
        return 'Flat'
    else:
        return 'Other'

houses['home_type'] = houses.desc.apply(convert_desc_to_home_type)

#Drop the few Others
print(f"Dropping {(houses.home_type=='Other').sum()} cases not identified as House or Flat")
houses = houses[houses.home_type.isin(['House','Flat'])].copy()

#Assign flat floor number? Not needed for model.

#Add other fields
houses['has_garden'] = houses.desc.str.contains('garden')
houses['has_yard'] = houses.desc.str.contains('yard')
#Most of the time has_garage is Yes when outbuilding is mentioned, but not always; these could be other buildings
houses['has_other_outbuilding'] = houses.apply(lambda row: 'Yes' if (re.search('outbuild', row['desc']) and row['has_garage']=='No') else 'No', axis=1)


#Get price change factors from 2005 to 2022
#Nationwide
price_change_nw = pd.read_csv('other_data/ni-hpi-by-property-type-q1-2005---q2-2022.csv').set_index('Quarter_Year')
price_change_2005_2022_nw = price_change_nw.loc['Q1 2022']['NI_Residential_Property_Price_Index'] / price_change_nw.loc['Q1 2005']['NI_Residential_Property_Price_Index']
houses['value_2022_by_nw'] = houses.value*price_change_2005_2022_nw
houses['price_per_sq_m_2022_by_nw'] = houses.value_2022_by_nw/houses.area_m2
#By LGD
price_change_lgd = pd.read_csv('other_data/standardised-price-and-index-by-lgd-q1-2005---q2-2022.csv').set_index('Quarter_Year')
price_changes_2005_2022_lgd = price_change_lgd.loc['Q1 2022'] / price_change_lgd.loc['Q1 2005']
price_changes_2005_2022_lgd = price_changes_2005_2022_lgd.loc[[c for c in price_changes_2005_2022_lgd.index if '_HPI' in c]]
price_changes_2005_2022_lgd.index = [c[:-4].replace('_',' ').replace('Armagh City','Armagh City,').replace('Newry','Newry,') for c in price_changes_2005_2022_lgd.index]
price_changes_2005_2022_lgd.index.name = 'LGD'
lgd_geom = gpd.read_file('other_data/LAD_MAY_2022_UK_BSC_V3/').to_crs("EPSG:4326")   #convert from something to lat lon geometry
ghouses = gpd.GeoDataFrame(houses, geometry=gpd.points_from_xy(houses.Longitude, houses.Latitude))\
    .set_crs("EPSG:4326")
#Join on the 2022 value calculated via LGD
houses = houses.join(
    ghouses[['address','postcode','value','geometry']]
    .assign(id_tmp = lambda df: df.index)
    .sjoin(lgd_geom[['LAD22NM','geometry']], how='inner')
    .merge(price_changes_2005_2022_lgd.to_frame(name='value_factor_by_lgd'), 
        left_on='LAD22NM', right_on='LGD', how='left')
    .rename(columns={'LAD22NM': 'lgd'})
    .set_index('id_tmp').rename_axis('id')
    .assign(value_2022_by_lgd = lambda df: df['value']*df['value_factor_by_lgd'])
    [['lgd','value_2022_by_lgd']],
    how='left'
)
houses['price_per_sq_m_2022_by_lgd'] = houses.value_2022_by_lgd/houses.area_m2

print(f'Filling in {houses.price_per_sq_m_2022_by_lgd.isnull().sum()} 2022 values with nw due to missing LGD value')
houses['value_2022_by_lgd'].fillna(houses.value_2022_by_nw, inplace=True)
houses['price_per_sq_m_2022_by_lgd'].fillna(houses.price_per_sq_m_2022_by_nw, inplace=True)

#Define terms for models
houses['has_garage_01'] = houses.has_garage.apply(lambda g: 1 if g=='Yes' else 0)
houses['has_other_outbuilding_01'] = houses.has_other_outbuilding.apply(lambda g: 1 if g=='Yes' else 0)
houses['has_garden_01'] = houses.has_garden.apply(lambda g: 1 if g else 0)
houses['has_yard_01'] = houses.has_yard.apply(lambda g: 1 if g else 0)


#Fit models
def calculate_glm_coefs(df, include_home_type_term=True):
    houses_mod = houses.copy()
    if include_home_type_term:
        houses_mod = pd.concat([houses_mod, pd.get_dummies(houses_mod['home_type'], prefix='home_type')], axis=1)

    #This gets too big to do in one go
    #houses_mod = pd.concat([houses_mod, pd.get_dummies(houses_mod['Postcode'], prefix='postcode')], axis=1)
    #postcode_cols = [c for c in houses_mod.columns if 'postcode_' in c]
    postcode_has_10_rows = (houses_mod.Postcode.value_counts() >= 10)
    #postcode_cols = [c for c in postcode_cols if c.split('_')[1] in postcode_has_10_rows.index[np.where(postcode_has_10_rows)]]
    postcode_cols = ['postcode_'+c for c in houses_mod.Postcode.unique() if c in postcode_has_10_rows.index[np.where(postcode_has_10_rows)]]
    print(f'Using {len(postcode_cols)} postcode terms with >= 10 rows')

    #Fit three times and average the coefficients
    glm_res = []
    for e in [1,2,3]:
        #Need to jumble up the postcodes to avoid fitting only one area at once
        jumbled_postcode_cols = copy.copy(postcode_cols)
        random.shuffle(jumbled_postcode_cols)

        postcode_glms = []
        for i in range(0,len(postcode_cols),500):
            postcode_cols_batch = jumbled_postcode_cols[i:min(i+500,len(postcode_cols))]
            postcodes_batch = [p.split('_')[1] for p in postcode_cols_batch]
            
            #train_batch = houses_mod[houses_mod.Postcode.isin(postcodes_batch)]
            #Can leave out the unused postcode cols
            #train_batch = train_batch[[c for c in houses_mod.columns if 'postcode_' not in c or c in postcode_cols_batch]]
            
            tmp_houses_mod = houses_mod[houses_mod.Postcode.isin(postcodes_batch)]
            train_batch = pd.concat([tmp_houses_mod, pd.get_dummies(tmp_houses_mod['Postcode'], prefix='postcode')], axis=1)
            
            try:
                features = ['has_garage_01','has_garden_01','has_yard_01','has_other_outbuilding_01']
                if include_home_type_term:
                    features += ['home_type_House','home_type_Flat'] 
                features += postcode_cols_batch
                this_glm = GLM(train_batch['price_per_sq_m_2022_by_lgd'], 
                    add_constant(train_batch[features])).fit()
                postcode_glms.append(this_glm)
                #print('Area coef = ', this_glm.params.area_m2)
            except np.linalg.LinAlgError:
                print('This batch GLM did not converge; skipping')
            #print(i, len(postcode_glms))

        glm_res.append(postcode_glms)
    #print(len(glm_res))

    pc_coefs = pd.concat([g.params for g in glm_res[0]]).filter(like='postcode').to_frame(name='coef_1')\
        .join(pd.concat([g.params for g in glm_res[1]]).filter(like='postcode').to_frame(name='coef_2'))\
        .join(pd.concat([g.params for g in glm_res[2]]).filter(like='postcode').to_frame(name='coef_3'))
    pc_coefs['pc_coef_avg'] = (pc_coefs.coef_1 + pc_coefs.coef_2 + pc_coefs.coef_3)/3
    pc_coefs['postcode'] = [s.split('_')[-1] for s in pc_coefs.index]
    pc_coefs = pc_coefs.reset_index()[['postcode','pc_coef_avg']]

    #Save non-postcode coefs for calculator
    non_pc_coefs = (pd.concat([g.params for g in np.ravel(glm_res)]).filter(regex='const|area|garage|garden|yard|outbuilding|home_type')
        .to_frame(name='value')
        .assign(coef = lambda df: df.index).reset_index(drop=True)
        .groupby('coef', as_index=False).agg(value_mean = ('value', np.mean))
    )

    return pc_coefs, non_pc_coefs

#All data
pc_coefs, non_pc_coefs = calculate_glm_coefs(houses)
#Houses only
pc_coefs_houses, _ = calculate_glm_coefs(houses[houses.home_type=='House'])
#Flats only
pc_coefs_flats, _ = calculate_glm_coefs(houses[houses.home_type=='Flat'])

def calculate_glm_coefs_by_LSOA(df):
    houses_mod = houses.copy()
    houses_mod = pd.concat([houses_mod, pd.get_dummies(houses_mod['home_type'], prefix='home_type')], axis=1)

    lsoa_has_10_rows = (houses_mod['LSOA Code'].value_counts() >= 10)
    lsoa_cols = ['lsoa_'+c for c in houses_mod['LSOA Code'].unique() if c in lsoa_has_10_rows.index[np.where(lsoa_has_10_rows)]]
    print(f'Using {len(lsoa_cols)} LSOA terms with >= 10 rows')

    houses_mod = houses_mod[houses_mod['LSOA Code'].isin(lsoa_has_10_rows.index[np.where(lsoa_has_10_rows)])]
    train_batch = pd.concat([houses_mod, pd.get_dummies(houses_mod['LSOA Code'], prefix='lsoa')], axis=1)

    features = ['has_garage_01','has_garden_01','has_yard_01','has_other_outbuilding_01']
    features += ['home_type_House','home_type_Flat'] 
    features += lsoa_cols
    this_glm = GLM(train_batch['price_per_sq_m_2022_by_lgd'], 
        add_constant(train_batch[features])).fit()

    lsoa_coefs = this_glm.params.filter(like='lsoa').to_frame(name='lsoa_coef')
    lsoa_coefs['LSOA Code'] = [s.split('_')[-1] for s in lsoa_coefs.index]
    lsoa_coefs = lsoa_coefs.reset_index()[['LSOA Code','lsoa_coef']]

    return lsoa_coefs

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

    #Make 15 colours but don't use the edge 2s
    fifteen_map = cm.get_cmap(cmap, 15)
    cols = [fifteen_map(x) for x in [2,3,4,5,6,7,8,9,10,11]]
    return colors.to_hex(cols[max(whichbin-1,0)])

#Save 2022 value summaries by different methods

#i) Price per sq m by postcode
summ = houses.groupby(['Postcode','LSOA Code'], as_index=False).agg(
    n = ('value', len),
    longitude = ('Longitude', np.max),
    latitude = ('Latitude', np.max),
    mean_val = ('value_2022_by_lgd', np.mean),
    mean_size = ('area_m2', np.mean),
    mean_price_per_sq_m = ('price_per_sq_m_2022_by_lgd', np.mean),
    sd_price_per_sq_m = ('price_per_sq_m', np.std),
    frac_houses = ('home_type', lambda h: np.mean(h=='House')),
    frac_w_garage = ('has_garage', lambda g: np.mean(g=='Yes')),
    frac_w_garden = ('has_garden', lambda g: np.mean(g))
)

#Wait to add SOA below before writing
# (summ
#  .assign(popup_text = lambda df: df.apply(lambda row: f'<b>{row.Postcode}</b><br/>{row.n:g} properties<br/>Mean value £{row.mean_val:,.0f}<br/>Mean £{row.mean_price_per_sq_m:,.0f} / m<sup>2</sup>', axis=1))
#  .assign(html_colour = lambda df: df.apply(lambda row: convert_price_to_colour(row.mean_price_per_sq_m, summ.mean_price_per_sq_m.values), axis=1))
#  [['Postcode','n','longitude','latitude','mean_val','mean_size','mean_price_per_sq_m','html_colour','popup_text']]
#  .query("n >= 10")
#  .to_csv('ppsqm_nge10_simplemethod.csv', index=False)
# )

#iia) Ppsqm by postcode, houses only
# summ_houses = (houses[houses.home_type=='House']
#     .groupby('Postcode', as_index=False)
#     .agg(
#         n = ('value', len),
#         longitude = ('Longitude', np.max),
#         latitude = ('Latitude', np.max),
#         mean_val = ('value_2022_by_lgd', np.mean),
#         mean_size = ('area_m2', np.mean),
#         mean_price_per_sq_m = ('price_per_sq_m_2022_by_lgd', np.mean)
#         )
# )
# summ_houses = (summ_houses
#     .assign(popup_text = lambda df: df.apply(lambda row: f'<b>{row.Postcode}</b><br/>{row.n:g} houses<br/>Mean value £{row.mean_val:,.0f}<br/>Mean £{row.mean_price_per_sq_m:,.0f} / m<sup>2</sup>', axis=1))
#     .assign(html_colour = lambda df: df.apply(lambda row: convert_price_to_colour(row.mean_price_per_sq_m, summ_houses.mean_price_per_sq_m.values, cmap='RdBu'), axis=1))
#     [['Postcode','n','longitude','latitude','mean_price_per_sq_m','html_colour','popup_text']]
#     .query("n >= 10")
# )
# summ_houses.to_csv('static/ppsqm_nge10_simplemethod_houses.csv', index=False)
# #colour bands for plot
# (summ_houses.groupby('html_colour', as_index=False).agg(
#     price_lower = ('mean_price_per_sq_m', lambda x: int(np.min(x))),
#     price_upper = ('mean_price_per_sq_m', lambda x: int(np.max(x))))
#     .sort_values('price_lower')
#     .to_csv('static/colour_bands_ppsqm_nge10_simplemethod_houses.csv', index=False)
# )

#iib) Ppsqm by postcode, flats only
# summ_flats = (houses[houses.home_type=='Flat']
#     .groupby('Postcode', as_index=False)
#     .agg(
#         n = ('value', len),
#         longitude = ('Longitude', np.max),
#         latitude = ('Latitude', np.max),
#         mean_val = ('value_2022_by_lgd', np.mean),
#         mean_size = ('area_m2', np.mean),
#         mean_price_per_sq_m = ('price_per_sq_m_2022_by_lgd', np.mean)
#         )
# )
# summ_flats = (summ_flats
#     .assign(popup_text = lambda df: df.apply(lambda row: f'<b>{row.Postcode}</b><br/>{row.n:g} flats<br/>Mean value £{row.mean_val:,.0f}<br/>Mean £{row.mean_price_per_sq_m:,.0f}/m<sup>2</sup>', axis=1),
#         html_colour = lambda df: df.apply(lambda row: convert_price_to_colour(row.mean_price_per_sq_m, summ_flats.mean_price_per_sq_m.values, cmap='PRGn_r'), axis=1))
#     [['Postcode','n','longitude','latitude','mean_price_per_sq_m','html_colour','popup_text']]
#     .query("n >= 10")
# )
# summ_flats.to_csv('static/ppsqm_nge10_simplemethod_flats.csv', index=False)
# #colour bands for plot
# (summ_flats.groupby('html_colour', as_index=False).agg(
#     price_lower = ('mean_price_per_sq_m', lambda x: int(np.min(x))),
#     price_upper = ('mean_price_per_sq_m', lambda x: int(np.max(x))),
#     n_postcodes = ('Postcode', len))
#     .sort_values('price_lower')
#     .to_csv('static/colour_bands_ppsqm_nge10_simplemethod_flats.csv', index=False)
# )

#iii) Ppsqm on 50x50 lon-lat quantile grid
def summarise_postcodes_in_box(postcodes):
    deduped_postcodes = np.sort(np.unique(postcodes))
    if len(deduped_postcodes) <= 3:
        return ', '.join(deduped_postcodes)
    else:
        return ', '.join(deduped_postcodes[:3]) + ', and more'

# summ_50x50 = (houses
#     .assign(lon_box = lambda df: pd.qcut(df.Longitude, 50),
#         lat_box = lambda df: pd.qcut(df.Latitude, 50))
#     .groupby(['lon_box','lat_box'])
#     .agg(
#         n = ('address', len),
#         postcodes = ('Postcode', summarise_postcodes_in_box),
#         longitude = ('Longitude', np.mean),
#         latitude = ('Latitude', np.mean),
#         mean_size = ('area_m2', np.mean),
#         mean_val = ('value_2022_by_lgd', np.mean),
#         mean_price_per_sq_m = ('price_per_sq_m_2022_by_lgd', np.mean)
#         )
# )
# (summ_50x50
#     .assign(popup_text = lambda df: df.apply(lambda row: f'<b>{row.postcodes}</b><br/>{row.n:g} properties<br/>Mean £{row.mean_price_per_sq_m:,.0f} / m<sup>2</sup>', axis=1),
#         html_colour = lambda df: df.apply(lambda row: convert_price_to_colour(row.mean_price_per_sq_m, summ_50x50.mean_price_per_sq_m.values), axis=1))
#     [['n','postcodes','longitude','latitude','mean_price_per_sq_m','html_colour','popup_text']]
#     .query("n >= 10")
#     .to_csv('static/ppsqm_nge10_50x50simplemethod.csv', index=False)
# )

#iv) GLM coef by LSOA
soas = gpd.read_file('other_data/soa2001_ODNI.json')\
    .set_crs('EPSG:29902', allow_override=True)\
    .to_crs('EPSG:4326')
soas['SOA_LABEL'] = soas.SOA_LABEL.apply(lambda l: l.replace('_',' '))

# summ_soas = (houses
#     .groupby('LSOA Code', as_index=False)
#     .agg(
#         n = ('address', len),
#         postcodes = ('Postcode', summarise_postcodes_in_box),
#         longitude = ('Longitude', np.mean),
#         latitude = ('Latitude', np.mean),
#         mean_size = ('area_m2', np.mean),
#         mean_val = ('value_2022_by_lgd', np.mean),
#         mean_price_per_sq_m = ('price_per_sq_m_2022_by_lgd', np.mean),
#         pc10_price_per_sq_m = ('price_per_sq_m_2022_by_lgd', lambda x: np.nanquantile(x, [0.10])),
#         pc90_price_per_sq_m = ('price_per_sq_m_2022_by_lgd', lambda x: np.nanquantile(x, [0.90]))
#         )
# )
# summ_soas = (summ_soas
#     .assign(html_colour = lambda df: df.apply(lambda row: convert_price_to_colour(row.mean_price_per_sq_m, summ_soas.mean_price_per_sq_m.values), axis=1))
#     .query("n >= 10")
# )
# summ_soas = summ_soas.merge(soas, left_on='LSOA Code', right_on='SOA_CODE', how='inner')
# summ_soas['popup_text'] = summ_soas.apply(lambda row: f"<b>{row.SOA_LABEL}</b><br/>{row.n:g} properties<br/>Mean £{row.mean_price_per_sq_m:,.0f} / m<sup>2</sup><br/>(~ £{row.pc10_price_per_sq_m:,.0f} - {row.pc90_price_per_sq_m:,.0f})", axis=1)
# summ_soas = gpd.GeoDataFrame(summ_soas)[['LSOA Code','n','mean_price_per_sq_m','html_colour','popup_text','geometry']]
# summ_soas.to_file('static/ppsqm_nge10_LSOAsimplemethod.json', driver='GeoJSON')
# #colour bands for plot
# (summ_soas.groupby('html_colour', as_index=False).agg(
#     price_lower = ('mean_price_per_sq_m', lambda x: int(np.min(x))),
#     price_upper = ('mean_price_per_sq_m', lambda x: int(np.max(x))),
#     n_postcodes = ('Postcode', len))
#     .sort_values('price_lower')
#     .to_csv('static/colour_bands_ppsqm_nge10_LSOAsimplemethod.csv', index=False)
# )
summ_soas_coefs = (houses.groupby('LSOA Code', as_index=False).agg(
    n = ('value', len),
    longitude = ('Longitude', np.max),
    latitude = ('Latitude', np.max),
    mean_val = ('value_2022_by_lgd', np.mean),
    mean_size = ('area_m2', np.mean)
    )
    .merge(lsoa_coefs, on='LSOA Code', how='inner')
    .rename(columns={'lsoa_coef': 'mean_price_per_sq_m'})
)
summ_soas_coefs = (summ_soas_coefs
    .merge(soas, left_on='LSOA Code', right_on='SOA_CODE', how='inner')
    .assign(popup_text = lambda df: df.apply(lambda row: f'<b>{row.SOA_LABEL}</b><br/>{row.n:g} properties<br/>Mean value £{row.mean_val:,.0f}<br/>Mean £{row.mean_price_per_sq_m:+,.0f} / m<sup>2</sup> compared to average', axis=1))
    .assign(html_colour = lambda df: df.apply(lambda row: convert_price_to_colour(row.mean_price_per_sq_m, summ_soas_coefs.mean_price_per_sq_m.values), axis=1))
    .query("n >= 10")
)
summ_soas_coefs = gpd.GeoDataFrame(summ_soas_coefs)[['LSOA Code','n','mean_price_per_sq_m','html_colour','popup_text','geometry']]
summ_soas_coefs.to_file('static/ppsqm_nge10_LSOAglmmethod.json', driver='GeoJSON')
#colour bands for plot
(summ_soas_coefs.groupby('html_colour', as_index=False).agg(
    price_lower = ('mean_price_per_sq_m', lambda x: int(np.min(x))),
    price_upper = ('mean_price_per_sq_m', lambda x: int(np.max(x))),
    n_lsoas = ('LSOA Code', len))
    .sort_values('price_lower')
    .to_csv(f'static/colour_bands_ppsqm_nge10_LSOAglmmethod.csv', index=False)
)


#i) Add SOA and finish writing simple method
(summ
    .merge(soas, left_on='LSOA Code', right_on='SOA_CODE', how='inner')
    .assign(popup_text = lambda df: df.apply(lambda row: f'<b>{row.Postcode}</b><br/>{row.n:g} properties<br/>Mean value £{row.mean_val:,.0f}<br/>Mean £{row.mean_price_per_sq_m:,.0f} / m<sup>2</sup>', axis=1))
    .assign(html_colour = lambda df: df.apply(lambda row: convert_price_to_colour(row.mean_price_per_sq_m, summ.mean_price_per_sq_m.values), axis=1))
    [['Postcode','SOA_LABEL','n','longitude','latitude','mean_val','mean_size','mean_price_per_sq_m','html_colour','popup_text']]
    .query("n >= 10")
    .to_csv('static/ppsqm_nge10_simplemethod.csv', index=False)
)

#v) GLM coefficient by postcode
def save_glm_coefs_files(houses, pc_coefs, soas, suffix='', cmap='BrBG'):
    summ_coefs = (houses.groupby(['Postcode','LSOA Code'], as_index=False).agg(
        n = ('value', len),
        longitude = ('Longitude', np.max),
        latitude = ('Latitude', np.max),
        mean_val = ('value_2022_by_lgd', np.mean),
        mean_size = ('area_m2', np.mean)
        )
        .merge(pc_coefs, left_on='Postcode', right_on='postcode', how='inner')
        .rename(columns={'pc_coef_avg': 'mean_price_per_sq_m'})
    )
    summ_coefs = (summ_coefs
        .merge(soas, left_on='LSOA Code', right_on='SOA_CODE', how='inner')
        .assign(popup_text = lambda df: df.apply(lambda row: f'<b>{row.Postcode}</b><br/>{row.n:g} properties<br/>Mean value £{row.mean_val:,.0f}<br/>Mean £{row.mean_price_per_sq_m:+,.0f} / m<sup>2</sup> compared to average', axis=1))
        .assign(html_colour = lambda df: df.apply(lambda row: convert_price_to_colour(row.mean_price_per_sq_m, summ_coefs.mean_price_per_sq_m.values, cmap=cmap), axis=1))
        [['Postcode','SOA_LABEL','n','longitude','latitude','mean_val','mean_size','mean_price_per_sq_m','html_colour','popup_text']]
        .query("n >= 10")
    )
    summ_coefs.to_csv(f'static/ppsqm_nge10_glmmethod{suffix}.csv', index=False)
    #colour bands for plot
    (summ_coefs.groupby('html_colour', as_index=False).agg(
        price_lower = ('mean_price_per_sq_m', lambda x: int(np.min(x))),
        price_upper = ('mean_price_per_sq_m', lambda x: int(np.max(x))),
        n_postcodes = ('Postcode', len))
        .sort_values('price_lower')
        .to_csv(f'static/colour_bands_ppsqm_nge10_glmmethod{suffix}.csv', index=False)
    )
    return summ_coefs
#All data
summ_coefs = save_glm_coefs_files(houses, pc_coefs, soas, cmap='BrBG')
#Houses only
_ = save_glm_coefs_files(houses[houses.home_type=='House'], pc_coefs_houses, soas, suffix='_houses', cmap='RdBu')
#Flats only
_ = save_glm_coefs_files(houses[houses.home_type=='Flat'], pc_coefs_flats, soas, suffix='_flats', cmap='PRGn_r')


#vi) GLM coefficient by short postcode
summ_coefs['Postcode_short'] = summ_coefs.Postcode.apply(lambda p: p.split(' ')[0])
summ_coefs_short = (summ_coefs.groupby('Postcode_short', as_index=False).agg(
    value_mean = ('mean_price_per_sq_m', np.mean),
    value_pc10 = ('mean_price_per_sq_m', lambda x: np.quantile(x, [0.1])),
    value_pc90 = ('mean_price_per_sq_m', lambda x: np.quantile(x, [0.9]))
    )
    .rename(columns={'Postcode_short': 'coef'})
)
summ_coefs_long = (summ_coefs.groupby('Postcode', as_index=False).agg(
    value_mean = ('mean_price_per_sq_m', np.mean),
    value_pc10 = ('mean_price_per_sq_m', lambda x: np.quantile(x, [0.1])),
    value_pc90 = ('mean_price_per_sq_m', lambda x: np.quantile(x, [0.9]))
    )
    .rename(columns={'Postcode': 'coef'})
)
summ_coefs_comb = pd.concat([non_pc_coefs, summ_coefs_short, summ_coefs_long])
summ_coefs_comb.to_csv('static/calculator_coefs.csv', index=False)

#vii) Nearest short postcodes
short_postcodes = (ghouses
                   .assign(postcode_short = lambda df: df.Postcode.apply(lambda s: s.split(' ')[0]))
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
 .query('postcode in @summ_coefs_comb.coef')
 .to_csv('static/postcodeshort_nearest_five.csv', index=False)
)
#This is for the calculator so only include postcodes that got GLM coefs

#viii) And nearest short to long postcodes
(ghouses.drop_duplicates(subset=['Postcode']).assign(dum = 1)
 .merge(short_postcodes, on='dum', suffixes=['_a','_b'])
 .assign(distance_degrees = lambda df: df.apply(lambda row: ((row['Longitude_a']-row['Longitude_b'])**2 + (row['Latitude_a']-row['Latitude_b'])**2)**0.5, axis=1))
 .rename(columns={'postcode_short': 'postcode_short_b'})
 .assign(postcode_short_a = lambda df: df.Postcode.apply(lambda s: s.split(' ')[0]))
 .query('postcode_short_a != postcode_short_b')
 .sort_values('distance_degrees')
 .groupby('Postcode', as_index=False)
 .agg(nearest_pcs_1 = ('postcode_short_b', lambda p: p[0:1]),
      nearest_pcs_2 = ('postcode_short_b', lambda p: p[1:2]),
      nearest_pcs_3 = ('postcode_short_b', lambda p: p[2:3]),
      nearest_pcs_4 = ('postcode_short_b', lambda p: p[3:4]),
      nearest_pcs_5 = ('postcode_short_b', lambda p: p[4:5]))
 .rename(columns={'Postcode': 'postcode'})
 .query('postcode in @summ_coefs_comb.coef')
 .to_csv('static/postcodelongtoshort_nearest_five.csv', index=False)
)