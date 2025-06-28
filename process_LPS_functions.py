import copy, gc, psutil, random, re, requests
import numpy as np
import pandas as pd
from statsmodels.api import GLM, add_constant
import statsmodels.formula.api as smf
from matplotlib import colors, cm
from collections import namedtuple
from itertools import chain

def read_nihpi_api_nationwide():
    """ Read the latest NIHPI series at nationwide level from the ODNI API """
    res = requests.get('https://admin.opendatani.gov.uk/api/3/action/datastore_search?resource_id=b47a047d-5ed1-41ea-af13-2954b4a2bd92')
    nihpi_nw = pd.DataFrame(res.json()['result']['records'])
    # Check sorted - pretend 'Qx' are month numbers to aid parsing
    nihpi_nw = nihpi_nw.sort_values('Quarter_Year', key=lambda ds: pd.to_datetime(ds, format='Q%m %Y'))
    price_changes_nw = (
        nihpi_nw
        .set_index('Quarter_Year')
        .filter(['NI_Detached_Property_Price_Index',
            'NI_SemiDetached_Property_Price_Index',
            'NI_Terrace_Property_Price_Index',
            'NI_Apartment_Price_Index',
            'NI_Residential_Property_Price_Index'])
        .astype(float)
        )
    return price_changes_nw

def read_nihpi_api_lgd():
    """ Read the latest NIHPI series at LGD level from the ODNI API """
    res = requests.get('https://admin.opendatani.gov.uk/api/3/action/datastore_search?resource_id=dc7af407-bcb5-4820-81c0-a5e0dd7cbcb9')
    nihpi_lgd = pd.DataFrame(res.json()['result']['records'])
    # Check sorted - pretend 'Qx' are month numbers to aid parsing
    nihpi_lgd = nihpi_lgd.sort_values('Quarter_Year', key=lambda ds: pd.to_datetime(ds, format='Q%m %Y'))
    price_changes_lgd = (
        nihpi_lgd
        .set_index('Quarter_Year')
        .filter([c for c in nihpi_lgd.columns if '_HPI' in c])
        .astype(float)
        )
    return price_changes_lgd


def load_and_filter_LPS_houses_data(lps_files,
    postcodes_file,
    min_property_area=2,
    max_property_area=8000
    ):
    """ Load LPS property data, correct some bad records, and filter some bad or non-standard records """
    houses = pd.concat([pd.read_csv(f) for f in lps_files])
    print(f'Read {houses.shape[0]} rows')

    houses = houses[houses.postcode.notnull()]
    print(f'{houses.shape[0]} after dropping missing postcodes')

    houses['desc'] = houses.desc.fillna('')
    houses = houses[~houses.desc.str.contains('presbytery|convent|sheltered|shletered|mobile home|shop|premises|self-catering unit|'+
                                              'public house|showhouse|wardens house|community|unfinished|vacant|freestanding caravan|'+
                                              'gate lodge|gatelodge|church|wardens|vacany|room|parochial house|'+
                                              'mobile|single level self contained|caravan|carvan')].copy()
    houses = houses[~houses.desc.isin([
        '', 'outbuilding', 'hut', 'houses (huts) garden', 'hut garden', 'storage',
        'manse', 'cabin', 'garage', 'store', 'bed & breakfast', 'prefab', 'log cabin',
        ])]
    print(f'{houses.shape[0]} after dropping empty or non-standard descs')

    #Add lon, lat from Doogal postcodes 
    postcodes = pd.read_csv(postcodes_file, usecols=['Postcode', 'Latitude', 'Longitude', 'LSOA Code'])
    houses = houses.merge(postcodes, left_on='postcode', right_on='Postcode')

    #Some postcode lon lat are 0
    print(f"Dropping {(houses.Longitude == 0).sum()} lon=lat=0 cases of {houses.shape[0]} total")
    houses = houses[houses.Longitude != 0]

    print(f"Dropping {(houses.type == 'Mixed').sum()} type=Mixed cases")
    houses = houses[houses.type == 'Domestic']

    #19000 is used in some suspicious cases; may be in bad disrepair or some other situation
    print(f'Dropping {(houses.value == 19000).sum()} cases of £19,000')
    houses = houses[houses.value != 19000]

    #Drop very big ones which would stretch a linear ppsqm fit or be not normal houses
    print(f'Dropping {(houses.area_m2 > max_property_area).sum()} cases bigger than {max_property_area}m^2')
    houses = houses[houses.area_m2 <= max_property_area]

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


    houses['price_per_sq_m'] = houses.value / houses.area_m2

    print(f'''Dropping {len(houses[(houses.value <= 10000) & (houses.price_per_sq_m < 100)])} 
        cases with value a few thousand and ppsqm < £100/m^2''')
    houses = houses[~((houses.value <= 10000) & (houses.price_per_sq_m < 100))].copy()

    print(f'  and {houses[houses.price_per_sq_m <= 200].shape[0]} rows with ppsqm < £200/m^2')
    houses = houses[houses.price_per_sq_m > 200].copy()

    #A few more look dodgy in size and/or value, or may be a non-house
    bad_ids = [
        217637, 207728, 227086, 249727, 279559, 241819, 283490, 283752,
        244679, 166273, 290979, 164369, 177058, 145552, 147965, 195652,
        124847, 138233, 116652, 145536,  84213, 113324,  89525,  66628,
         64802,  85170, 200077,  63280
        ]
    print(f'Dropping {len(bad_ids)} ids found to be dodgy')
    houses = houses[~houses.index.isin([bad_ids])].copy()

    #Assign home type
    houses['home_type'] = houses.desc.apply(convert_desc_to_home_type)

    #Drop the few Others
    print(f"Dropping {(houses.home_type == 'Other').sum()} cases not identified as House or Flat")
    houses = houses[houses.home_type.isin(['House', 'Flat'])].copy()

    return houses


def convert_desc_to_home_type(desc_string):
    """ Convert LPS property description field 'desc' to home type 'House', 'Flat', or 'Other' """
    if 'house' in desc_string:
        return 'House'
    elif desc_string[:2] in ['h ','h(','hg','ho','h+','h&','h/','h.','h0','hy','h<','h,','hs','h[','h{']:
        return 'House'
    elif desc_string in ['bungalow', 'cottage', 'h', 'ouse outbuilding garden', 'shouse outbuildings garden',
                         'hag', 'hag o', "h's o g", 'hp & g', '1st & 2nd floor house', 'use outbuilding garden']:
        return 'House'
    elif re.search('maisonette|farmhouse|detached rural dwelling|chalet|' +
        'cottage|townhouse|bungalow|converted barn|converted outbuilding|' +
        'huose|hiuse|huse|nouse|hiouse|hpuse', desc_string):
        return 'House'
    elif re.search('flat|flt|falt|flaat|fla t|glat|flct|flst|bedsit|penthouse|' +
                   'apartment|appartment|apart|apt|aparment|app|a\\[artment|' +
                   'aparatment|apar|aoartment|aartment|aprtment|apatrtment|apratment|' +
                   'flast |flzt |flar |room |rm |rooms |fl 1st|flay |fla\\(1st\\)|fla gd|' +
                   'fat|flag|first \\(first floor\\)|lat \\(1st floor\\)|f;at \\(1st floor\\)', desc_string):
        return 'Flat'
    elif desc_string in ['<1st>', 'app', '1lat (1st)', '1f app', 'app(gf)']:
        return 'Flat'
    else:
        return 'Other'


def calculate_glm_coefs(df, formula_minus_postcodes, min_properties_per_postcode=10, normalise_pc_coefs=False):
    """
    Fit GLMs in batches by postcode to property data `df` using a formula of `formula_minus_postcodes` plus postcode terms.
    Returns two dataframes of coefficients, one for postcode terms and one for non-postcode terms
    """
    #Modelling all postcodes is too big to do in one go
    postcode_has_min_rows = (df.Postcode.value_counts() >= min_properties_per_postcode)
    postcode_cols = [f'postcode_{c}' for c in postcode_has_min_rows.index[np.where(postcode_has_min_rows)]]
    print(f'Using {len(postcode_cols)} postcode terms with >= {min_properties_per_postcode} rows')

    #Fit three times, in shuffled batches of 400, and average the coefficients; glm can't go much bigger than 400
    step = 400
    glm_res = []
    for e in [1,2,3]:
        #Need to jumble up the postcodes to avoid fitting only one area at once
        jumbled_postcode_cols = copy.copy(postcode_cols)
        random.shuffle(jumbled_postcode_cols)

        postcode_glms = []
        for i in range(0,len(postcode_cols),step):
            if i % 2000 == 0:
                print(f'iter {e}, {i} / {len(postcode_cols)}')
            postcode_cols_batch = jumbled_postcode_cols[i:min(i+step,len(postcode_cols))]
            postcodes_batch = [p.split('_')[1] for p in postcode_cols_batch]
            
            train_batch = df[df.Postcode.isin(postcodes_batch)]
            train_batch = pd.concat([train_batch, pd.get_dummies(train_batch['Postcode'], prefix='postcode', sparse=True)], axis=1)
            # for formula method
            train_batch = train_batch.rename(columns = {c: c.replace(' ', 'SPACE') for c in postcode_cols_batch})

            try:
                # New formula method to get interactions
                full_formula = formula_minus_postcodes + ' + ' + ' + '.join([c.replace(' ', 'SPACE') for c in postcode_cols_batch])

                train_batch = train_batch[[c for c in train_batch.columns if c in full_formula]]

                this_glm = smf.glm(data=train_batch, formula=full_formula).fit()

                postcode_glms.append(namedtuple('glm_slim', ['params', 'bse'])(this_glm.params, this_glm.bse))
            except np.linalg.LinAlgError:
                print('This batch GLM did not converge; skipping')
            
            del train_batch, this_glm
            gc.collect()

        glm_res.append(postcode_glms)

    pc_coefs = (
        pd.concat([g.params for g in glm_res[0]]).filter(like='postcode').to_frame(name='coef_1')
        .join(pd.concat([g.params for g in glm_res[1]]).filter(like='postcode').to_frame(name='coef_2'))
        .join(pd.concat([g.params for g in glm_res[2]]).filter(like='postcode').to_frame(name='coef_3'))
    )
    pc_coefs['pc_coef_avg'] = (pc_coefs.coef_1 + pc_coefs.coef_2 + pc_coefs.coef_3) / 3
    
    pc_coef_serrors = (
        pd.concat([g.bse for g in glm_res[0]]).filter(like='postcode').to_frame(name='coef_se_1')
        .join(pd.concat([g.bse for g in glm_res[1]]).filter(like='postcode').to_frame(name='coef_se_2'))
        .join(pd.concat([g.bse for g in glm_res[2]]).filter(like='postcode').to_frame(name='coef_se_3'))
    )
    pc_coef_serrors['pc_coef_se_avg'] = (pc_coef_serrors.coef_se_1 + pc_coef_serrors.coef_se_2 + pc_coef_serrors.coef_se_3) / 3
    pc_coefs = pc_coefs.join(pc_coef_serrors)

    pc_coefs['postcode'] = [s.split('_')[-1].replace('SPACE', ' ') for s in pc_coefs.index]
    pc_coefs = pc_coefs.reset_index()[['postcode', 'pc_coef_avg', 'pc_coef_se_avg']]

    if normalise_pc_coefs:
        # Normalise coefficients to zero for the map
        pc_coefs['pc_coef_avg'] -= pc_coefs.pc_coef_avg.mean()
        # SE sizes should still make sense

    #Save non-postcode coefs for calculator
    non_pc_coefs = (pd.concat([
            g.params.to_frame(name='value').join(g.bse.to_frame(name='se'))
            for g in chain.from_iterable(glm_res)])
        .filter(regex='const|area|garage|garden|yard|outbuilding|home_type', axis=0)
        .assign(coef = lambda df: df.index).reset_index(drop=True)
        .groupby('coef', as_index=False)
        .agg(
            value_mean = ('value', lambda v: np.round(np.mean(v),5)),
            se_mean = ('se', lambda v: np.round(np.mean(v),5))
            )
    )

    return pc_coefs, non_pc_coefs


def calculate_glm_coefs_by_LSOA(df, formula_minus_lsoas, min_properties_per_lsoa=50, normalise_lsoa_coefs=True):
    """
    Fit GLMs in LSOA batches to property data `df` using a formula of `formula_minus_lsoas` plus LSOA terms.
    Returns a dataframes of coefficients for the LSOA terms.
    """
    lsoa_has_50_rows = (df['LSOA Code'].value_counts() >= min_properties_per_lsoa)
    lsoa_cols = [f'lsoa_{c}' for c in lsoa_has_50_rows.index[np.where(lsoa_has_50_rows)]]
    print(f'Using {len(lsoa_cols)} LSOA terms with >= {min_properties_per_lsoa} rows')

    step = 150
    jumbled_lsoa_cols = copy.copy(lsoa_cols)
    random.shuffle(jumbled_lsoa_cols)
    all_lsoa_coefs = []
    for i in range(0, len(lsoa_cols), step):
        print(f'{i} / {len(lsoa_cols)}')
        lsoa_cols_batch = jumbled_lsoa_cols[i:min(i+step,len(lsoa_cols))]
        lsoa_batch = [p.split('_')[1] for p in lsoa_cols_batch]

        train_batch = df[df['LSOA Code'].isin(lsoa_batch)]
        train_batch = pd.concat([train_batch, pd.get_dummies(train_batch['LSOA Code'], prefix='lsoa', sparse=True)], axis=1)

        # New formula method
        full_formula = formula_minus_lsoas + ' + ' + ' + '.join(lsoa_cols_batch)

        train_batch = train_batch[[c for c in train_batch.columns if c in full_formula]]

        this_glm = smf.glm(data=train_batch, formula=full_formula).fit()

        lsoa_coefs = this_glm.params.filter(like='lsoa').to_frame(name='lsoa_coef')

        del train_batch, this_glm
        gc.collect()
        
        print(psutil.Process().memory_info().rss / (1024 * 1024))
    
        # Old method
        # #I think the excluded LSOA has a zero coefficient (if no intercept)
        # print(f'Setting {lsoa_cols_batch[-1]} coef to 0 before normalising')
        # lsoa_coefs.loc[lsoa_cols_batch[-1]] = 0.

        all_lsoa_coefs.append(lsoa_coefs)

    lsoa_coefs = pd.concat(all_lsoa_coefs).drop_duplicates()

    lsoa_coefs['LSOA Code'] = [s.split('_')[-1] for s in lsoa_coefs.index]
    lsoa_coefs = lsoa_coefs.reset_index()[['LSOA Code', 'lsoa_coef']]

    if normalise_lsoa_coefs:
        #These aren't necessarily centred on zero; re-centre here
        #  (OK to change their relationship with the non-lsoa coefs
        #   because this will only be used for the map, not the calculator)
        lsoa_coefs['lsoa_coef'] -= lsoa_coefs.lsoa_coef.mean()

    return lsoa_coefs


def convert_price_to_colour(this_ppsqm, all_ppsqm, cmap='BrBG'):
    """ Convert a price per square metre `this_ppsqm` to a colour by comparing to the full domain of prices `all_ppsqm` """
    breaks = np.nanpercentile(all_ppsqm, [i * 100 / 10 for i in range(0, 11)])
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
    cols = [fifteen_map(x) for x in [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]]  #list of length 10
    return colors.to_hex(cols[max(whichbin - 1, 0)])  #index the list with 0-9 inclusive

def run_one_smooth_pass(summ_coefs, alpha=1000):
    '''
    Smooth a set of postcode coefficients spatially.
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
            .assign(weight = lambda df: np.exp(-alpha*df.dist) * df.n_properties_y / df.n_properties_x,  #inv prop to distance and prop to number of properties, keeping weight=1 for the self row
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

def summarise_coefs(houses, coefs_df, soas, region_type='Postcode', suffix='', cmap='BrBG', alpha=None):
    """ Combine property df `houses` with coefficients `coefs_df` to produce map-ready format """
    if region_type == 'Postcode':
        group_keys = ['Postcode', 'LSOA Code']
        coefs_join_key_left = 'Postcode'
        coefs_join_key_right = 'postcode'
        coefs_ppsqm_name = 'pc_coef_avg'
        popup_region_column = 'Postcode'
    elif region_type == 'LSOA':
        group_keys = ['LSOA Code']
        coefs_join_key_left = 'LSOA Code'
        coefs_join_key_right = 'LSOA Code'
        coefs_ppsqm_name = 'lsoa_coef'
        popup_region_column = 'SOA_LABEL'

    summ_coefs = (houses.groupby(group_keys, as_index=False).agg(
        n_properties = ('value', len),
        longitude = ('Longitude', np.max), # there is only one value per postcode so any agg will do
        latitude = ('Latitude', np.max),
        mean_val = ('value_current_by_lgd', lambda v: round(np.mean(v))),
        mean_size = ('area_m2', lambda a: round(np.mean(a))),
        mean_price_per_sq_m = ('price_per_sq_m_current_by_lgd', lambda p: round(np.mean(p)))
        )
        .merge(coefs_df, left_on=coefs_join_key_left, right_on=coefs_join_key_right, how='inner')
        .rename(columns={coefs_ppsqm_name: 'ppsqm_delta'})
    )
    summ_coefs['ppsqm_delta'] = np.round(summ_coefs.ppsqm_delta, 1)

    #Do smoothing here; leave mean_price_per_sq_m as is but smooth ppsqm_delta
    if alpha is not None:
        for _ in [1, 2, 3]:
            summ_coefs = run_one_smooth_pass(summ_coefs, alpha=alpha)
    
    high_low_popup_colours = {
        'BrBG': ('#7f4909', '#015c53'), 
        'RdBu': ('#a11228', '#1b5a9b'), 
        'PRGn_r': ('#156c31', '#6a2076'),
        'PuOr': ('#a75106', '#4b1e7a')
    }[cmap]
    high_low_strings = (f"<span style='color: {high_low_popup_colours[1]}'><b>above</b></span>",
        f"<span style='color: {high_low_popup_colours[0]}'><b>below</b></span>")
    summ_coefs = (summ_coefs
        .merge(soas, left_on='LSOA Code', right_on='SOA_CODE', how='inner')
        .assign(popup_text = lambda df: df.apply(
            lambda row: f'<b>{row[popup_region_column]}</b><br/>' + \
                f'Mean value £{row.mean_val:,.0f}<br/>' + \
                f"<b>£{abs(row.ppsqm_delta):,.0f}</b> per sq. m {high_low_strings[0] if row.ppsqm_delta > 0 else high_low_strings[1]} average", axis=1))
        .assign(html_colour = lambda df: df.apply(lambda row: convert_price_to_colour(row.ppsqm_delta, summ_coefs.ppsqm_delta.values, cmap=cmap), axis=1))
        .query("n_properties >= 10")  # for the LSOA case; not needed for Postcode
    )

    col_list = [
        coefs_join_key_left,
        'longitude', 'latitude', 'n_properties',
        'mean_val', 'mean_size', 'mean_price_per_sq_m', 'ppsqm_delta',
        'html_colour', 'popup_text',
    ]
    if region_type == 'LSOA':
        col_list.append('geometry')

    return summ_coefs[col_list]

def save_colour_bands_file(summ_coefs_df, output_filename, unit_variable='Postcode'):
    """ Save a file of display colours and their bin's range of ppsqm_delta values """
    (summ_coefs_df.groupby('html_colour', as_index=False).agg(
        price_lower = ('ppsqm_delta', lambda x: int(np.min(x))),
        price_upper = ('ppsqm_delta', lambda x: int(np.max(x))),
        n_units = (unit_variable, len))
        .sort_values('price_lower')
        .to_csv(output_filename, index=False)
    )

def summ_values_by_postcode(summ_coefs_for_calculator, postcode_column):
    """ Aggregate a set of coefficients by a geography column `postcode_column`, e.g., short postcode """
    res = (summ_coefs_for_calculator.groupby(postcode_column, as_index=False).agg(
        value_mean = ('ppsqm_delta', lambda v: np.round(np.mean(v), 4)),
        value_lower = ('ppsqm_delta', lambda x: np.round(np.quantile(x, [0.05]), 4)),
        value_upper = ('ppsqm_delta', lambda x: np.round(np.quantile(x, [0.95]), 4))
        )
        .rename(columns={postcode_column: 'coef'})
    )
    return res

def combine_coefs_for_calculator(pc_coefs, non_pc_coefs, houses, soas, alpha=None):
    """
    Concatenate postcode and non-postcode coefficients for use in the calculator.
    Use summarise_coefs just to get the spatial smoothing on the pc_coef_avg; keep the standard errors from unsmoothed.
    """
    pc_coefs_calculator_smoothed = (
        summarise_coefs(houses, pc_coefs, soas, alpha=alpha)
        .filter(['Postcode', 'ppsqm_delta'])
        .merge(pc_coefs[['postcode', 'pc_coef_se_avg']].rename(columns={'postcode': 'Postcode'}),
            on='Postcode', how='left')
        )
    comb_coefs_calculator = pd.concat([
        (non_pc_coefs
            .assign(value_lower = lambda df: df.value_mean - 2*df.se_mean, 
                value_upper = lambda df: df.value_mean + 2*df.se_mean)),
        (pc_coefs_calculator_smoothed
            .assign(value_lower = lambda df: df.ppsqm_delta - 2*df.pc_coef_se_avg,
                value_upper = lambda df: df.ppsqm_delta + 2*df.pc_coef_se_avg)
            .rename(columns={'Postcode': 'coef', 'ppsqm_delta': 'value_mean'})),
        summ_values_by_postcode(
            pc_coefs_calculator_smoothed.assign(Postcode_short = lambda df: df.Postcode.apply(lambda p: p.split(' ')[0])),
            'Postcode_short')
        ])[['coef', 'value_mean', 'value_lower', 'value_upper']]
    return comb_coefs_calculator


def save_nearest_five_postcode_files(
    ghouses,
    comb_coefs_calculator_houses,
    comb_coefs_calculator_flats,
    postcodeshort_nearest_five_path,
    postcodelongtoshort_nearest_five_path):
    """ Find the nearest five postcodes geographically to each postcode, both short-to-short and long-to-short, and save as two files. """
    short_postcodes = (ghouses
                       .assign(postcode_short = lambda df: df.postcode.apply(lambda s: s.split(' ')[0]))
                       .groupby('postcode_short', as_index=False)[['Longitude', 'Latitude']].mean()
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
     .query('(postcode in @comb_coefs_calculator_houses.coef) | (postcode in @comb_coefs_calculator_flats.coef)')  #This is for the calculator so only include postcodes that got GLM coefs
     .to_csv(postcodeshort_nearest_five_path, index=False)
    )

    #Nearest short to long postcodes
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
     .query('(postcode in @comb_coefs_calculator_houses.coef) | (postcode in @comb_coefs_calculator_flats.coef)')
     .to_csv(postcodelongtoshort_nearest_five_path, index=False)
    )
