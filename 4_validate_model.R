# Compare calculator predictions to listed data for samples in January 23.

# It could be that areas scraped are generally too low, but particularly so for
#   Simon Brien; TR predictions are quite good.
# TODO: do another sample and get more accurate house areas from EPC?

library(dplyr)
library(ggplot2)
library(magrittr)

new_build_factor <- 1.4
#New build maybe increases raw ppsqm by 25%.
#Boof: https://news.twentyea.co.uk/blog/the-new-build-price-premium-how-much-more-will-we-pay-for-a-brand-new-property
#    They say 25% premium; 10-50% by region.
#1.5 is about right to match both SB and TR prices

coefs <- read.csv('static/calculator_coefs_smoothed_alpha3000.csv')
coefs_postcodes <- subset(coefs, grepl('BT',coef))
coefs <- subset(coefs, !grepl('BT', coef))

lps_vals_2024 <- read.csv('/media/shared_storage/data/ni-house-prices_data/other_data/processed_LPS_valuations.csv')

# BELOW ASSUMES THE OLD COEFS FORMAT

#HG 48 glenholm crescent scraped area 140 should be 100 incl garage; misprint on one room; REMOVED
#HG 18 balmoral lane scraped 63 should probably be ~80
#HG 29 pond park road scraped 91 should be 91+hall+2 bathrooms, though still will be very underestimated
#HG areas are usually missing rooms with no way to tell -> EXCLUDE

listings <- rbind(
    read.csv('listing_data/michaelchandler_listings_jan23.csv') %>% mutate(agent='MC'),
    read.csv('listing_data/simonbrien_listings_jan23.csv') %>% mutate(agent='SB'),
    read.csv('listing_data/templeton_listings_jan23.csv') %>% mutate(agent='TR')
) %>%
    mutate(is_apartment = (property_type %in% c('Apartment')),
           has_garage_01 = ifelse(has_garage=='True',1,0),
           has_garden_01 = ifelse(has_garden=='True',1,0),
           has_yard_01 = ifelse(has_yard=='True',1,0),
           has_other_outbuilding_01 = ifelse(has_other_outbuilding=='True',1,0),
           is_new_build = grepl('site', property_name) | nchar(postcode) <= 4) %>%
    filter(price < 400000)  #don't expect to do well on expensive ones
#add SB new builds - scraped once including and once excluding, so the difference are all new builds
#I got many more from SB that were new builds than not, because info is missing for many of the non-nbs;
#  limit to the first 100 
sb_new_builds <- read.csv('listing_data/simonbrien_listings_inclnewbuilds_jan23.csv') %>% mutate(agent='SB') %>%
    filter(!(property_name %in% listings$property_name)) %>%
    mutate(is_apartment = (property_type %in% c('Apartment')),
           has_garage_01 = ifelse(has_garage=='True',1,0),
           has_garden_01 = ifelse(has_garden=='True',1,0),
           has_yard_01 = ifelse(has_yard=='True',1,0),
           has_other_outbuilding_01 = ifelse(has_other_outbuilding=='True',1,0),
           is_new_build = TRUE) %>%
    filter(price < 400000) %>%
    head(100)
listings <- rbind(listings, sb_new_builds)


ggplot(listings %>% filter(!is_new_build)) + geom_point(aes(area, price, colour=agent), alpha=0.5) + 
    geom_smooth(aes(area, price, colour=agent), method='lm', se=FALSE, formula='y~x')

#Join and make predictions
pred_prices <- listings %>% 
    inner_join(coefs_postcodes %>% select(postcode=coef, 
                                          coef_mean=value_mean,
                                          coef_pc10=value_pc10,
                                          coef_pc90=value_pc90),
               by='postcode') %>%
    mutate(pred_ppsqm = ifelse(is_apartment,
                               coef_mean + subset(coefs, coef=='home_type_Flat')$value_mean,
                               coef_mean + subset(coefs, coef=='home_type_House')$value_mean + 
                                   has_garage_01*subset(coefs,coef=='has_garage_01')$value_mean +
                                   has_garden_01*subset(coefs,coef=='has_garden_01')$value_mean +
                                   has_yard_01*subset(coefs,coef=='has_yard_01')$value_mean +
                                   has_other_outbuilding_01*subset(coefs,coef=='has_other_outbuilding_01')$value_mean),
           pred_ppsqm_pc10 = ifelse(is_apartment,
                                    coef_pc10 + subset(coefs, coef=='home_type_Flat')$value_mean,
                                    coef_pc10 + subset(coefs, coef=='home_type_House')$value_mean + 
                                        has_garage_01*subset(coefs,coef=='has_garage_01')$value_mean +
                                        has_garden_01*subset(coefs,coef=='has_garden_01')$value_mean +
                                        has_yard_01*subset(coefs,coef=='has_yard_01')$value_mean +
                                        has_other_outbuilding_01*subset(coefs,coef=='has_other_outbuilding_01')$value_mean),
           pred_ppsqm_pc90 = ifelse(is_apartment,
                                    coef_pc90 + subset(coefs, coef=='home_type_Flat')$value_mean,
                                    coef_pc90 + subset(coefs, coef=='home_type_House')$value_mean + 
                                        has_garage_01*subset(coefs,coef=='has_garage_01')$value_mean +
                                        has_garden_01*subset(coefs,coef=='has_garden_01')$value_mean +
                                        has_yard_01*subset(coefs,coef=='has_yard_01')$value_mean +
                                        has_other_outbuilding_01*subset(coefs,coef=='has_other_outbuilding_01')$value_mean),
           pred_price = pred_ppsqm * area,
           pred_price_pc10 = pred_ppsqm_pc10 * area,
           pred_price_pc90 = pred_ppsqm_pc90 * area,
           # increase by 25% for presumed new builds
           pred_price = ifelse(is_new_build, pred_price*new_build_factor, pred_price),
           pred_price_pc10 = ifelse(is_new_build, pred_price_pc10*new_build_factor, pred_price_pc10),
           pred_price_pc90 = ifelse(is_new_build, pred_price_pc90*new_build_factor, pred_price_pc90),
           # for new builds with only short postcode, put central prediction at 70th percentile 
           #   (new long postcode is usually more desirable than average) (10% + 0.75*80% = 75%)
           pred_price = ifelse(is_new_build & nchar(postcode) <= 4, 
                               pred_price_pc10 + 0.75*(pred_price_pc90-pred_price_pc10), pred_price)
    )

#For the map, absolute values are not important; we just want to see that postcodes are ordered correctly
#  by ppsqm
pred_prices %>% filter(!is_new_build, nchar(postcode) > 4) %>% group_by(postcode) %>%
    summarise(n=n(), mean_ppsqm=sum(price)/sum(area), mean_pred_ppsqm=mean(pred_ppsqm), .groups='drop') %T>%
    {
        print(ggplot(.) + geom_point(aes(mean_pred_ppsqm, mean_ppsqm)) + 
                  geom_abline(linetype=2, colour='blue') + expand_limits(x=0, y=0))
    } %>%
    summarise(spearman = cor(mean_ppsqm, mean_pred_ppsqm, method='spearman'))
#Only one case of most postcodes; r=0.53


#Apartments - easier to compare
pred_prices %>% filter(is_apartment, !is_new_build) %T>%
    {
        print(
            ggplot(.) + geom_pointrange(aes(price/1000, y=pred_price/1000, 
                                            ymin=pred_price_pc10/1000, ymax=pred_price_pc90/1000, 
                                              colour=agent), size=0.3) +
                geom_abline(linetype=2) +
                scale_x_continuous(breaks=seq(0,1000,200), limits=c(0,400)) + 
                scale_y_continuous(breaks=seq(0,1000,200), limits=c(0,400)) +
                labs(y='Predicted price / £000', x='Listed price / £000',
                     title='Predicted prices for apartments',
                     subtitle='') +
                theme(legend.position=c(0.15,0.8))
            )
        } %>%
    group_by(agent) %>%
    summarise(n=n(),
              mean_pred_price = mean(pred_price), mean_listed_price = mean(price),
              frac_too_low = mean(pred_price < price))


#Houses with other terms - now well calibrated (non new builds; NBs are OK too)
pred_prices %>% filter(!is_apartment, !is_new_build) %T>%
    {
        print(
            ggplot(.) + geom_pointrange(aes(price/1000, y=pred_price/1000, ymin=pred_price_pc10/1000, ymax=pred_price_pc90/1000, 
                                            colour=agent), size=0.3) +
                geom_abline(linetype=2) +
                facet_grid(has_garden ~ has_garage, labeller=label_both) +
                labs(y='Predicted price / £000', x='Listed price / £000',
                     title='Predicted prices for houses') +
                scale_x_continuous(breaks=seq(0,1000,200), limits=c(0,500)) + 
                scale_y_continuous(breaks=seq(0,1000,200), limits=c(0,600)) +
                theme(legend.position=c(0.1,0.85))
            )
    } %>%
    group_by(agent, is_new_build) %>%
    summarise(n=n(), mean_pred_price = mean(pred_price), mean_listed_price = mean(price),
              frac_too_low = mean(pred_price < price), .groups='drop')
#Houses are well calibrated: mean pred 189 vs listed 198 for MC, 228 vs 223 SB, 185 vs 189 TR.

pred_prices %>% filter(is_new_build) %>% group_by(agent) %>% 
    summarise(n=n(), mean(pred_price), mean(price), frac_too_low = mean(pred_price < price), .groups='drop')
#Factor of 1.4 for new builds is a bit low; 1.5 is spot on, but 50% seems too high to put on website - 
#  may be partly that model predictions are too low for these props somehow, and the new build part doesn't 
#  really add as much as 50%.

#Are new builds really 50% more than equivalent non-nb?
merge(pred_prices %>% filter(!is_apartment, !is_new_build) %>% mutate(ppsqm = price/area) %>% 
          group_by(property_type) %>% summarise(n_reg=n(), mean_area_reg=mean(area), mean_ppsqm_reg=mean(ppsqm), .groups='drop'),
      pred_prices %>% filter(!is_apartment, is_new_build) %>% mutate(ppsqm = price/area) %>% 
          group_by(property_type) %>% summarise(n_nb=n(), mean_area_nb=mean(area), mean_ppsqm_nb=mean(ppsqm), .groups='drop'),
      by='property_type') %>% filter(n_reg >= 5 & n_nb >= 5)
#Detached +25%, semi +13%, +30%, but not accounting for postcode and garden etc differences
#Nbs often only have the short postcode, for which the prediction is the average for that short postcode,
#  but nbs will often create a new postcode that is more valuable than the short postcode average.
#  So now putting these pred_price at the 70pc point rather than mean - works about right
#Where we have the full postcode
pred_prices %>% filter(is_new_build) %>% group_by(agent, is_full_postcode=(nchar(postcode)>=5)) %>% 
    summarise(n=n(), mean_pred_price=mean(pred_price), mean_true_price=mean(price), 
              frac_too_low = mean(pred_price < price), .groups='drop')
#SB and TR with full postcode are a bit low when using a factor of 1.4


#Compare to LPS data:
#- 39-kimberley-street (terrace) scraped area 58, LPS 93; scrape looks correct -> LPS wrong?
#- 25-fernwood-street (terrace) scraped area 66, LPS 85; scrape just missed hall, def looks like 
#     not more than 75 including stairs+hall -> LPS wrong but close if include garden+yard
#- 19-mount-merrion-drive (semi detached) scraped 65, LPS 89; scrape has all rooms -> LPS must include outside
#- 28-the-close (bungalow) scraped 95, LPS 114; pictures show total is 6.9*15.1 = 104; scrape just misses storage; 
#     garden is much bigger so can't be included; LPS might be counting garage.
#- 33-woodcroft-park (detached) scraped 171, LPS 163; scrape missed bathroom - OK
#- 2a-rugby-avenue (End Terrace) scraped 97, LPS 108; scrape only misses hall, which at 3 floors could be 10 - OK
#- 20-sandymount-street (Terrace) scraped 74, LPS 109; picture shows total is 99 so scrape is bad
#     but it misses only bathroom and hall/stairs -> LPS a bit high, hall/stairs should add 20!
#- 2-baronscourt-drive (Detached) scraped 113, LPS 141; scrape missed hall and guest WC, and internal garage,
#     so should be 113+15+2+~8 = 138 -> LPS OK, scrape didn't list garage so will be wrong
#- 36-rosetta-road (Semi-Detached) scraped 94, LPS 129 -> scrape just misses hall, 3 floors so could be 20 ->
#     LPS looks high but scrape is low because of hall
#- 9a-cherryhill-gardens (Detached) scraped 81, LPS 88; scrape missed ensuite, store, landing - OK
#- 8-lisdoonan-road (bungalow) scraped 106, LPS 148; scrape misses hall,porch,cloakroom ~10 -> LPS including garage?
#- 11-sunwich-street (terrace) scraped 63, LPS 84; scrape misses hall ~10 -> LPS counting yard?
#- 31-upper-newtownards-road (detached) scraped 98, LPS 117; scrape just misses stairs -> LPS a bit high
#- 22-belmont-avenue (terrace) scraped 73, LPS 69 - OK
#- 179-lower-braniel-road (bungalow) scraped 71, LPS 93; scrape misses garage, maybe greenhouse? -> scrape is low
#- 6a-killaire-road (bungalow) scraped 164, LPS 264; scrape misses garage ~50! -> LPS seems high
#- 63-ballymoney-road (detached) scraped 167, LPS 214; scrape misses garage 38 -> OK if include garage

#-> LPS maybe includes garage (and shed/greenhouse?) area; some others are larger than explained

listings %>% filter(!is_apartment) %>%
    inner_join(coefs_postcodes %>% select(postcode=coef, coef_mean=value_mean), by='postcode') %>%
    mutate(pred_ppsqm = coef_mean + subset(coefs, coef=='home_type_House')$value_mean,
           pred_ppsqm = pred_ppsqm + ifelse(property_type %in% c('Terrace','End Terrace'), 80, 270),
           pred_price = pred_ppsqm * area) %>%
    group_by(property_type) %>% 
    summarise(n=n(), mean_area=mean(area), 
              mean_pred_price = mean(pred_price), mean_listed_price = mean(price),
              frac_too_low = mean(pred_price < price)) %>%
    arrange(-n)
#

#For website, give postcode scatter plot with r=0.53, and this:

tmp <- pred_prices %>% filter(nchar(postcode) > 4, !is_new_build) %>% group_by(postcode) %>%
    summarise(n=n(), mean_ppsqm=sum(price)/sum(area), mean_pred_ppsqm=mean(pred_ppsqm), .groups='drop')
ggplot(tmp) + geom_point(aes(mean_ppsqm, mean_pred_ppsqm)) + 
    geom_abline(linetype=2, colour='grey') + expand_limits(x=0, y=0) +
    annotate('text', x=1100, y=2900, label=sprintf('Spearman r = %.2f',
                                                  cor(tmp$mean_ppsqm, tmp$mean_pred_ppsqm, method='spearman'))) +
    scale_y_continuous(labels = function(x) sprintf('£%i', x), limits=c(500,NA)) +
    scale_x_continuous(labels = function(x) sprintf('£%i', x), limits=c(500,NA)) +
    labs(title='NI House Price Map model price validation by postcode',
         subtitle='Listed prices obtained from three agent websites in January 2023; excluding new builds',
         y='Predicted price per square metre', x='Listed price per square meter') +
    theme_minimal() +
    theme(plot.subtitle = element_text(size=8))
ggsave('static/nihousepricemap_validation_plot_1.png', width=6, height=4) 

pred_prices %>% 
    mutate(category = ifelse(is_new_build, 'New build house',
                             ifelse(is_apartment, 'Apartment', 'House')),
           agent_anon = ifelse(agent=='MC', 'Agent 1',
                               ifelse(agent=='SB', 'Agent 2', 'Agent 3'))) %>%
    ggplot() + geom_pointrange(aes(price/1000, y=pred_price/1000, 
                                ymin=pred_price_pc10/1000, ymax=pred_price_pc90/1000, 
                                colour=agent_anon), size=0.35, alpha=0.7) +
    geom_abline(linetype=2, color='grey') +
    facet_wrap(~category, nrow=2) +
    scale_x_continuous(breaks=seq(0,1000,200), limits=c(0,400), labels = function(x) sprintf('£%i', x)) + 
    scale_y_continuous(breaks=seq(0,1000,200), limits=c(0,560), labels = function(x) sprintf('£%i', x)) +
    scale_colour_brewer(palette='Set1') +
    labs(y='Predicted price (thousands)', x='Listed price (thousands)',
         title='NI House Price Map model price validation, January 2023',
         subtitle=paste0(sprintf('Listed prices obtained from three agent websites in January 2023\nFactor %.1f applied to new build predictions;',new_build_factor),
                         ' where only a short postcode is listed, the central prediction is placed at the 70th percentile for the short postcode and 10-90% range is shown'),
         colour=NULL) +
    theme_minimal() +
    theme(legend.position=c(0.77,0.22),
          plot.subtitle = element_text(size=8))
ggsave('static/nihousepricemap_validation_plot_2.png', width=8, height=5)


# 1/6/24
# NOW USING NEW MODEL INCLUDING AREA TERMS

# I know that scaling LPS 2005 prices to 2024 x1.62 is correct - this covers non-new-builds

# Now check that the calculator reproduces the value_2022s without bias
lps_vals_2024 %>%
    inner_join(coefs_postcodes %>% select(Postcode=coef, postcode_coef=value_mean),
               by='Postcode') %>% 
    mutate(area_squared = area_m2**2,
           area_lt_60_01 = ifelse(area_m2 < 60, 1, 0)) %>%
    mutate(pred_value = ifelse(home_type=='Flat',
                               subset(coefs, coef=='home_type[Flat]')$value_mean +
                                   subset(coefs, coef=='home_type[Flat]:area_m2')$value_mean * area_m2 +
                                   subset(coefs, coef=='home_type[Flat]:area_squared')$value_mean * area_squared +
                                   subset(coefs, coef=='home_type[Flat]:area_lt_60_01')$value_mean * area_lt_60_01,
                               subset(coefs, coef=='home_type[House]')$value_mean +
                                   subset(coefs, coef=='home_type[House]:area_m2')$value_mean * area_m2 +
                                   subset(coefs, coef=='home_type[House]:area_squared')$value_mean * area_squared +
                                   subset(coefs, coef=='home_type[House]:area_lt_60_01')$value_mean * area_lt_60_01)
           + subset(coefs, coef=='has_garage_01')$value_mean * has_garage_01 +
               subset(coefs, coef=='has_garden_01')$value_mean * has_garden_01 +
               subset(coefs, coef=='has_other_outbuilding_01')$value_mean * has_other_outbuilding_01 +
               subset(coefs, coef=='has_yard_01')$value_mean * has_yard_01 +
               postcode_coef) %>%
    mutate(area_quintile = ntile(area_m2, 5)) %>% 
    group_by(area_quintile) %>% summarise(n=n(),
                                          mean_area_m2 = mean(area_m2),
                                          max_area_m2 = max(area_m2),
                                          mean_actual=mean(price_per_sq_m_2022_by_lgd),
                                          mean_pred=mean(pred_value),
                                          .groups='drop') %>% 
    mutate(residual = mean_actual - mean_pred,
           residual_value = residual*mean_area_m2)

# Underpredicts for low area (< ~100 m2) and overpredicts for high area (>150)
# This is expected, as value doesn't drop to zero as area goes towards 0, and value must saturate at some large area.
# Could correct for this, though it's only a £4500 underprediction for small areas.
# -> corrected model with area, area^2 and area_lt_60 terms, interacting with home_type;
#    now all quintiles are slightly (~1%) under but the pattern is mostly gone
#    Houses are slightly low, flats slightly high.


# Look at linearity vs area - might be compromising the 100-300k range and getting higher values good?
lps_vals_2024 %>% group_by(area_grp = ntile(area_m2, 20)) %>% mutate(mean_area = mean(area_m2)) %>% ungroup() %>%
    ggplot() + geom_boxplot(aes(mean_area, value_2022_by_lgd, group=mean_area),
                            outlier.size=0.3, outlier.alpha=0.5) +
    geom_smooth(aes(mean_area, value_2022_by_lgd), se=FALSE, method='lm', colour='coral') +
    facet_wrap(~home_type) +
    ylim(0, 8e5) +
    theme_light()
# Value is quite linear with area but is a bit sub-linear at the high end,
#   even though in this view, the biggest area bins probably also contain the best postcodes,
#   so we would expect higher than linear values.

lps_vals_2024 %>% 
    filter(area_m2 <= 200) %>% 
    inner_join(coefs_postcodes %>% select(Postcode=coef, postcode_coef=value_mean),
               by='Postcode') %>% 
    group_by(area_grp = ntile(area_m2, 20)) %>% mutate(mean_area = mean(area_m2)) %>% ungroup() %>%
    mutate(price_minus_postcode = price_per_sq_m_2022_by_lgd - postcode_coef) %>%
    ggplot() + geom_boxplot(aes(mean_area, price_minus_postcode, group=mean_area),
                            outlier.size=0.3, outlier.alpha=0.5) +
    geom_smooth(aes(mean_area, price_minus_postcode), se=FALSE, method='lm', colour='coral') +
    facet_wrap(~home_type) +
    ylim(0, 3500) +
    theme_light()
# Remove the postcode component (may introduce some distortion as coefs were linear fits) and look at ppsqm.
# Sub-linear at high area is clearer now
# The range in ppsqm is ~1520 to 1650, so by using this target without area as a feature,
#   predictions for small area values may be ~60ppsqm*100m2 = £6k low, and high areas ~60*250m2 = £15k high.
# Now appears in both Houses and Flats but with different slopes
# We see
# - Particularly above linear for houses at 50-60m2; i.e. a 50m2 house is worth more than half a 100m2 house (maybe 1.2x half)
# - There is a big sublinear trend for flats 40-100m2; i.e. a 100m2 flat is worth much less than twice a 50m2 flat
#     (maybe 80% of twice, i.e. 1.6x); sublinear trend continues at higher areas but flattens.
# So will need a model with interacting home_type and area terms, and extra term for <=60, and maybe <=100 flats.
