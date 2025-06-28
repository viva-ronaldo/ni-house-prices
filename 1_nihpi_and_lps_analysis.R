#NIHPI reports do not match ODNI data. Standardised prices are different, including relative
#  value by LGD; percent changes between quarters don't match, e.g. several LGDs show drop in Q3 2022
#  but none here do; looks like these values are more tied to the nw index somehow;
#  note that the HPI and Standardised_Price fields in lgd are just scaled forms of the same.

#The prices in the reports are just the average of sales of each property
#  type in the quarter; ODNI is probably smoothed.

#These are prices rather than index but relative movement should be the same
# nw_from_reports <- data.frame(
#     'Quarter_Year' = c('Q2 2021','Q3 2021','Q4 2021','Q1 2022','Q2 2022','Q3 2022'),
#     'NI_Detached_Property_Price_Index' = c(278235,286072,291288,286344,291071,293422),
#     'NI_SemiDetached_Property_Price_Index' = c(176690,180641,176018,185143,187861,187613),
#     'NI_Terrace_Property_Price_Index' = c(129942,130579,128496,136241,138942,140231),
#     'NI_Apartment_Price_Index' = c(143578,140021,144869,145990,147594,149977),
#     'NI_Residential_Property_Price_Index' = c(195242,198821,198890,202325,205628,207247),
#     stringsAsFactors = FALSE
# ) %>% mutate(Quarter_Year = factor(Quarter_Year, levels = Quarter_Year))
# ggplot(nw_from_reports %>% pivot_longer(cols=ends_with('Index'))) + geom_path(aes(Quarter_Year, value, colour=name, group=name))

library(dplyr)
library(ggplot2)
library(lubridate)
library(tidyr)
library(ggrepel)
library(cowplot)

#i) Relative moves of LGD prices ----

lgd_wide <- read.csv('/media/shared_storage/data/ni-house-prices_data/other_data/standardised-price-and-index-by-lgd-q1-2005---q4-2024.csv')
lgd_by_type_wide <- read.csv('/media/shared_storage/data/ni-house-prices_data/other_data/ni-hpi-by-property-type-q1-2005---q4-2024.csv')

# rel_value is the value of an LGD relative to the mean of all LGDs (should be size weighted?)
lgd <- lgd_wide %>% select(Quarter_Year, ends_with('Price')) %>% 
  pivot_longer(cols=ends_with('Price')) %>% 
  mutate(name=sub('_Standardised_Price','',name), yq_dt = parse_date_time(substr(Quarter_Year,2,7), '%q %Y')) %>% 
  group_by(Quarter_Year) %>% mutate(rel_value=value/mean(value)) %>% ungroup() 

# Index starting points seem to be meaningless between types; convert these to starting points of 100 each before pivoting
lgd_by_type <- lgd_by_type_wide %>% 
    mutate(NI_Apartment_Price_Index = NI_Apartment_Price_Index / first(NI_Apartment_Price_Index),
           NI_Residential_Property_Price_Index = NI_Residential_Property_Price_Index / first(NI_Residential_Property_Price_Index)) %>%
    # Apartments make up 11% of the market (from multiple quarterly reports) so get the houses component as 8/9 of the total
    mutate(NI_Houses_Price_Index = (9*NI_Residential_Property_Price_Index - NI_Apartment_Price_Index)/8) %>%
    select(Quarter_Year, NI_Apartment_Price_Index, NI_Houses_Price_Index) %>% 
    pivot_longer(cols=ends_with('Price_Index')) %>% 
    mutate(name=sub('_Price_Index','',name), yq_dt = parse_date_time(substr(Quarter_Year,2,7), '%q %Y'))


# Do for houses and apartments in absolute terms first, then show LGD relative changes
lgd_by_type %>%
    ggplot() + geom_path(aes(yq_dt, value, colour=name, group=name))
#

#ggplot(lgd) + geom_path(aes(yq_dt, rel_value, colour=name, group=name))

#or divide by each LGD's starting value to see relative moves more clearly
inner_join(lgd,
           lgd %>% filter(Quarter_Year=='Q1 2005') %>% select(name, orig_value=rel_value), by=c('name')) %>% 
    ggplot() + geom_path(aes(yq_dt, rel_value/orig_value, colour=name, group=name))
#Mid Ulster 0.93, Fermanagh 0.96, Mid East Antrim 0.96 are down; Causeway Coast 1.05, Derry 1.05, Newry 1.06 are up

#Trace paths
# inner_join(lgd,
#            lgd %>% filter(Quarter_Year=='Q1 2005') %>% select(name, orig_value=value), by=c('name')) %>% 
#     arrange(yq_dt) %>% 
#     ggplot() + geom_path(aes(value, rel_value, colour=name, group=name), alpha=0.5) + 
#     geom_point(aes(ifelse(Quarter_Year %in% c('Q1 2005','Q4 2024'), value, NA), rel_value, colour=name), size=5)
#or maybe cleaner to do start to end point movement plot only
inner_join(lgd %>% filter(Quarter_Year=='Q1 2005') %>% select(name, value_1=value),
           lgd %>% filter(Quarter_Year=='Q4 2024') %>% select(name, value_2=value), by=c('name')) %>% 
    mutate(value_1 = value_1/mean(value_1), value_2 = value_2/mean(value_2),
           rank_value_2 = rank(value_2)) %>%
    pivot_longer(cols=c('value_1','value_2'), names_to='value_date') %>%
    mutate(name = gsub('_',' ',name),
           name_1_with_NA = ifelse(value_date=='value_1' & rank_value_2 %% 2 == 0, name, NA),
           name_2_with_NA = ifelse(value_date=='value_2' & rank_value_2 %% 2 == 1, name, NA)) %>%
    ggplot() + geom_path(aes(value_date, value, group=name, colour=name)) +
    geom_point(aes(value_date, value, colour=name), size=3) +
    geom_text_repel(aes(value_date, value, label=name_1_with_NA, colour=name), hjust=1.1, size=3, direction='y') +
    geom_text_repel(aes(value_date, value, label=name_2_with_NA, colour=name), hjust=-0.1, size=3, direction='y') +
    scale_x_discrete(labels=c('Q1 2005', 'Q4 2024'), expand=c(0.6,0.6)) +
    ylim(0.8,1.2) +
    labs(y='Value relative to nationwide average', x=NULL) +
    guides(colour='none') +
    theme_minimal()
# Some moves of a few % relative to average; AND/LC swap at top two; CCG, NMD get closer to top 2 in 3rd,4th;
#   Mid Ulster drops 5% and 2 places; DCS rises 6% from last to ~tied second last.

# Facet by LGD - best view; maybe shouldn't use the rel value as the mean by LGD would need to be size weighted
#   but can get away with it..
p2 <- inner_join(lgd,
           lgd %>% filter(Quarter_Year=='Q1 2005') %>% select(name, orig_value=rel_value), by=c('name')) %>% 
    mutate(name = gsub('_', ' ', name),
           name = gsub('and ', 'and\n', name),
           name = ifelse(name=='Armagh City Banbridge and\nCraigavon', 'Armagh City,\nBanbridge and Craigavon', name),
           name = ifelse(name=='Newry Mourne and\nDown', 'Newry, Mourne\nand Down', name)) %>%
    ggplot() + geom_smooth(aes(yq_dt, rel_value, group=name),colour='black',size=0.5,
                           se=FALSE, method='loess', formula='y~x', span=0.15) +
    facet_wrap(~name) +
    scale_x_continuous(breaks=parse_date_time(c('2010-01','2020-01'), '%Y-%m'), labels=c('2010','2020')) +
    labs(x=NULL, y='Value / mean(all LGD values)') +
    theme_light() +
    theme(strip.text = element_text(size=5), axis.text = element_text(size=7),
          panel.grid.minor = element_blank(), panel.grid.major.x=element_blank())
p2
#
p1 <- lgd_by_type %>%
    ggplot() + geom_path(aes(yq_dt, value, colour=name, group=name)) +
    labs(x=NULL, y='Value index (2005 = 1)') +
    scale_x_continuous(breaks=parse_date_time(c('2005-01','2010-01','2015-01','2020-01','2025-01'), '%Y-%m'), labels=c('2005','2010','2015','2020','2025')) +
    scale_y_continuous(breaks=seq(0.8,2.0,0.20)) +
    scale_colour_manual(values=c('NI_Apartment'='maroon', 'NI_Houses'='dodgerblue')) +
    annotate('label', x=parse_date_time('2022-01','%Y-%m'), y=1.12, label='Apartments', colour='maroon', size=2) +
    annotate('label', x=parse_date_time('2022-01','%Y-%m'), y=1.50, label='Houses', colour='dodgerblue', size=2) +
    guides(colour='none') +
    theme_light() + 
    theme(legend.position=c(0.6,0.8), axis.text = element_text(size=8),
          panel.grid.minor = element_blank(), panel.grid.major.x=element_blank())
#plot_grid(p1, p2, labels = c("A", "B"), scale=c(0.7,1))
plot_grid(
    plot_grid(NULL, p1, NULL, nrow=3, labels=NULL, rel_heights = c(0.2, 0.6, 0.2)),
    #p2, 
    plot_grid(NULL, p2, nrow=2, labels=NULL, rel_heights = c(0.05, 0.95)),
    labels = c("Countrywide NIHPI", "LGD prices relative to countrywide average"),
    vjust=1.5, hjust=c(-0.15, +0.0), label_fontfamily='sans', label_size=12,
    ncol=2, rel_widths=c(0.4,0.6)
)
# ** PLOT **
#ggsave('pictures/nihpi_to2024q1_series_2panel_housevsapt_lgdsrelative.png', width=7, height=4)
ggsave('pictures/nihpi_to2024q4_series_2panel_housevsapt_lgdsrelative.png', width=7, height=4)

#ii) Changes in LGD housing stock ----
stock <- read.csv('/media/shared_storage/data/ni-house-prices_data/other_data/ni-housing-stock-by-lgd-2008---2022.csv', check.names = FALSE) %>%
    filter(`District Council`!='') %>% 
    pivot_longer(cols=ends_with('Stock'), names_to='year')
inner_join(stock %>% filter(year=='2008 Housing Stock') %>% select(`District Council`, value_1=value),
           stock %>% filter(year=='2022 Housing Stock') %>% select(`District Council`, value_2=value), by=c('District Council')) %>% 
    mutate(value_1 = value_1/mean(value_1), value_2 = value_2/mean(value_2),
           rank_value_2 = rank(value_2)) %>%
    pivot_longer(cols=c('value_1','value_2'), names_to='value_date') %>%
    rename(name=`District Council`) %>%
    mutate(name_1_with_NA = ifelse(value_date=='value_1' & rank_value_2 %% 2 == 0, name, NA),
           name_2_with_NA = ifelse(value_date=='value_2' & rank_value_2 %% 2 == 1, name, NA)) %>%
    ggplot() + geom_path(aes(value_date, value, group=name, colour=name)) +
    geom_point(aes(value_date, value, colour=name), size=3) +
    geom_text_repel(aes(value_date, value, label=name_1_with_NA), hjust=1.1, size=3, direction='y') +
    geom_text_repel(aes(value_date, value, label=name_2_with_NA), hjust=-0.1, size=3, direction='y') +
    scale_x_discrete(labels=c('Q1 2005', 'Q3 2022'), expand=c(0.6,0.6)) +
    #ylim(0.8,1.2) +
    labs(y='Value relative to nationwide average', x=NULL) +
    guides(colour='none')
#Belfast skews plot, and sizes of regions compared to one another don't really matter
#Just do table of growth since 2005
inner_join(stock %>% filter(year=='2008 Housing Stock') %>% select(`District Council`, value_1=value),
           stock %>% filter(year=='2022 Housing Stock') %>% select(`District Council`, value_2=value), by=c('District Council')) %>%
    mutate(growth_since_2005_pct = (value_2-value_1)*100/value_1) %>% select(`District Council`, growth_since_2005_pct) %>%
    arrange(-growth_since_2005_pct)
#Lisburn +20%, Newry, Armagh, Mid Ulster +16%
#Belfast +8%, Causeway Coast, Mid East Antrim +11%
#No obvious link to price changes


#iii) LPS valuation analysis ----

lps_valuation_files <- Sys.glob('/media/shared_storage/data/ni-house-prices_data/LPS_data/lps_valuations*csv')

lps_sample <- lapply(lps_valuation_files %>% sample(200, replace=FALSE), read.csv) %>% bind_rows() %>%
    filter(area_m2 < 1000)
dim(lps_sample)

ggplot(lps_sample) + geom_smooth(aes(area_m2, value))
#Value is linear with area up to 500, which covers 99.8%; 98% are <=300
ggplot(subset(lps_sample, area_m2<=300)) + geom_smooth(aes(area_m2, value))
#
lps_sample %>% filter(area_m2 <= 200, area_m2 > 10) %>% mutate(area_decile = cut(area_m2, 10)) %>% 
    group_by(area_decile) %>% mutate(mean_area = mean(area_m2), mean_value = mean(value)) %>% ungroup() %>%
    ggplot() + geom_boxplot(aes(mean_area, value, group=area_decile), width=15) +
    geom_smooth(aes(area_m2, value), se=TRUE, method='gam') +
    ylim(0,500000)
#Doesn't look like it goes sub-linear when area increases, as might expect (next 50m2 above 100m2 might
#  be worth less than the first 50m2).

lps_sample %>% filter(area_m2 <= 200, area_m2 > 10) %>%
    mutate(has_garage = grepl('outbuilding', desc)) %>% 
    ggplot() + geom_smooth(aes(area_m2, value, colour=has_garage, group=has_garage))
#
lps_sample %>% filter(area_m2 <= 200, area_m2 > 10, grepl('house', desc)) %>%
    group_by(cut(area_m2, 10)) %>% summarise(frac_garage = mean(grepl('outbuilding', desc)),
                                             frac_garden = mean(grepl('garden', desc)),
                                             frac_yard = mean(grepl('yard', desc)), .groups='drop')
#Garages are fairly uncommon for area <70m2; gardens are 70-90% for everything >=35m2 (surprising, some terrace may count);
#  yards are uncommon but most likely at 50-90m2
lps_sample %>% filter(area_m2 <= 200, area_m2 > 10) %>%
    mutate(home_type = ifelse(grepl('flat|apartment', desc), 'flat', 'house')) %>% 
    ggplot() + geom_smooth(aes(area_m2, value, colour=home_type, group=home_type))
#Flat areas are mostly 30-100m2; possible change in price gradient at ~60m2, becoming steeper,
#  i.e. increase in area from 30 to 60 is not worth as much as 60 to 100; probably separates
#  council flats from private ones.

#TODO
#- flat floor number effect
#- use has outbuilding/garden to determine non-terrace?

