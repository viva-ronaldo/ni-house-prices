# NI House Prices

Modelling of Northern Ireland house prices using [Land and Property Services ](https://www.finance-ni.gov.uk/land-property-services-lps) (LPS) valuation data. The goal is to derive coefficients for postcodes and larger geographies, representing the value contributed by each area.

The results of this project are shown at [NI House Price Map](https://www.nihousepricemap.com/).

Contents:
- 1_nihpi_and_lps_analysis.R: brief analysis of [NI House Price Index](https://admin.opendatani.gov.uk/dataset/nihpi-by-propertytype) and LPS data
- 2_explore_and_model_LPS.ipynb: developing a model for property value and postcode coefficients, using LPS valuation data
- 3_process_LPS.py: process the LPS data following methods developed in 2.
- 4_validate_model.R: some validation on unseen estate agent listing data.
- index.html, robots.txt, static/: content for the website at nihousepricemap.com

