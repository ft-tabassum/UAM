 **Demand-driven Vertiport Siting by Machine Learning and Agent-based Transport Simulation for UAM Network Expansion**

**Introduction:** Urban Air Mobility (UAM) is a novel concept that aims to minimize travel time and environmental impact while traveling to and from significant locations in and around metropolitan areas. The effective integration of Urban Air Mobility (UAM) is contingent upon the strategic selection of vertiport locations. The process of selecting suitable vertiport sites must account for a variety of factors, such as demand, land utilization, costs, and operational efficiency. The willingness of residents to utilize UAM services significantly affects the determination of vertiport locations within a region. As the UAM network expands, the demand and performance of existing networks will serve as critical criteria for the identification of new vertiport sites. Moreover, advancements in machine learning are enhancing the precision of demand forecasts for UAM services.

**Objective:** Develop a machine learning model to predict the demand for UAM services and determine the new vertiport locations in the uncovered areas.

**Data Source:** This project utilizes two primary data sources for UAM analysis in the Munich region:
1) Stated Preference Survey (Munich International Airport):
-A survey conducted in March 2023 collected 218 responses from Bavaria (67.9%) and Austria (17.7%) to analyze mode choice preferences and willingness to pay for UAM AirShuttle services to Munich International Airport (MUC).
-Includes travel behavior (e.g., 38.9% private car drivers, 39.4% public transport users), sociodemographic data (age, income, education), and attitudes toward automation and flight safety.
-Pricing data for ground modes derived from ADAC (2022) for private cars (€0.65/km) and Uber/FreeNow for ride-hailing (€1.8-3.3/km).
-Funded by the Air Mobility Initiative (Grant HAM-2109-0006).

2) MITO Synthetic Population Dataset (Extended Munich Metropolitan Region):
-The Microscopic Transportation Orchestrator (MITO) model, based on OpenStreetMap and SILO land use data, generates travel demand for the Extended Munich Metropolitan region.
-Calibrated with the German national household survey (Lenz et al., 2010) and validated using BASt traffic counts (2022).
-Home locations from the synthetic population serve as demand data points for weighted clustering, incorporating demographic attributes (e.g., income, car ownership) and spatial data processed via GIS tools.
