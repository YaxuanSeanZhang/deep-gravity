# Deep-Gravity

## Overview

This project explored using deep learning models to predict human mobility flows between locations in an urban environment. We aimed to show deep learning can better capture interactions between locations vs. traditional methods like the gravity model.

The final report summarizing the project's findings is available in the file `CSCI_5527_Final Report.pdf`.

## Data
  * Mobility: Event-level mobile positioning dataset from PlaceIQ used to construct mobility o-d flows within the Twin Cities metro.
      July 1-8: Training
      July 9: Testing
  * Place characteristics: The count of various infrastructural elements such as restaurants, schools, hospitals, and bus stations.
  * Weather: Temperature and precipitation data for July 2021 from [Iowa Environmental Mesonet API](https://mesonet.agron.iastate.edu).
      Interpolated across fishnet grid (3km) with inverse distance weighting (IDW).

## Method
  * Gravity Model: Standard baseline where flows decrease with distance but increase with population.
  * Deep Learning Model: Based on Simini et al (2021). The model has multiple parallel neural network structures. For each origin-destination pair:
      1) We combine the place features(`data/fishnet_3km_poi_type_count.geojson`), weather data(`data/weather.csv`), and distance between the locations(`data/dist_matrix.csv`) into input vectors.
      2) These input vectors are fed into identical feed-forward neural networks.
      3) The output is a score between -infinity and infinity for that location pair. Higher scores mean a greater chance of a trip occurring.
      4) We convert these scores to probabilities with a softmax function. The final flow prediction is the probability multiplied by the total outflow from the origin location.
 
## Files
  * `data-processing/PlaceIQ_process.py`: defines functions for data processing, aggregation, and flow generation 
  * `data-processing/visit_to_network_fishnet.ipynb`: Script to aggregate PlaceIQ flows to fishnet grid level, conduct exploratory analysis and data scaling
  * `data-processing/Pull_Interpolation.ipynb` pulls raw weather data from API and interpolates across fishnet grid 
  * `data-processing/Demo_visit2traj.ipynb` constructs trajectories from PlaceIQ visit data
  * `model/gravity_model_metrics.ipynb`implements and evaluates baseline gravity model
  * `model/DeepGravity.py` defines deep learning model architecture
  * `model/utils.py`: Utility functions for data processing and modeling
  * `model/Optim.py`: Optimization functions for training deep learning model
  * `model/main.py`: Main script to execute deep learning model training and evaluation

## Output
To evaluate the overall model fit and predictive performance, the Common Part of Commuters (CPC), the Pearson correlation coefficient(r), the Root Mean Squared Error (RMSE), and the Jensen-Shannon divergence (JSD) are used. 

The deep gravity model outperformed the traditional gravity model. Incorporating weather data further improved the neural network's performance. This shows both urban features and environmental factors influence human movement.

## Author
  * Yaxuan Zhang
  * Zhongfu Ma
  * Xiaohuan Zeng
