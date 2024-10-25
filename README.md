# Sea Ice Anomaly Detection and Prediction

This repository contains several Python scripts for analyzing sea ice anomalies, with specific training, testing, and validation periods.

# Download Sea Ice Extent Images

Sea ice extent images (648 MB for 2000 - 2022) can be downloaded from the following link: [NOAA Sea Ice Extent Images](https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/images/).

# Download Annual Minimum Antarctic Sea Ice Extent Data

Annual Minimum Antarctic Sea Ice Extent data can be found at the following link: [Understanding Climate: Antarctic Sea Ice Extent](https://www.climate.gov/news-features/understanding-climate/understanding-climate-antarctic-sea-ice-extent).


## Training, Testing, and Validation Periods

| **Training Period**         | **Testing Period** | **Validation Period** | **Description**                                                                                         | **Python File**              |
|-----------------------------|--------------------|------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------|
| 2000 - 2004, 2011 - 2022    | 2005 - 2009        | 2010                   | Trains the model on historical data, evaluating performance during the testing period from 2005 to 2009 and validating with data from 2010. | `_70_2000_2005.py`           |
| 2006 - 2022                 | 2000 - 2004        | 2005                   | Uses a broad training set from 2006 to 2022, testing against earlier data from 2000 to 2004, validating effectiveness on the anomalies of 2005.  | `_70_2005_2010.py`           |
| 2000 - 2010, 2017 - 2022    | 2011 - 2015        | 2016                   | Combines earlier and more recent training data, testing the model from 2011 to 2015, and validating against anomalies detected in 2016.          | `_70_2011_2016.py`           |
| 2000 - 2016                 | 2017 - 2021        | 2022                   | Trains on a comprehensive dataset up to 2016, assessing model performance with testing data from 2017 to 2021 and validating results against 2022 data. | `_70_2017_2022.py`           |

Each row represents a distinct approach to training and evaluating the model, allowing for thorough analysis of sea ice anomalies across various periods. 

# Results Analysis

All results and analyses from the model, along with plot generation, are presented in `Image_processing_result.ipynb`.



## Usage

To utilize the scripts, clone the repository and run the appropriate Python files corresponding to your chosen training and testing periods. Ensure you have the necessary libraries installed which is mentioned in.py file.
