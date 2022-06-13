# RANDOM FORESTS APPLICATION IN PUBLIC TRANSPORTATION DATABASE

### version 2 - mainly updated graphic generation and data mining process (twice as fast now with glob), some formatting fixes on the other files

## Quick Summary
This project consists in the application of a myriad of random forests regressor models, varying in the hyperparameters of 'max_depth' and 'n_estimators', with the objective to analyze its capacity to predict the number of passengers per bus line per day in a certain time of the year, with a certain set of weather conditions.

This is a undergraduate thesis to get a bachelor's degree in Civil Engineering on the Federal University of Rio de Janeiro (Brasil).

The public transportation dataset used for the project is from Belo Horizonte city in the state of Minas Gerais, the 6º biggest brazillian city in population, from 2016 to 2021, unfortunaly the data is not longer widely available on their website. 

The weather dataset is from INMET (https://portal.inmet.gov.br/dadoshistoricos).

The graduate underthesis this project is based on will be available soon (in PT-BR).

## Disclaimer
Following the removal of the public transportation dataset, a few adjustments had to be made, as there was no way to web scrap anymore. So to avoid uploading >5GB of spreadsheets, a few steps were skipped, thats why the first part of the code in `main.py` is between quotation marks. The objective of that code was to go through every passenger spreadsheet and compile the data, then groupby the trips by date and bus line, getting the quantity of passengers in each bus line, each day of the period between 2016 and 2021. As the code is between quotation marks, this step was skipped, and the final spreadsheet with all the trips grouped by day and bus line with the total of passengers is actually on `final_passenger_data.csv` already.

## Structure
* `main.py` -> code should run through here, calls all functions
* `data_mining.py` -> compiling data
* `preprocessing.py` -> formatting data
* `random_forests.py` -> machine learning 
* `graphics.py` -> results in graphics (used in thesis)
* `final_passenger_data.csv` -> passenger data, read disclaimer above
* `mgXX.csv` -> weather data from each year between 2016 and 2021
## Approach
* Reading passenger spreadsheets, getting the data from: date, ticket gate in arrival and departure, bus line.
* Getting the number of `passengers` per trip by subtracting ticket gate number at departure from ticket gate number at arrival.
* Concatenating all data in a dataframe, while creating new columns such as `day`, `month`, `year` (division of previous date column); `day of the week`, `week of the year` (taken from previous date column); `holiday`, `pre holiday`, `after holiday` (1 if is, 0 if its not); `pandemic` (1 if its 15th of march 2020 onwards, 0 if it is backwards from this date).
* Reading weather spreadsheets, getting the data from: date, hour of measurement (each line is one hour of distance from the one before and after), rain at the period and temperature at the period.
* Group measurements by day, getting the sum of `rain` in that day, and `mean`, `min` and `max` temperatures in the day.
* Clean anomalous values from all datasets.
* Group everything in one dataframe. Separate `passengers` column.
* Use `X` (`features`) as every columns except `passengers` (`Y`).
* Run everything through 25 Random Forests models, varying `n_estimators` using the following values:(2,10,20,50,100) and `max_depth`:(5,10,20,50,100).
* Scatter plot real values vs predictions, of every iteration of the division between train and test made by the growing-window forward-validation.
* Plot separately the results per iterations of the model with `max_depth` = 100 and `n_estimators` = 100.
* Plot the time elapsed in each random forest model.
* Boxplot all the random forest models next to each other.
* Repeat the approach using the dataset with only the pandemic years (2020, 2021) and then with the years without pandemic (2016-2019). To observe a few characteristics of a time series.
## Results
### General look at passenger data
![General look on passenger data](https://i.imgur.com/6Z2wMRC.png)

### Passenger data cleaned and grouped by day
![Passenger data cleaned and grouped by day](https://i.imgur.com/vtJll2Z.png)

### General look at temperature data
![General look on temperature data](https://i.imgur.com/mrWBwxm.png)

### Temperature data cleaned
![Temperature data cleaned](https://i.imgur.com/1SHoCr2.png)

### General look at rain data
![General look on rain data](https://i.imgur.com/0N0h1X8.png)

### Rain data cleaned
![Rain data cleaned](https://i.imgur.com/OHUb8Sz.png)
### R² Boxplot of all random forest models - full dataset
![R² Boxplot of all random forest models - full dataset](https://i.imgur.com/62Z6H79.png)

### R² Boxplot of all random forest models - pre pandemic dataset
![R² Boxplot of all random forest models - pre pandemic dataset](https://i.imgur.com/ipt55ra.png)

### R² Boxplot of all random forest models - pandemic dataset
![R² Boxplot of all random forest models - pandemic dataset](https://i.imgur.com/L8ghGEP.png)

### MAE Boxplot of all random forest models - full dataset
![MAE Boxplot of all random forest models - full dataset](https://i.imgur.com/G9FfYzX.png)

### MAE Boxplot of all random forest models - pre pandemic dataset
![MAE Boxplot of all random forest models - pre pandemic dataset](https://i.imgur.com/1PMUfV8.png)

### MAE Boxplot of all random forest models - pandemic dataset
![MAE Boxplot of all random forest models - pandemic dataset](https://i.imgur.com/esP7LZB.png)

### Time elapsed per random forest model - full dataset
![Time elapsed per random forest model - full dataset](https://i.imgur.com/AlFOY1o.png)

### Time elapsed per random forest model - pre pandemic dataset
![Time elapsed per random forest model - pre pandemic dataset](https://i.imgur.com/KSFJ8U6.png)

### Time elapsed per random forest model - pandemic dataset
![Time elapsed per random forest model - pandemic dataset](https://i.imgur.com/HrPUQhL.png)

### Models with max_depth set to 20 onwards seem to overfitted a bit, while the models with max_depth = 10 seem to underfitted the data, see the next pics for a in depth look. Pandemic data makes the model very erratic, some changes could be made to change the big variations on the data.
### R² and MAE values per iteration on A100N100 model - full dataset
![R² and MAE values per iteration on A100N100 model - full dataset](https://i.imgur.com/faV9WgG.png)

### Above but with training values - seems overfitted although the test values seem good
![Above with training values](https://i.imgur.com/Q9VcDPY.png)

### R² and MAE values per iteration on A10N10 model - full dataset
![R² and MAE values per iteration on A100N100 model - full dataset](https://i.imgur.com/7LR2IJP.png)

### Above but with training values
![Above with training values](https://i.imgur.com/7kG9hXn.png)

### R² and MAE values per iteration on A100N100 model - pre pandemic dataset
![R² and MAE values per iteration on A100N100 model - pre pandemic dataset](https://i.imgur.com/LOIXmjf.png)

### Above but with training values - seems overfitted although the test values seem good
![Above with training values](https://i.imgur.com/bA8Jbge.png)

### R² and MAE values per iteration on A10N10 model - pre pandemic dataset
![R² and MAE values per iteration on A100N100 model - pre pandemic dataset](https://i.imgur.com/pCQuDVD.png)

### Above but with training values
![Above with training values](https://i.imgur.com/wLeyMb2.png)

### R² and MAE values per iteration on A100N100 model - pandemic dataset
![R² and MAE values per iteration on A100N100 model - pandemic dataset](https://i.imgur.com/PftcB1b.png)

### Above but with training values
![Above with training values](https://i.imgur.com/KabSsg9.png)

### R² and MAE values per iteration on A10N10 model - pandemic dataset
![R² and MAE values per iteration on A100N100 model - pandemic dataset](https://i.imgur.com/PYoh212.png)

### Above but with training values
![Above with training values](https://i.imgur.com/7kG9hXn.png)

### Closer look to what is happening on the iterations of the growing window on the full dataset of the A100N100 model, and what happens when it reaches the pandemic, red line being the regression of predictions vs real values and green line the perfect 'predictions = real values'
![Closer look to what is happening on the iterations of the growing window on the full dataset of the A100N100 model, and what happens when it reaches the pandemic, red line being the regression of predictions vs real values and green line the perfect predictions = real values line](https://i.imgur.com/wWBYxay.png)

### Next update or studies could be done to improve model A10N10 performance without overfitting it, and finding a better way to deal with the variation of the pandemic data. Also, I'm still figuring out how the models with max_depth set to 20 onwards seem overfitted by their training perfomance even though the test perfomance are not from an usual overfitted model.

More analysis like this and the discussion behind those results are in the thesis.

