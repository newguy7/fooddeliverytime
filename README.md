### Food Delivery Time Projects to predict the delivery time based on the Hidden Factors of Delivery Speed

### Column Unnamed: 14 only has NaN value. 
### Columns - ID and Delivery_person_ID are only used for tracking purpose. It wont impact the delivery time prediction
### We will remove Columns - Unnamed: 14, ID and Delivery_person_ID before preparing the data for the model

### We also need to handle missing values for columns - Restaurant_latitude, Restaurant_longitude, Traffic_Level and Distance (km)

### Since missing values for Restaurant_latitude, Restaurant_longitude means the starting point is unknown, we will remove the rows that has these nan values. 
### For Distance (km), we will use function to calculate distance using OSRM API to fill the missing values. After this if there are still some nan values, we will remove the rows.
### For Traffic_Level, we will use KNNImputer to handle the missing values

### Since the output column is named "TARGET" and it has 541 missing values. We dont want to include generated data for this column as it impacts the models learning accuracy.
## So we will remove the rows that has nan values for column TARGET
