STATISTICAL METHODS OF MACHINE LEARNING\
TASK 1: Predicting Stock Prices with Linear Regression\
• Collect historical data on a daily basis (e.g., opening prices, high,
low, close, and trading volume) for the stock. You will use an API
service like Alpha Vantage (``https://www.alphavantage.co/``) as in the code
example given to you. You'll need to sign up to get your own API for
free.\
• For the prediction, focus on the "close" price. The goal is to predict
the average closing price of the month based on the previous N months
and the corresponding trading volume.\
• Add attributes based on past closing values to help the model identify
patterns. For example, create attributes such as:\
o close_t -1: The average closing price a month ago.\
o close_t -2: The average closing price two months ago.\
....\
o volume_t -1: The average trading volume one month ago.\
o volume_t -2: Average trading volume 2 months ago\
o ...\
These delayed values will be the input data for the regression model.\
• Divide the data into training set and validation set to evaluate the
model. Use 2025 data for validation and previous years for training.
Later you will use your model for real predictions.\
• It may be good to have done a pre -treatment of the values e.g. with a
gaussian filter (select the σ) to eliminate too much noise.

```python
from scipy.ndimage import gaussian_filterid
import numpy as np

# Sample ID data
data = np.array([220, 221, 223, 222, 225, 226, 230, 229])

# Apply Gaussian filter
smoothed_data = gaussian_filterid(data, sigma=1) # Adjust sigma for smoothing level
```

A. Use a linear regression model to find the relationship between past
closing values (delay characteristics) and the target (next closing
price).\
Choose how far in time you want to go -- do more testing. Be careful not
to raise the number of parameters too much.\
Indicate the parameters of the model you calculated. Report the
appropriate error metrics for the training set and validation set.\
B. Use a polynomial regression model with L1, L2 normalization norms.
Select appropriate hyperparameters. Indicate the parameters of the model
you calculated. Report the appropriate error metrics for the training
set and validation set.\
C. Reduce the dimension by following PCA, CFA and a wrapper method of
your choice. Compare the results.\
D. Provide price prediction for December 2025 and January 2025.\
General instructions:\
• From the list of stock symbols, each student should choose a unique
stock symbol. In the relevant online form you will have to fill in your
code so that it is visible to everyone that the symbol has already been
assigned to a student.\
• My symbol is NFLX for the Netflix, Inc. and the G ICS Sector:
Communication Services .

• Each symbol may need a different model, depending on its volatility.
There is no one -size- fits-all solution.\
• You will save the data in a .csv file from where you will then upload
it for further processing. Recovery/storage are described in the
accompanying code file given to you.

Use the code in the folder ``Provided Code`` that was given to me by my teacher, but also generate a more correct or useful whenever needed and also explaining it.
