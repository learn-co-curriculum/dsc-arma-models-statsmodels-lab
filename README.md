# ARMA Models in StatsModels - Lab 

## Introduction

In this lesson, you'll fit an ARMA model using `statsmodels` to a real-world dataset. 


## Objectives

In this lab you will: 

- Decide the optimal parameters for an ARMA model by plotting ACF and PACF and interpreting them 
- Fit an ARMA model using StatsModels 

## Dataset

Run the cell below to import the dataset containing the historical running times for the men's 400m in the Olympic games.


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

data = pd.read_csv('winning_400m.csv')
data['year'] = pd.to_datetime(data['year'].astype(str))
data.set_index('year', inplace=True)
data.index = data.index.to_period("Y")
```


```python
# __SOLUTION__ 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

data = pd.read_csv('winning_400m.csv')
data['year'] = pd.to_datetime(data['year'].astype(str))
data.set_index('year', inplace=True)
data.index = data.index.to_period("Y")
```


```python
# Preview the dataset
data
```


```python
# __SOLUTION__ 
# Preview the dataset
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>winning_times</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1900</th>
      <td>49.4</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>49.2</td>
    </tr>
    <tr>
      <th>1908</th>
      <td>50.0</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>48.2</td>
    </tr>
    <tr>
      <th>1920</th>
      <td>49.6</td>
    </tr>
    <tr>
      <th>1924</th>
      <td>47.6</td>
    </tr>
    <tr>
      <th>1928</th>
      <td>47.8</td>
    </tr>
    <tr>
      <th>1932</th>
      <td>46.2</td>
    </tr>
    <tr>
      <th>1936</th>
      <td>46.5</td>
    </tr>
    <tr>
      <th>1948</th>
      <td>46.2</td>
    </tr>
    <tr>
      <th>1952</th>
      <td>45.9</td>
    </tr>
    <tr>
      <th>1956</th>
      <td>46.7</td>
    </tr>
    <tr>
      <th>1960</th>
      <td>44.9</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>45.1</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>43.8</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>44.7</td>
    </tr>
    <tr>
      <th>1976</th>
      <td>44.3</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>44.6</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>44.3</td>
    </tr>
    <tr>
      <th>1988</th>
      <td>43.9</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>43.5</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>43.5</td>
    </tr>
  </tbody>
</table>
</div>



Plot this time series data. 


```python
# Plot the time series
```


```python
# __SOLUTION__ 
# Plot the time series
data.plot(figsize=(12,6), linewidth=2, fontsize=12)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Winning times (in seconds)', fontsize=12);
```


    
![png](index_files/index_7_0.png)
    


If you plotted the time series correctly, you should notice that it is not stationary. So, difference the data to get a stationary time series. Make sure to remove the missing values.


```python
# Difference the time series
data_diff = None
data_diff
```


```python
# __SOLUTION__ 
# Difference the time series
data_diff = data.diff().dropna()
data_diff
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>winning_times</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1904</th>
      <td>-0.2</td>
    </tr>
    <tr>
      <th>1908</th>
      <td>0.8</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>-1.8</td>
    </tr>
    <tr>
      <th>1920</th>
      <td>1.4</td>
    </tr>
    <tr>
      <th>1924</th>
      <td>-2.0</td>
    </tr>
    <tr>
      <th>1928</th>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1932</th>
      <td>-1.6</td>
    </tr>
    <tr>
      <th>1936</th>
      <td>0.3</td>
    </tr>
    <tr>
      <th>1948</th>
      <td>-0.3</td>
    </tr>
    <tr>
      <th>1952</th>
      <td>-0.3</td>
    </tr>
    <tr>
      <th>1956</th>
      <td>0.8</td>
    </tr>
    <tr>
      <th>1960</th>
      <td>-1.8</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>-1.3</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>0.9</td>
    </tr>
    <tr>
      <th>1976</th>
      <td>-0.4</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>0.3</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>-0.3</td>
    </tr>
    <tr>
      <th>1988</th>
      <td>-0.4</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>-0.4</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Use `statsmodels` to plot the ACF and PACF of this differenced time series. 


```python
# Plot the ACF

```


```python
# __SOLUTION__ 
# Plot the ACF
from statsmodels.graphics.tsaplots import plot_acf
fig, ax = plt.subplots(figsize=(8,3))
plot_acf(data_diff,ax=ax, lags=8);
```


    
![png](index_files/index_13_0.png)
    



```python
# Plot the PACF

```


```python
# __SOLUTION__ 
# Plot the PACF
from statsmodels.graphics.tsaplots import plot_pacf
fig, ax = plt.subplots(figsize=(8,3))
plot_pacf(data_diff,ax=ax, lags=8, method='ywm');
```


    
![png](index_files/index_15_0.png)
    


Based on the ACF and PACF, fit an ARMA model with the right orders for AR and MA. Feel free to try different models and compare AIC and BIC values, as well as significance values for the parameter estimates. 


```python

```


```python
# __SOLUTION__ 
# Import ARIMA
from statsmodels.tsa.arima.model import ARIMA

# Fit an ARMA(1,0) model
mod_arma = ARIMA(data_diff, order=(1,0,0))
res_arma = mod_arma.fit()

# Print out summary information on the fit
print(res_arma.summary())
```

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:          winning_times   No. Observations:                   21
    Model:                 ARIMA(1, 0, 0)   Log Likelihood                 -20.054
    Date:                Fri, 03 Mar 2023   AIC                             46.107
    Time:                        12:34:30   BIC                             49.241
    Sample:                    12-31-1904   HQIC                            46.787
                             - 12-31-1996                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -0.2885      0.081     -3.559      0.000      -0.447      -0.130
    ar.L1         -0.7186      0.144     -5.005      0.000      -1.000      -0.437
    sigma2         0.3819      0.180      2.121      0.034       0.029       0.735
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.04   Jarque-Bera (JB):                 1.19
    Prob(Q):                              0.84   Prob(JB):                         0.55
    Heteroskedasticity (H):               0.33   Skew:                             0.20
    Prob(H) (two-sided):                  0.16   Kurtosis:                         1.91
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).



```python

```


```python
# __SOLUTION__ 
# Fit an ARMA(2,1) model
mod_arma = ARIMA(data_diff, order=(2,0,1))
res_arma = mod_arma.fit()

# Print out summary information on the fit
print(res_arma.summary())
```

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:          winning_times   No. Observations:                   21
    Model:                 ARIMA(2, 0, 1)   Log Likelihood                 -19.931
    Date:                Fri, 03 Mar 2023   AIC                             49.862
    Time:                        12:34:31   BIC                             55.084
    Sample:                    12-31-1904   HQIC                            50.995
                             - 12-31-1996                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -0.2834      0.092     -3.079      0.002      -0.464      -0.103
    ar.L1         -0.6102      2.583     -0.236      0.813      -5.673       4.453
    ar.L2          0.1280      1.848      0.069      0.945      -3.493       3.749
    ma.L1         -0.0208      2.564     -0.008      0.994      -5.046       5.004
    sigma2         0.3774      0.181      2.088      0.037       0.023       0.732
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.04   Jarque-Bera (JB):                 1.21
    Prob(Q):                              0.83   Prob(JB):                         0.55
    Heteroskedasticity (H):               0.31   Skew:                             0.22
    Prob(H) (two-sided):                  0.14   Kurtosis:                         1.91
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).



```python

```


```python
# __SOLUTION__ 
# Fit an ARMA(2,2) model
mod_arma = ARIMA(data_diff, order=(2,0,2))
res_arma = mod_arma.fit()

# Print out summary information on the fit
print(res_arma.summary())
```

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:          winning_times   No. Observations:                   21
    Model:                 ARIMA(2, 0, 2)   Log Likelihood                 -16.472
    Date:                Fri, 03 Mar 2023   AIC                             44.943
    Time:                        12:34:35   BIC                             51.210
    Sample:                    12-31-1904   HQIC                            46.303
                             - 12-31-1996                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -0.2718      0.103     -2.630      0.009      -0.474      -0.069
    ar.L1         -1.7573      0.117    -14.997      0.000      -1.987      -1.528
    ar.L2         -0.9180      0.120     -7.667      0.000      -1.153      -0.683
    ma.L1          1.5669     47.062      0.033      0.973     -90.673      93.807
    ma.L2          0.9985     59.964      0.017      0.987    -116.529     118.526
    sigma2         0.2126     12.706      0.017      0.987     -24.691      25.116
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.03   Jarque-Bera (JB):                 0.87
    Prob(Q):                              0.86   Prob(JB):                         0.65
    Heteroskedasticity (H):               0.41   Skew:                            -0.30
    Prob(H) (two-sided):                  0.26   Kurtosis:                         2.20
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).


## What is your final model? Why did you pick this model?


```python
# Your comments here
```


```python
# __SOLUTION__ 

"""
ARMA(1,0), ARMA(2,2) and ARMA(2,1) all seem to have decent fits with significant parameters. 
Depending on whether you pick AIC or BIC as a model selection criterion, 
your result may vary. In this situation, you'd generally go for a model with fewer parameters, 
so ARMA(1,0) seems fine. Note that we have a relatively short time series, 
which can lead to a more difficult model selection process.
"""
```




    "\nARMA(1,0), ARMA(2,2) and ARMA(2,1) all seem to have decent fits with significant parameters. \nDepending on whether you pick AIC or BIC as a model selection criterion, \nyour result may vary. In this situation, you'd generally go for a model with fewer parameters, \nso ARMA(1,0) seems fine. Note that we have a relatively short time series, \nwhich can lead to a more difficult model selection process.\n"



## Summary 

Well done. In addition to manipulating and visualizing time series data, you now know how to create a stationary time series and fit ARMA models. 
