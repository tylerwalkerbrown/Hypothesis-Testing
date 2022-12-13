# What is the effect of a new vs old golf ball on driving distance ?


```python
import pandas as pd
import numpy as np 
import math 
import scipy.stats as stats 
import matplotlib.pyplot as plt
%matplotlib inline
import os
```


```python
os.chdir("Desktop/Practice data")
```


```python
#Customer order data between the years 2019-Current 
df = pd.read_csv("Golf.csv")
```

# Exploring the Data


```python
#Orginal shape of the data that will be tested 
df.shape
```




    (40, 2)




```python
#Multiple different columns with NAs but non in our total column that would impact the analysis 
df.isna().sum()
```




    Current    0
    New        0
    dtype: int64




```python
#Examining the top 5 observations 
df.head()
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
      <th>Current</th>
      <th>New</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>264</td>
      <td>277</td>
    </tr>
    <tr>
      <th>1</th>
      <td>261</td>
      <td>269</td>
    </tr>
    <tr>
      <th>2</th>
      <td>267</td>
      <td>263</td>
    </tr>
    <tr>
      <th>3</th>
      <td>272</td>
      <td>266</td>
    </tr>
    <tr>
      <th>4</th>
      <td>258</td>
      <td>262</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Looking at the bottom 5 observations
df.tail()
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
      <th>Current</th>
      <th>New</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>267</td>
      <td>263</td>
    </tr>
    <tr>
      <th>36</th>
      <td>279</td>
      <td>261</td>
    </tr>
    <tr>
      <th>37</th>
      <td>274</td>
      <td>255</td>
    </tr>
    <tr>
      <th>38</th>
      <td>276</td>
      <td>263</td>
    </tr>
    <tr>
      <th>39</th>
      <td>262</td>
      <td>279</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Looking at central tendency 
df.describe()
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
      <th>Current</th>
      <th>New</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>40.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>270.275000</td>
      <td>267.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.752985</td>
      <td>9.896904</td>
    </tr>
    <tr>
      <th>min</th>
      <td>255.000000</td>
      <td>250.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>263.000000</td>
      <td>262.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>270.000000</td>
      <td>265.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>275.250000</td>
      <td>274.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>289.000000</td>
      <td>289.000000</td>
    </tr>
  </tbody>
</table>
</div>



No outliers spotted between the new vs current golf balls. The New golf balls have a lower median than the current golf balls were using.


```python
#Anomaly detection
columns = [df.New, df.Current]
labels = ('New', 'Current')

fig, ax = plt.subplots()
ax.boxplot(columns, labels=labels)
plt.show()
```


    
![png](output_11_0.png)
    


Looking at the distribution the current golf balls have a wider distibution tail showing a more uniform like distribution than the new golf balls.


```python
#Distribution plot for the new vs current golf balls 
plt.figure(1 , figsize = (15 , 6))
n = 0 
for x in ['Current','New']:
    n += 1
    plt.subplot(2 , 2 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.distplot(df[x] , bins = 20)
    plt.title('Distplot of {}'.format(x))
plt.show()
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)



    
![png](output_13_1.png)
    



```python
#Looking at the overlay of data
import seaborn as sns
sns.distplot(df.Current,color='green',hist=False, label = 'current ball')
sns.distplot(df.New,color='red',hist=False, label = 'new ball')
plt.legend()
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
      warnings.warn(msg, FutureWarning)
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
      warnings.warn(msg, FutureWarning)





    <matplotlib.legend.Legend at 0x7fa0fb1ebb20>




    
![png](output_14_2.png)
    


# Hypothesis Testing


```python
#Looking at the mean amount spent to create a null hypothesis 
print("Null - New golf balls have no effect =",df['Current'].mean())
print("Alternative - New balls have an effect =!",df['Current'].mean())
```

    Null - New golf balls have no effect = 270.275
    Alternative - New balls have an effect =! 270.275



```python
#Storing the sample mean, pop mean, sample size and standard deviation 
sample_mean = int(df.Current.mean()) # current old golf balls
h_mean = int(df['New'].mean()) #
N = len(df)
dev = np.std(df.melt(id_vars=["Current"], 
        var_name="Current"))
print("Sample mean = ", sample_mean)
print("Hypothesis mean = ", h_mean)
print("Sample size = ", N )
print("Standard deviation = ", dev)
```

    Sample mean =  270
    Hypothesis mean =  267
    Sample size =  40
    Standard deviation =  value    9.77241
    dtype: float64


    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/numpy/core/fromnumeric.py:3579: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      return std(axis=axis, dtype=dtype, out=out, ddof=ddof, **kwargs)



```python
# Z-score 
z = (sample_mean - h_mean)/(dev/math.sqrt(N))
print("Z-score = ", z)
```

    Z-score =  value    1.941554
    dtype: float64



```python
# P value 
p_value = stats.norm.sf(abs(z))
print("P-value = ", p_value)
```

    P-value =  [0.02609553]



```python
#If statement to reject or accept null hypothesis 
if p_value > .05:
    print("Accept the null: New Balls drive further than old.")
else:
    print("Reject the null and accept the alternative: New golf balls have no effect on drive.")
```

    Reject the null and accept the alternative: New golf balls have no effect on drive.

