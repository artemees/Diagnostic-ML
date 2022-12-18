# Diagnostic-ML
ML classification on Brain stroke, Diabetes mellitus  &amp; Cardiovascular disease.


```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import confusion_matrix #for confusion matrix
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```


```python
df =pd.read_csv('brain_stroke.csv')
```


```python
df.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Female</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>81.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>186.21</td>
      <td>29.0</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[df['age']<18, 'work_type'].value_counts()
```




    children         673
    Private          132
    Self-employed     12
    Govt_job           6
    Name: work_type, dtype: int64




```python
df.loc[df['age']==17, 'work_type'].value_counts()
```




    Private          47
    Self-employed     3
    Govt_job          3
    Name: work_type, dtype: int64




```python
df.loc[df['age']==16, 'work_type'].value_counts()
```




    Private          35
    children         11
    Self-employed     1
    Govt_job          1
    Name: work_type, dtype: int64




```python
df.loc[df['age']<=16 , 'work_type']='children'
```


```python
df.loc[df['age']==16, 'work_type'].value_counts()
```




    children    48
    Name: work_type, dtype: int64




```python
df.stroke.value_counts()
```




    0    4733
    1     248
    Name: stroke, dtype: int64



##### The data is highy imbalaced.

##### The number of those without stroke(4733) overwhelms when compared to those with stroke (248).

### Visualizing the data in its original form.


```python
features = ['stroke','gender','ever_married','hypertension', 'heart_disease','work_type', 'Residence_type','smoking_status']
plt.figure(figsize =(30,20))
for i in enumerate(features):
    plt.subplot(4,3,i[0]+1)
    sns.countplot(i[1], data = df, palette = 'cividis')
```


    
![bs1](https://user-images.githubusercontent.com/59312765/208315151-7d81f76d-358d-4a17-8761-351f6dfa4f60.png)

    


### Summary of visualization 1.
1. The number of people without stroke and/or other disease is overwhelmingly higher than those who have had them. This indicates that the data is highly imbalanced and will affect the outcome of trained data used in the predictive models.

2. A significant number of people in this dataset are  married, female and work in the private sector.


3. The data also shows that most of the people fall in the 'never smoked' and 'unknown' category.


```python
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.distplot(df.bmi, kde = False)

plt.subplot(1,2,2)
sns.distplot(df.avg_glucose_level, kde = False)
plt.show()
```


    
![bs2](https://user-images.githubusercontent.com/59312765/208315241-eb54aa77-b5d9-4099-8be4-29d4bbacc3f4.png)
    


### Summary of visualization 2.

BMI:  patients bmi peaks btween ranges 23 to 32 which indicates that a majority of patients are overweight.
AGL(average glucose levels): Peaks at 70-90 

### Confusuion Matrix Before Label Encoding


```python
plt.figure(figsize=(12,10))
sns.heatmap(new_df1.corr(), annot=True, fmt= ".2f")
plt.show()
```



![bs3](https://user-images.githubusercontent.com/59312765/208315273-8711c4fa-d985-463b-966e-9b5bb6af75ed.png)

 


### Data Manipulation, Exploration & Visualization.


```python
new_df=df
new_df1 = df
```


```python
# Splitting the age into bins and in a new column.
new_df['age_bins'] =pd.cut(new_df['age'],
       bins=[0,10,20,30,40,50,60,70,80,90],
       labels=['10',
               '20', 
               '30', 
               '40','50','60','70','80', '90'])
```


```python
# Splitting bmi into bins and in a new column.
new_df['weight_status']=pd.cut(new_df['bmi'],
       bins=[0,18.5, 24.9, 30,200],
       labels=['underweight', 
               'normal', 
               'overweight', 
               'obese'])
```


```python
#categorizing those who have had stroke and 1 more disease, all three and singularly.
def categorise(row):  
    if row['hypertension']<1 and row['heart_disease'] < 1 and row['stroke'] < 1:
        return 'low risk'
    if row['hypertension']>0 and row['heart_disease'] >0 and row['stroke'] >0:
        return 'HTN/HD/STRK'
    elif row['hypertension'] >0 and row['stroke'] >0  and row['heart_disease'] < 1:
        return 'STRK/HTN'
    elif row['heart_disease'] >0 and row['stroke']>0 and row['hypertension'] < 1:
        return 'HD/STRK'
    elif row['heart_disease']>0 and row['hypertension']>0 and row['stroke'] < 1:
        return 'HD/HTN'
    elif row['stroke']>0 and row['hypertension']<1 and row['heart_disease'] < 1:
        return 'STRK'
    elif row['hypertension']>0 and row['heart_disease']<1 and row['stroke'] < 1:
        return 'HTN'
    elif row['heart_disease']>0 and row['hypertension']<1 and row['stroke'] < 1:
        return 'HD'
    return 'unknown'
```


```python
new_df['health_status'] = new_df.apply(lambda row: categorise(row), axis=1)
```


```python
plt.figure(figsize=(25,20))
plt.subplot(3,3,1)
sns.countplot('gender', data = new_df.query('stroke>0'), palette='cividis')

plt.subplot(3,3,2)
sns.countplot('age_bins', data = new_df.query('stroke>0'),palette='cividis')

plt.subplot(3,3,3)
sns.countplot('ever_married', data = new_df.query('stroke>0'),palette='cividis')

plt.subplot(3,3,4)
sns.countplot('heart_disease', data = new_df.query('stroke>0'),palette='cividis')
#plt.xticks(rotation = 45)

plt.subplot(3,3,5)
sns.countplot('hypertension', data = new_df.query('stroke>0'),palette='cividis')

plt.subplot(3,3,6)
sns.countplot('weight_status', data = new_df.query('stroke>0'),palette='cividis')

plt.subplot(3,3,7)
sns.countplot('health_status', data = new_df.query('stroke>0'), hue = 'weight_status',palette='cividis')

plt.subplot(3,3,8)
sns.countplot('work_type', data = new_df.query('stroke>0'),palette='cividis')

plt.subplot(3,3,9)
sns.countplot('smoking_status', data = new_df.query('stroke>0'), palette='cividis')



plt.show()  
```


    
![png](output_24_0.png)
    


## Further observations focusing on patients who have had stroke.

##### The above visiualizations gives insights into the data trends focusing solely on patients who have had stroke:

There are more females than males who have had stroke.

The age bracket peaks at the 80's range. The majority of them fall within 60 to 80 years.

Patients with stroke are mostly overweight and obese.

Most of the patients have never smoked. However, a large amount are listed as former smokers. 

The number of people who have had hypertension are more than those who have had heart disease.

##### The health status chart shows the following: 

- There are those who have had one ore more comobordities.i.e stroke and one or more diseases.

- Within each health bracket, the majority of patients are obese particularly within the heart disease/stroke bracket and especially those who have had all three (heart disease, hypertension and stroke).  

They are mostly private workers and self employed. An overwhelming amount of them are married.

### Label Encoding


```python
enc = LabelEncoder()
new_df.loc[:,['gender','ever_married','work_type','Residence_type','smoking_status']] = \
new_df.loc[:,['gender','ever_married','work_type','Residence_type','smoking_status']].apply(enc.fit_transform)
new_df.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
      <th>age_bins</th>
      <th>weight_status</th>
      <th>health_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>1</td>
      <td>1</td>
      <td>70</td>
      <td>obese</td>
      <td>HD/STRK</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>2</td>
      <td>1</td>
      <td>80</td>
      <td>obese</td>
      <td>HD/STRK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>3</td>
      <td>1</td>
      <td>50</td>
      <td>obese</td>
      <td>STRK</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>2</td>
      <td>1</td>
      <td>80</td>
      <td>normal</td>
      <td>STRK/HTN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>81.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>186.21</td>
      <td>29.0</td>
      <td>1</td>
      <td>1</td>
      <td>90</td>
      <td>overweight</td>
      <td>STRK</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_df = new_df.drop(['age_bins', 'weight_status', 'health_status'], axis = 1)
new_df.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>81.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>186.21</td>
      <td>29.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### SMOTE-ENN (Balancing out the data)


```python
from imblearn.combine import SMOTEENN
sen = SMOTEENN(random_state = 2)
```


```python
x_sen = new_df.drop('stroke', axis = 1)
y_sen= new_df.stroke
x_sen.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>81.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>186.21</td>
      <td>29.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_train_sen , y_train_sen = sen.fit_resample(x_sen, y_sen)
x_train_sen.shape, y_train_sen.shape
```




    ((8021, 10), (8021,))



### Scaling the data.


```python
scaler_sen = StandardScaler()
scaler_sen.fit(x_train_sen)
std_data = scaler_sen.transform(x_train_sen)
x_train_sen = std_data
x_train_sen
```




    array([[ 1.49474887, -2.23378339, -0.28943473, ..., -0.43509499,
            -1.80390742, -1.2764539 ],
           [-0.6690087 , -2.01581621, -0.28943473, ..., -0.14907959,
            -1.86922826, -1.2764539 ],
           [-0.6690087 ,  1.07931777, -0.28943473, ..., -0.76228065,
             0.97222816, -1.2764539 ],
           ...,
           [-0.6690087 ,  1.06996911, -0.28943473, ..., -0.97107543,
             2.30493231,  0.79547501],
           [ 1.49474887,  0.27754847,  3.45501039, ..., -0.77487909,
            -0.27879696,  0.79547501],
           [ 1.49474887,  0.22313895, -0.28943473, ..., -0.14244249,
             0.64101379, -0.24048944]])



### Train_test_split Application (training the data).


```python
x_train_sen , x_test, y_train_sen, y_test = train_test_split(x_train_sen, y_train_sen, test_size = 0.3, random_state = 42)
```


```python
x_train_sen.shape, x_test.shape, y_train_sen.shape, y_test.shape
```




    ((5614, 10), (2407, 10), (5614,), (2407,))




```python
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
```

### K_Nearest Neighbors algorithm.


```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train_sen, y_train_sen)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
acc
```




    0.9135853759867054




```python
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

print('accuracy score :', acc)
print('precision score :', prec)
print('recall score :', recall)
print('f1 score :', f1)

target_name =['class 1', 'class 0']
print(classification_report(y_test, y_predict, target_names = target_name,))
cf = confusion_matrix(y_test, y_predict)
cf
```

    accuracy score : 0.9135853759867054
    precision score : 0.8824352694191743
    recall score : 0.9692544196771714
    f1 score : 0.9238095238095239
                  precision    recall  f1-score   support
    
         class 1       0.96      0.85      0.90      1106
         class 0       0.88      0.97      0.92      1301
    
        accuracy                           0.91      2407
       macro avg       0.92      0.91      0.91      2407
    weighted avg       0.92      0.91      0.91      2407
    
    




    array([[ 938,  168],
           [  40, 1261]], dtype=int64)



### Creating a Predictive Model based on the KNN Algorithm.


```python
#new_df.sample(frac=1).reset_index(drop=True).head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>36.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>71.32</td>
      <td>43.9</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>87.06</td>
      <td>28.3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>55.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>57.30</td>
      <td>41.5</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>59.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>92.04</td>
      <td>24.2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>70.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>239.07</td>
      <td>26.1</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



##### I will be testing the predictive model to see if it is able to predict accurately those who have had stroke and those who have not.
##### The data set (row 2) above shows one instance of a person who had stroke. This data as well as an instance of a person who has not had stroke,
##### will be inputed in the model to see if it would predict correctly. 
##### Data inputed, excludes the stroke column.

### Predictive model for K-Nearest Neighbors


```python
inputed_data = (1, 70.0, 0, 1, 1, 1, 0, 239.07, 26.1, 2)          #Data From row 4.
data_as_np = np.asarray(inputed_data)
inputed_reshaped = data_as_np.reshape(1,-1)

std_data = scaler.transform(inputed_reshaped)
print(std_data)
prediction = model.predict(std_data)
print(prediction)


if (prediction[0] == 0):
    print('Nope!!! This person has not had stroke')
else:
    print('Oh yeah. definitely had stroke')
```

    [[ 1.00305350e+00  6.99434281e+01  5.61167343e-03  9.87139929e-01
       1.00854890e+00  9.91528461e-01 -7.48478438e-03  2.39964211e+02
       2.59704324e+01  1.98405105e+00]]
    [1]
    Oh yeah. definitely had stroke
    


```python
inputed_data = (1, 59.0, 1, 0, 1, 1, 0, 92.04, 24.2, 2)          #Data From row 3.
data_as_np = np.asarray(inputed_data)
inputed_reshaped = data_as_np.reshape(1,-1)

std_data = scaler.transform(inputed_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)


if (prediction[0] == 0):
    print('Nope!!! This person has not had stroke')
else:
    print('Oh yeah. definitely had stroke')
```

    [[ 1.00305350e+00  5.89551638e+01  1.01325346e+00 -4.34753173e-03
       1.00854890e+00  9.91528461e-01 -7.48478438e-03  9.23902033e+01
       2.40812005e+01  1.98405105e+00]]
    [0]
    Nope!!! This person has not had stroke
    

### Success!!! The model was trained successfully and has correctly predicted stroke and no stroke instances.
