"""
This was a mini project for Graphic Era University , in this we used Cluster 
Analysis, Gradient Boosting classifier (85.48% Accuracy) , Random 
Forest classifier (85.03% Accuracy) and K-NN classifier (85.71% 
Accuracy).

Author- Varun Singh Rana.

"""
import pandas                       # for dataframes
import matplotlib.pyplot as plt     # for plotting graphs
import seaborn as sns               # for plotting graphs

data=pandas.read_csv('emp_attrition.csv')

#-----------------------------------------------------------------------------------------------------------

#Cluster Analysis who didnt left
from sklearn.cluster import KMeans
# Filter data
Attrition_emp =  data[['JobSatisfaction', 'YearsAtCompany']][data.Attrition == "No"]
# Create groups using K-means clustering.
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(Attrition_emp)  

# Add new column "label" annd assign cluster labels.
Attrition_emp['label'] = kmeans.labels_
# Draw scatter plot
plt.scatter(Attrition_emp['JobSatisfaction'], Attrition_emp['YearsAtCompany'], c=Attrition_emp['label'],cmap='Accent')
plt.xlabel('JobSatisfaction')
plt.ylabel('YearsAtCompany')
plt.title('3 Clusters of employees who didnt left')
plt.show()          # Green- Happy , Blue- Frustated , Grey- unhappy  

#Cluster Analysis who left
from sklearn.cluster import KMeans
# Filter data
Attrition_emp =  data[['JobSatisfaction', 'YearsAtCompany']][data.Attrition == "Yes"]
# Create groups using K-means clustering.
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(Attrition_emp)  

# Add new column "label" annd assign cluster labels.
Attrition_emp['label'] = kmeans.labels_
# Draw scatter plot
plt.scatter(Attrition_emp['JobSatisfaction'], Attrition_emp['YearsAtCompany'], c=Attrition_emp['label'],cmap='Accent')
plt.xlabel('JobSatisfaction')
plt.ylabel('YearsAtCompany')
plt.title('3 Clusters of employees who left')
plt.show()          # Green- Happy , Blue- Frustated , Grey- unhappy 
 


#-----------------------------------------------------------------------------------------------------------
    
 # Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
data['Department']=le.fit_transform(data['Department'])

#Spliting data into Feature and
X=data[['JobSatisfaction', 'YearsInCurrentRole', 'JobInvolvement',
       'MonthlyIncome', 'YearsAtCompany', 'JobLevel',
       'YearsSinceLastPromotion', 'Department', 'AppraisalRating' , 'PercentSalaryHike']]
y=data['Attrition']

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 


#-----------------------------------------------------------------------------------------------------------

#Import Gradient Boosting Classifier model
from sklearn.ensemble import GradientBoostingClassifier

#Create Gradient Boosting Classifier
gb = GradientBoostingClassifier()

#Train the model using the training sets
gb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gb.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Gradient Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("-------------------------------------------------------------------------------------------")
print("\n")

#-----------------------------------------------------------------------------------------------------------

#Converting Yes and No to 1 and 0
y_test = y_test.map({'Yes': 1, 'No': 0}).astype(int)
y_train = y_train.map({'Yes': 1, 'No': 0}).astype(int)

#-----------------------------------------------------------------------------------------------------------

# Random Forest Classifier
   
from sklearn.ensemble import RandomForestClassifier 

classifier = RandomForestClassifier(n_estimators=100 ,criterion='entropy', random_state=100 )
classifier.fit(X_train , y_train)

rf_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report , accuracy_score , recall_score

print("Random Forest")
print(classification_report(y_test, rf_pred))
print("Recall Score : ", recall_score(y_test, rf_pred))
print("Acuuracy_Train_Log : ", classifier.score(X_train, y_train))
print("Acuuracy_Test_Log : ", accuracy_score(y_test, rf_pred)) 
print("-------------------------------------------------------------------------------------------")
print("\n")

#-----------------------------------------------------------------------------------------------------------

#Converting 1 and 0 to Yes and No
y_test = y_test.map({ 1 : 'Yes', 0 : 'No'}).astype(str)
y_train = y_train.map({ 1 : 'Yes', 0 : 'No'}).astype(str)

#-----------------------------------------------------------------------------------------------------------

#K -NN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

for K in range(25):
    K_value = K+1
    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
    neigh.fit(X_train, y_train) 
    predict_y = neigh.predict(X_test)
    print ("K-NN Accuracy is ", accuracy_score(y_test,predict_y)*100,"% for K-Value:",K_value)
    
print("-------------------------------------------------------------------------------------------")    
#-----------------------------------------------------------------------------------------------------------

#Creating DataFrame for Gradient Boosting Classifier (pred + train)

gbpr = pandas.DataFrame(y_pred)
gbpr.index = y_test.index           # Changing the prediction index with y_test index to link the emp
 
gbrs = pandas.concat([gbpr , y_train])  # Merging the whole Dataset back 
gbrs = gbrs.sort_index()                # Sorting the whole Dataset

#-----------------------------------------------------------------------------------------------------------

#Creating DataFrame for Random Forest Classifier (pred + train)

rfpr = pandas.Series(rf_pred)                         #Converting into Series 
rfpr = rfpr.map({ 1 : 'Yes', 0 : 'No'}).astype(str)   #Converting 0 and 1 to Yes and No

rfrs = pandas.DataFrame(rfpr)
rfrs.index = y_test.index

rfrs = pandas.concat([rfrs , y_train])             # Merging the whole Dataset back 
rfrs = rfrs.sort_index()                           # Sorting the whole Dataset

#-----------------------------------------------------------------------------------------------------------

#Creating DataFrame for K-NN (pred + train)

knnpr = pandas.DataFrame(predict_y)
knnpr.index = y_test.index           # Changing the prediction index with y_test index to link the emp
 
knnrs = pandas.concat([knnpr , y_train])  # Merging the whole Dataset back 
knnrs = knnrs.sort_index()                # Sorting the whole Dataset

#-----------------------------------------------------------------------------------------------------------
# Converting DataFrames to Series

gbrs = pandas.Series(gbrs[0].values)  
rfrs = pandas.Series(rfrs[0].values)  
knnrs = pandas.Series(knnrs[0].values)

gbrs = gbrs.map({'Yes': 1, 'No': 0}).astype(int)
rfrs = rfrs.map({'Yes': 1, 'No': 0}).astype(int)
knnrs = knnrs.map({'Yes': 1, 'No': 0}).astype(int)

#Creating a DataFrame with all Predictions 
attr = (gbrs , rfrs , knnrs)
attr = pandas.concat(attr, axis=1)
attr=attr.set_axis(['GB', 'RF', 'KNN'], axis=1)  # Axis 0 (Row) , 1(Column)

#Taking the average of all the models prediction
mn= attr.mean(axis = 1 )

#Converting decimals into 1 and 0
for i in range(1,len(mn)-1): 
    if mn[i] < 0.5 : mn[i]=0
    else : mn[i]=1
    
#Creating a final DataFrame
  
data['Department']=le.inverse_transform(data['Department'])  # Converting Number back to labels 
  
attr = (gbrs , rfrs , knnrs, mn ,data.AppraisalRating, data.PercentSalaryHike , data.EmployeeNumber , data.JobInvolvement , data.TotalWorkingYears , data.JobLevel ,data.MonthlyIncome ,data.JobRole ,data.Department , data.JobSatisfaction , data.YearsSinceLastPromotion , data.TrainingTimesLastYear , data.OverTime)
attr = pandas.concat(attr, axis=1)
attr=attr.set_axis(['GB', 'RF', 'KNN' , 'Average' , 'Appraisal Rating', 'PercentSalaryHike', 'EmployeeNumber', 'Job Involvement ', 'Total Woring Years' , 'Job Level' , 'Monthy Income' , 'Job Role' , 'Department' ,'Job Satisfaction' ,'Years Since Last Promotion' , 'Training Times Last Year' , 'OverTime'], axis=1)  # Axis 0 (Row) , 1(Column)

attr = attr.sort_index()                # Sorting the whole Dataset
attr.index=attr.index+1                 # Starting value was 0 before this
attr.index.name= 'Index'

print(attr)

#-----------------------------------------------------------------------------------------------------------

attr.to_csv('Emp_Attrition_pred.csv')    # save a CSV file 

#-----------------------------------------------------------------------------------------------------------

#SeaBorn Graphs

# Overall

features=['Education', 'JobInvolvement','TotalWorkingYears','YearsAtCompany','YearsSinceLastPromotion','JobSatisfaction','PercentSalaryHike','JobLevel']

fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data = data, hue='Attrition')
    plt.title("Employee's Detail")
    plt.suptitle('Overall Details 1')
     
features=['TrainingTimesLastYear','YearsInCurrentRole','AppraisalRating','YearsWithCurrManager','Department','NumCompaniesWorked','EmployeeStockOwnership']

fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data = data, hue='Attrition')
    plt.xticks(rotation=90)
    plt.title("Employee's Detail")    
    plt.suptitle('Overall Details 2')

features=['OverTime','Over18','EmployeeStockOwnership']

fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data = data, hue='Attrition')
    plt.title("Employee's Detail")
    plt.suptitle('Overall Details 3')       

#-----------------------------------------------------------------------------------------------------------

# HR 

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
  
fig.suptitle('HRs Details')

sns.barplot(ax=axes[0, 0] , x="JobInvolvement", y="JobLevel", hue="Attrition", data=data)
sns.barplot(ax=axes[0, 1] , x="NumCompaniesWorked", y="MonthlyIncome", hue="Attrition", data=data)
sns.barplot(ax=axes[0, 2] , x="YearsWithCurrManager", y="AppraisalRating", hue="Attrition", data=data)
sns.barplot(ax=axes[1, 0] , x="TrainingTimesLastYear", y="Department", hue="Attrition", data=data)
sns.barplot(ax=axes[1, 1] , x="PercentSalaryHike", y="TotalWorkingYears", hue="Attrition", data=data)
sns.barplot(ax=axes[1, 2] , x="EmployeeStockOwnership", y="YearsAtCompany", hue="Attrition", data=data)

#-----------------------------------------------------------------------------------------------------------

# Manager

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
  
fig.suptitle('Managers Details')

sns.barplot(ax=axes[0, 0] , x="JobInvolvement", y="JobLevel", hue="Attrition", data=data)
sns.barplot(ax=axes[0, 1] , x="YearsInCurrentRole", y="JobSatisfaction", hue="Attrition", data=data)
sns.barplot(ax=axes[0, 2] , x="TrainingTimesLastYear", y="YearsSinceLastPromotion", hue="Attrition", data=data)
sns.barplot(ax=axes[1, 0] , x="JobLevel", y="JobRole", hue="Attrition", data=data)
sns.barplot(ax=axes[1, 1] , x="AppraisalRating", y="YearsInCurrentRole", hue="Attrition", data=data)
sns.barplot(ax=axes[1, 2] , x="Department", y="YearsInCurrentRole", hue="Attrition", data=data)

#---------------------------------------------------------------------------------------------------------------------