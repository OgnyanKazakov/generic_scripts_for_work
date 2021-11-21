## Example structure script

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from time import time
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score



###############################################
## Step 0 
## Data Overview
###############################################


sample_dict = {
    "incidentId": "e6071230-ede1-476c-824e-518d1b2dd6b2",
    "externalIncidentId": "INC0013550",
    "meta": {
        "source": "ServiceNow",
        "incommingData": {
            "parent": "",
            "made_sla": "true",
            "caused_by": "",
            "watch_list": "",
            "upon_reject": "cancel",
            "sys_updated_on": "2018-11-17 03:23:11",
            "child_incidents": "0",
            "hold_reason": "",
            "approval_history": "",
            "number": "INC0013550",
            "resolved_by": "{\"link\":\"https://ven02315.service-now.com/api/now/v1/table/sys_user/6816f79cc0a8016401c5a33be04be441\",\"value\":\"6816f79cc0a8016401c5a33be04be441\"}",
            "sys_updated_by": "admin",
            "opened_by": "{\"link\":\"https://ven02315.service-now.com/api/now/v1/table/sys_user/6816f79cc0a8016401c5a33be04be441\",\"value\":\"6816f79cc0a8016401c5a33be04be441\"}",
            "user_input": "",
            "sys_created_on": "2018-11-17 03:23:08",
            "sys_domain": "{\"link\":\"https://ven02315.service-now.com/api/now/v1/table/sys_user_group/global\",\"value\":\"global\"}",
            "state": "7",
            "sys_created_by": "admin",
            "knowledge": "false",
            "order": "",
            "calendar_stc": "3",
            "closed_at": "2018-11-17 03:23:11",
            "cmdb_ci": "",
            "delivery_plan": "",
            "impact": "3",
            "active": "false",
            "work_notes_list": "",
            "business_service": "",
            "priority": "5",
            "sys_domain_path": "/",
            "rfc": "",
            "time_worked": "",
            "expected_start": "",
            "opened_at": "2018-11-17 03:23:08",
            "business_duration": "1970-01-01 00:00:00",
            "group_list": "",
            "work_end": "",
            "caller_id": "",
            "reopened_time": "",
            "resolved_at": "2018-11-17 03:23:11",
            "approval_set": "",
            "subcategory": "disk",
            "work_notes": "",
            "short_description": "RR_SN_TESTAUTO-1542424886056-RRSNHwDkStClsd-P-5a74b486dbf56700b8f99c16db9619ff",
            "close_code": "",
            "correlation_display": "",
            "delivery_task": "",
            "work_start": "",
            "assignment_group": "",
            "additional_assignee_list": "",
            "business_stc": "0",
            "description": "",
            "EXECUTE_INCIDENT_ID": "e6071230-ede1-476c-824e-518d1b2dd6b2",
            "calendar_duration": "1970-01-01 00:00:03",
            "FILTER_ID": "SERVICENOW_GW_RSQA",
            "close_notes": "",
            "notify": "1",
            "sys_class_name": "incident",
            "closed_by": "{\"link\":\"https://ven02315.service-now.com/api/now/v1/table/sys_user/6816f79cc0a8016401c5a33be04be441\",\"value\":\"6816f79cc0a8016401c5a33be04be441\"}",
            "follow_up": "",
            "parent_incident": "",
            "sys_id": "5a74b486dbf56700b8f99c16db9619ff",
            "contact_type": "",
            "reopened_by": "",
            "incident_state": "7",
            "urgency": "3",
            "problem_id": "",
            "company": "",
            "reassignment_count": "0",
            "activity_due": "",
            "assigned_to": "",
            "severity": "3",
            "comments": "",
            "approval": "not requested",
            "sla_due": "",
            "comments_and_work_notes": "",
            "due_date": "",
            "sys_mod_count": "3",
            "reopen_count": "0",
            "sys_tags": "",
            "escalation": "0",
            "upon_approval": "proceed",
            "correlation_id": "",
            "location": "",
            "category": "hardware"
        }
    }
}
 
sample_dict["meta"]["incommingData"].keys()
print(sample_dict["meta"]["incommingData"]["short_description"])
    
## Dates:
##
## sys_updated_on
## sys_created_on
## closed_at
## opened_at
## business_duration
## resolved_at
## calendar_duration

        
opened_at = pd.to_datetime(sample_dict["meta"]["incommingData"]["opened_at"])
print(type(opened_at))

sys_updated_on = pd.to_datetime(sample_dict["meta"]["incommingData"]["sys_updated_on"])
sys_created_on = pd.to_datetime(sample_dict["meta"]["incommingData"]["sys_created_on"])
closed_at = pd.to_datetime(sample_dict["meta"]["incommingData"]["closed_at"])
business_duration = pd.to_datetime(sample_dict["meta"]["incommingData"]["business_duration"])
resolved_at = pd.to_datetime(sample_dict["meta"]["incommingData"]["resolved_at"])
calendar_duration = pd.to_datetime(sample_dict["meta"]["incommingData"]["calendar_duration"])
default_date = pd.to_datetime("1970-01-01 00:00:00")

print("The time for ticket resolution is {} seconds (difference between \
      open and close date)".format(closed_at - opened_at))

print("The time for ticket resolution is {} seconds (difference between calendar_duration \
      and default_date)".format(calendar_duration - default_date))
 

###############################################
## Step 1
## Visual Representation
###############################################

## SOM -need to be changed

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)


## t-SNE

t_SNE_Train = grouped_df_mean.copy()

t_SNE_Train.head()

index = t_SNE_Train.index

## Scale the data before t-SNE
sc = StandardScaler()
t_SNE_Train_result = sc.fit_transform(t_SNE_Train)



X_train_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(t_SNE_Train_result)


X_train_embedded = pd.DataFrame(X_train_embedded, columns=['Component 1', 'Component 2']).set_index(index)

X_train_embedded.head()



X_train_embedded.reset_index(inplace = True)



# Set style of scatterplot
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")    

# Create scatterplot of dataframe
sns.lmplot(x='Component 1',
           y='Component 2',
           data=X_train_embedded,
           fit_reg=False,
           legend=True,
           size=9,
           hue='cuisine', palette = sns.color_palette("cubehelix", 20),
           scatter_kws={"s":300, "alpha": 0.9})

plt.title('t-SNE Results:', weight='bold').set_fontsize('14')
plt.xlabel('Comp 1', weight='bold').set_fontsize('10')
plt.ylabel('Comp 2', weight='bold').set_fontsize('10')




###############################################
## Step 2
## Drop the categorical variable
## Perform clusterisation
###############################################

## Drop the categorical variable
data_copy = data.copy()

data.drop(, inplace = True, axes = 1)


###############################################
## Step 3
## Classification model based on 20 known categiries and 
## one other category for all not hard-coded categories
## One vs others classification 
###############################################

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)

## Fit one vs rest linear Support Vector Classifier

classifier = OneVsRestClassifier(LinearSVC(random_state = 0))


classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_train_scaled)


### Test
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

###############################################
## Perform the confusion matrix

cm = confusion_matrix(y_train, y_pred)
print(cm)