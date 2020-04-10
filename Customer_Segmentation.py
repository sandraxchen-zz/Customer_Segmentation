#!/usr/bin/env python
# coding: utf-8

# ## Data Prep

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


# In[3]:


# function to create bar plot
def bar_count(df, column):
    df_count = df.copy()
    df_count = df_count[column].value_counts().reset_index().sort_values('index')

    plt.bar(df_count['index'], df_count[column], align='center', alpha=0.5)
    plt.title(column)
    plt.show()


# In[4]:


#read in dataframe
df = pd.read_csv("transactions_n100000.csv")

df.head()


# In[5]:


# get rid of item details in order to get unique ticket_id
df_ticket = df.drop(['item_name','item_count'], axis =1).drop_duplicates()

# change to date time
df_ticket['order_timestamp'] = pd.to_datetime(df_ticket['order_timestamp'])

# create hour
df_ticket['hour'] = df_ticket['order_timestamp'].dt.hour

def hour_type(x):
    if x > 10 and x < 15:
        return "Lunch"
    elif x >= 15 and x < 22:
        return "Dinner"
    else:
        return "Late Night"

df_ticket['hour_type'] = df_ticket['hour'].apply(hour_type)

# day
df_ticket['day'] = df_ticket['order_timestamp'].dt.dayofweek

df_ticket.head()


# In[6]:


#pivot items
items = pd.pivot_table(df, index = 'ticket_id', columns = 'item_name', values ='item_count', aggfunc ='sum', fill_value= 0).reset_index()

items.head()


# In[7]:


# create new features about items
items_newfeatures = items[['ticket_id']]

items_newfeatures['meal_no'] = items['fries'].values
items_newfeatures['burger_ind'] = np.where(items['burger'] > 0, 1, 0)
items_newfeatures['salad_ind'] = np.where(items['salad'] > 0, 1, 0)
items_newfeatures['shake_ind'] = np.where(items['shake'] > 0, 1, 0)

items_newfeatures.head()


# In[8]:


#join item features to df_ticket
df_ticket = df_ticket.merge(items_newfeatures, on = 'ticket_id')

print(len(df_ticket))
df_ticket.head()


# In[9]:


# turning categorical into binary
df_dummies = df_ticket[['location', 'hour_type']].astype(str)
#df_ticket['location'] = df_ticket['location'].astype(str)

df_dummies = pd.get_dummies(df_dummies, drop_first = False)

df_dummies.head()


# In[10]:


df_model = df_ticket[['ticket_id','meal_no', 'burger_ind', 'salad_ind', 'shake_ind']]

df_model = pd.concat([df_model,df_dummies], axis = 1)

df_model.head()


# In[11]:


# plot location

df_location = df_ticket.groupby(['lat', 'long', 'location']).ticket_id.agg('count').reset_index()

df_location.head()

df_location.to_csv('location_count.csv', index = False)


# ## Data Exploration

# In[9]:


print("Rows in dataframe: %d" % len(df.ticket_id))
print("Unique tickets: %d" % len(df.ticket_id.unique()))


# #### Exploring Items

# In[141]:


# burgers and salad are meals. fries match them.
# can create a number of meals and also a y/n flag for salad and burgers
sum(items['burger'] + items['salad'] == items['fries'])


# In[158]:


# people most likely to not get shakes. Then split between getting same as fries or less than fries.
# i think i can group it as a yes/no flag

shake = []

for i in range(0,len(items)):
    if items.iloc[i,4] == 0:
        shake.append("No Shake")
    elif items.iloc[i,4] == items.iloc[i,2]:
        shake.append("Same as Fries")
    elif items.iloc[i,4] < items.iloc[i,2]:
        shake.append("Less than Fries")
    else:
        shake.append("More than Fries")

plt.bar(np.unique(shake, return_counts=True)[0],np.unique(shake, return_counts=True)[1])


# In[126]:


items.describe()


# In[79]:


# fries appears in every order

bar_count(df, 'item_name')


# In[15]:


item_count = df.item_name.value_counts().reset_index()
item_count['pct'] = item_count['item_name']/100000

item_count


# In[31]:


params = {'figure.figsize': (16,8),
          "font.size" : 16,
          'axes.labelsize': 16,
          'axes.titlesize': 18,
          'xtick.labelsize': 18,
          'ytick.labelsize': 14}
plt.rcParams.update(params)

N = len(item_count)

fig, ax = plt.subplots()
plt.box(False)

ind = np.arange(N)    # the x locations for the groups
width = 0.7         # the width of the bars

p1 = ax.bar(ind, item_count['pct'], width, bottom=0, color = '#76b7b2')

#ax.set_title(title)
ax.set(xlabel='', ylabel = '% of transactions', title= '')
ax.set_xticks(ind)
ax.set_xticklabels(['Fries', 'Burger', 'Shake', 'Salad'])

vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
ax.autoscale_view()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:,.1%}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),  # points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(p1)

plt.show()


# In[128]:


#how many orders for each item
for item in df['item_name'].unique():
    item_count = df[df['item_name'] == item]['item_count'].value_counts().reset_index()

    plt.bar(item_count['index'], item_count['item_count'], align='center', alpha=0.5)
    plt.title(item)
    plt.show()


# In[80]:


bar_count(df_ticket, 'location')


# #### Looking at Dates

# In[178]:


# only 2019 data
date_count = df_ticket['order_timestamp'].dt.date.value_counts().reset_index().sort_values('index')

plt.plot(date_count['index'], date_count['order_timestamp'])


# In[179]:


month_count = df_ticket['order_timestamp'].dt.month.value_counts().reset_index().sort_values('index')

plt.plot(month_count['index'], month_count['order_timestamp'])

# less in feb but I think it's because of less days in month. Generally quite stable.


# In[81]:


bar_count(df_ticket, 'hour')


# In[107]:


bar_count(df_ticket, 'hour_type')


# In[108]:


bar_count(df_ticket, 'day')


# In[121]:


dayandhour = pd.pivot_table(df_ticket, columns = 'hour_type', index = 'day', values = 'ticket_id', aggfunc ='count').reset_index()

dayandhour

p1, = plt.plot(dayandhour['day'], dayandhour['Dinner'], label = "Dinner")
p2, = plt.plot(dayandhour['day'], dayandhour['Lunch'], label = "Lunch")
p3, = plt.plot(dayandhour['day'], dayandhour['Late Night'], label = "Late Night")

leg = plt.legend()

plt.show()

# looks pretty stable


# ## Create Clusters

# In[28]:


df_model.iloc[:,:]


# In[50]:


# determine k - look at wcss for each number of clustrs to determine how number of clusters
wcss = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters=i, init='k-means++', tol = 1e-6, n_init=10, random_state=0)
    kmeans.fit(df_model.iloc[:,1:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 21), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[11]:


# create model
nclusters = 3

kmeans = KMeans(n_clusters=nclusters, init='k-means++', tol = 1e-6, n_init=10, random_state=0).fit(df_model.iloc[:,2:])
clusters = kmeans.fit_predict(df_model.iloc[:,1:])


# In[12]:


centroids = pd.DataFrame(kmeans.cluster_centers_)

centroids.columns = df_model.columns[1:]

centroids


# In[13]:


np.unique(clusters, return_counts=True)

