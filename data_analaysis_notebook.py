# Load libraries and loading data files,file structure and Content


import  pandas as pd            # data processng
import warnings
from datetime  import timedelta
import re
from collections import Counter

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# Load Data

train = pd.read_csv("/Users/sai/Documents/GitHub/DNN2018/data/train_1.csv")


# ################ Missing Values###################

print('\nCheck Number of Records and Columns')
print('Number of Record and Column:', train.shape)

print('\nNull Analysis')
null_samples = train[train.isnull().any(axis=1)]
print('Number of Records Containing 1 + null Columns:', null_samples.shape[0])

# ############# Missing Values######################


# Function to Retrive Page Language


def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org', page)
    if res:
        return res[0][0:2]
    return 'na'


train['lang'] = train.Page.map(get_language)


print(Counter(train.lang))

# Flattening the Data using melt
train_flat_table = pd.melt(train[list(train.columns[-61:])+['Page']], id_vars=['Page', 'lang'], var_name='date',
                                                              value_name='Visits')
train_flat_table['date'] = train_flat_table['date'].astype('datetime64[ns]')
train_flat_table['weekend'] = ((train_flat_table.date.dt.dayofweek) // 5 == 1).astype(float)

# Median by page
df_median = pd.DataFrame(train_flat_table.groupby(['Page'])['Visits'].median())
df_median.columns = ['median']

# Average by page
df_mean = pd.DataFrame(train_flat_table.groupby(['Page'])['Visits'].mean())
df_mean.columns = ['mean']

# Merging data
train_flat_table = train_flat_table.set_index('Page').join(df_mean).join(df_median)

train_flat_table.reset_index(drop=False, inplace=True)

train_flat_table['weekday'] = train_flat_table['date'].apply(lambda x: x.weekday())

# Feature engineering with the date
train_flat_table['year']=train_flat_table.date.dt.year
train_flat_table['month']=train_flat_table.date.dt.month
train_flat_table['day']=train_flat_table.date.dt.day

print(train_flat_table.head())

# Aggregation & Visualisation


# Mean
plt.figure(figsize=(50, 8))
mean_group = train_flat_table[['Page', 'date', 'Visits']].groupby(['date'])['Visits'].mean()
plt.plot(mean_group)
plt.title('Time Series - Average')


# Median
plt.figure(figsize=(50, 8))
median_group = train_flat_table[['Page','date','Visits']].groupby(['date'])['Visits'].median()
plt.plot(median_group, color = 'r')
plt.title('Time Series - Median')

# Std
plt.figure(figsize=(50, 8))
std_group = train_flat_table[['Page','date','Visits']].groupby(['date'])['Visits'].std()
plt.plot(std_group, color = 'g')
plt.title('Time Series - STD')


# For the next graphics
train_flat_table['month_num'] = train_flat_table['month']
train_flat_table['month'].replace('11', '11 - Nov', inplace=True)
train_flat_table['month'].replace('12', '12 - Dec', inplace=True)
train_flat_table['month'].replace('10', '10 - Oct', inplace=True)
train_flat_table['month'].replace('09', '12 - Sep', inplace=True)
train_flat_table['month'].replace('08', '12 - Aug', inplace=True)
train_flat_table['month'].replace('07', '12 - Jul', inplace=True)
train_flat_table['month'].replace('06', '12 - Jun', inplace=True)
train_flat_table['month'].replace('05', '12 - May', inplace=True)
train_flat_table['month'].replace('04', '12 - Apr', inplace=True)
train_flat_table['month'].replace('03', '12 - Mar', inplace=True)
train_flat_table['month'].replace('02', '12 - Feb', inplace=True)
train_flat_table['month'].replace('01', '12 - Jan', inplace=True)


train_flat_table['weekday_num'] = train_flat_table['weekday']
train_flat_table['weekday'].replace(0, '01 - Monday', inplace=True)
train_flat_table['weekday'].replace(1, '02 - Tuesday', inplace=True)
train_flat_table['weekday'].replace(2, '03 - Wednesday', inplace=True)
train_flat_table['weekday'].replace(3, '04 - Thursday', inplace=True)
train_flat_table['weekday'].replace(4, '05 - Friday', inplace=True)
train_flat_table['weekday'].replace(5, '06 - Saturday', inplace=True)
train_flat_table['weekday'].replace(6, '07 - Sunday', inplace=True)

# Web traffic Across Weeks

train_group = train_flat_table.groupby(["month", "weekday"])['Visits'].mean().reset_index()
train_group = train_group.pivot('weekday', 'month', 'Visits')
train_group.sort_index(inplace=True)

sns.set(font_scale=1.5)

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(train_group, annot=False, ax=ax, fmt="d", linewidths=2)
plt.title('Web Traffic Months cross Weekdays')


# Web Traffic Months Across Days

train_day = train_flat_table.groupby(["month", "day"])['Visits'].mean().reset_index()
train_day = train_day.pivot('day', 'month', 'Visits')
train_day.sort_index(inplace=True)
train_day.dropna(inplace=True)

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(train_day, annot=False, ax=ax, fmt="d", linewidths=2)
plt.title('Web Traffic Months cross days')


# Traffic influence By Page Language


tsets_lang = {}
tsets_lang['en'] = train[train.lang=='en'].iloc[:, 0:-1]
tsets_lang['ja'] = train[train.lang=='ja'].iloc[:, 0:-1]
tsets_lang['de'] = train[train.lang=='de'].iloc[:, 0:-1]
tsets_lang['na'] = train[train.lang=='na'].iloc[:, 0:-1]
tsets_lang['fr'] = train[train.lang=='fr'].iloc[:, 0:-1]
tsets_lang['zh'] = train[train.lang=='zh'].iloc[:, 0:-1]
tsets_lang['ru'] = train[train.lang=='ru'].iloc[:, 0:-1]
tsets_lang['es'] = train[train.lang=='es'].iloc[:, 0:-1]

sums = {}
for key in tsets_lang:
    sums[key] = tsets_lang[key].iloc[:,1:].sum(axis=0) / tsets_lang[key].shape[0]

days = [r for r in range(sums['en'].shape[0])]

fig = plt.figure(1, figsize=[10, 10])
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title('Pages in Different Languages')
labels = {'en': 'English', 'ja': 'Japanese', 'de': 'German',
          'na': 'Media', 'fr': 'French', 'zh': 'Chinese',
          'ru': 'Russian', 'es': 'Spanish'
          }

for key in sums:
    plt.plot(days, sums[key], label=labels[key])

plt.legend()
plt.show()
























