
# coding: utf-8

# In[2]:


import datetime
import pandas as pd
import numpy as np
import pickle


# In[3]:


# Count total number of dpds from payment history
def tototalDpdDaysCount(arr):
# def totalDpdDaysCount(arr=np.array([['"""123sssttt300"""'],['"""fffhhh089"""'],['']]),m=None):
    value = []
    for index, x in np.ndenumerate(arr):
        count=0
        x = x.strip('"""')
        f = [x[i:i+3] for i in range(0,len(x),3)]
        for item in f:
            try:
                val=int(item)
                count+=1
                continue
            except(ValueError, Exception) as e:
                count+=0
                continue
        value.append(count)
    return value

# Sum of num of days past between two dpds.
def totaldpdDaysDiff(arr):
# def totaldpdDaysDiff(arr=np.array([['"""STDSTDSTDXXXXXXXXXXXXXXXSTDXXXXXXXXXXXXXXXSTD"""'],['"""fffhhh089"""'],['ffffff']])):
    value = []
    for index, x in np.ndenumerate(arr):
        dpdlist = []
        count=0
        x = x.strip('"""')
        f = [x[i:i+3] for i in range(0,len(x),3)]
        for i, item in enumerate(f):
            try:
                val=int(item)
                dpdlist.append(i)
                continue
            except(ValueError, Exception) as e:
                continue
        k = [(dpdlist[i+1]-dpdlist[i]-1)*30 for i in range(len(dpdlist)-1)]
        if k:
            value.append(k[0])
        else:
            value.append(0)
    return(value)


# In[4]:


# Return Most recent dpd from payment history
def latestdpd(arr):
# def latestdpd(arr=np.array([['"""000150150150150150120090090060030015000015000000000000"""'],['"""fffhhh089"""'],['']]),m=None):
    value = []
    count=0
    for index, x in np.ndenumerate(arr):
        x = x.strip('"""')
        f = [x[i:i+3] for i in range(0,len(x),3)]
        try:
            count=int(f[1])
#             print(count)
            value.append(count)
            continue
        except(ValueError, Exception) as e:
            count=0
            value.append(count)
            continue
    return value


# In[5]:


# Count total number of dpds in days from payment history
def totalDpdDays(arr):
# def totalDpdDays(arr=np.array([['"""123sssttt300"""'],['"""fffhhh089"""'],['']]),m=None):
    value = []
    for index, x in np.ndenumerate(arr):
        count=0
        x = x.strip('"""')
        f = [x[i:i+3] for i in range(0,len(x),3)]
        for item in f:
            try:
                val=int(item)
                count+=val
                continue
            except(ValueError, Exception) as e:
                count+=0
                continue
        value.append(count)
    return value


# In[6]:


df_data_train = pd.read_csv(r'C:\Users\Saurabh Banga\Desktop\Tookitaki\test data\raw_data_70_new.csv',engine='python')
df_account_train = pd.read_csv(r'C:\Users\Saurabh Banga\Desktop\Tookitaki\test data\raw_account_30_new.csv',engine='python', parse_dates = ['dt_opened','upload_dt', 'opened_dt', 'paymt_str_dt', 'paymt_end_dt','last_paymt_dt','closed_dt','reporting_dt'])


# In[7]:


newdf = df_account_train
newdf1 = df_data_train
col_null_count = newdf.isnull().sum(axis=0)
# col_null_count


# In[8]:


newdf.last_paymt_dt.fillna(datetime.datetime.now(),inplace=True)
newdf.opened_dt.fillna(datetime.datetime.now(),inplace=True)


# In[9]:


# newdf.loc[(newdf.opened_dt.isnull()==True) & (newdf.last_paymt_dt.isnull()==False), 'customer_no'].head()


# In[10]:


newdf.creditlimit.fillna(newdf.high_credit_amt, inplace=True)


# In[11]:


h_1 = newdf.groupby('customer_no', as_index=False).agg({'high_credit_amt': ['count','mean']})
l_1 = (h_1[h_1[('high_credit_amt','count')] == 0].customer_no).tolist()
for id in l_1:
    newdf.loc[newdf['customer_no'] == id,'high_credit_amt'] = 0.1
    
h_1 = newdf.groupby('customer_no', as_index=False).agg({'cur_balance_amt': ['count','mean']})
l_1 = (h_1[h_1[('cur_balance_amt','count')] == 0].customer_no).tolist()
for id in l_1:
    newdf.loc[newdf['customer_no'] == id,'cur_balance_amt'] = 0.1

h_1 = newdf.groupby('customer_no', as_index=False).agg({'creditlimit': ['count','mean']})
l_1 = (h_1[h_1[('creditlimit','count')] == 0].customer_no).tolist()
for id in l_1:
    newdf.loc[newdf['customer_no'] == id,'creditlimit'] = 0.1


# In[12]:


# l = [3876,5822,10211,11850,15270,16373,16750,18722,19318,19543,20867,22453,22788]


# In[13]:


loan_account_list = [1,2,3,4,5,6,7,8,9,13,15,17,32,33,34,41,41,42,51,54,59,60]
card_list = [10,14,31,35]


# In[14]:


newdf['diff_opened_lastPaymt_dt'] = newdf['opened_dt'].sub(newdf['last_paymt_dt'], axis=0)
newdf['diff_opened_lastPaymt_dt'] = newdf['diff_opened_lastPaymt_dt'] / np.timedelta64(1, 'D')


# In[15]:


newdf['highcr-creditlimt'] = newdf['high_credit_amt'] - newdf['creditlimit']
newdf['highcr-currBal'] = newdf['high_credit_amt'] - newdf['cur_balance_amt']


# In[16]:


newdf['latestDPD'] = latestdpd(newdf['paymenthistory1'].values)
# newdf['recentDPD'].head()


# In[17]:


newdf['totalDPDdays'] = totalDpdDays(newdf['paymenthistory1'].values)
# newdf['totalDPDdays'].head()


# In[18]:


newdf['dayspastDPD'] = totaldpdDaysDiff(newdf['paymenthistory1'].values)
# newdf['dayspastDPD'].head()


# In[19]:


group1 = newdf.groupby('customer_no', as_index=False).agg({'dayspastDPD': ['mean','sum'],'latestDPD': ['mean','sum'],'totalDPDdays': ['mean','sum']})


# In[20]:


# group1.isnull().sum(axis=0)


# In[21]:


df_derived = pd.DataFrame()
df_derived['customer_no'] = group1['customer_no']
len(df_derived)


# In[22]:


d1 = newdf.groupby('customer_no', as_index=False).agg({'highcr-creditlimt': ['sum']})

d2 = newdf.groupby('customer_no', as_index=False).agg({'highcr-creditlimt': ['mean']})

d3 = newdf.groupby('customer_no', as_index=False).agg({'highcr-currBal': ['sum']})

d4 = newdf.groupby('customer_no', as_index=False).agg({'highcr-currBal': ['mean']})


# In[23]:


df_derived['diff_highcr_creditlim_sum'] = d1[('highcr-creditlimt','sum')]
df_derived['diff_highcr_creditlim_mean'] = d2[('highcr-creditlimt','mean')]
df_derived['diff_highcr_currbal_sum'] = d3[('highcr-currBal','sum')]
df_derived['diff_highcr_currbal_mean'] = d4[('highcr-currBal','mean')]


# In[24]:


d1 = newdf.groupby('customer_no',as_index=False)['high_credit_amt'].apply(lambda x: x.sum(skipna=True))

d2 = newdf.groupby('customer_no',as_index=False)['high_credit_amt'].apply(lambda x: x.mean(skipna=True))

d3 = newdf.groupby('customer_no',as_index=False)['creditlimit'].apply(lambda x: x.sum(skipna=True))

d4 = newdf.groupby('customer_no',as_index=False)['creditlimit'].apply(lambda x: x.mean(skipna=True))

d5 = newdf.groupby('customer_no', as_index=False).agg({'cur_balance_amt': ['sum']})

d6 = newdf.groupby('customer_no', as_index=False).agg({'cur_balance_amt': ['mean']})


# In[25]:


df_derived['high_credit_amt_mean'] = d2
df_derived['creditlimit_mean'] = d4
df_derived['cur_balance_amt_mean'] = d6[('cur_balance_amt','mean')]
df_derived['ratio_totalhighcr_totalcrlim'] = d1/d3
df_derived['ratio_totalCurrbal_totalcrlim'] = d5[('cur_balance_amt','sum')]/d3
df_derived['utilisationTrend'] = df_derived['ratio_totalCurrbal_totalcrlim']/d4


# In[26]:


df_derived.replace(np.nan,0,inplace=True)


# In[27]:


d1 = newdf.groupby('customer_no',as_index=False)['diff_opened_lastPaymt_dt'].apply(lambda x: x.sum(skipna=True))
df_derived['sum_diff_opened_lastPaymt_dt'] = d1


# In[28]:


def countCredit(x):
    count=0
    for index, item in np.ndenumerate(x.values):
        if item in [10,35,14,31]:
            count+=1
    return count/len(x)


# In[29]:


def countAccount(x):
    count=0
    for index, item in np.ndenumerate(x.values):
        if item in [1,2,3,4,5,6,7,8,9,13,15,17,32,33,34,41,41,42,51,54,59,60]:
            count+=1
    return count/len(x)


# In[30]:


def paymthistorylength(x):
    count=0
    for index, str in np.ndenumerate(x):
        str = str.strip('"""')
        count+=len(str)
    return count/len(x)


# In[31]:


d1 = newdf.groupby('customer_no',as_index=False)['acct_type'].apply(countCredit).reset_index(name='creditcount')
df_derived['avg_creditcount'] = d1.creditcount


# In[32]:


d1 = newdf.groupby('customer_no',as_index=False)['acct_type'].apply(countAccount).reset_index(name='loancount')
df_derived['avg_loancount'] = d1.loancount


# In[33]:


d1 = newdf.groupby('customer_no',as_index=False)['paymenthistory1'].apply(paymthistorylength).reset_index(name='historylength')
df_derived['avg_payhistlength'] = d1.historylength


# In[34]:


# df_derived[df_derived['cur_balance_amt_mean'].isnull()==True]
# newdf.columns.values


# In[35]:


activeaccount = newdf.ix[(newdf['latestDPD'] > 0)]
group2 = activeaccount.groupby('customer_no').size().reset_index(name='allaccountdpd')
l = (group1[~group1['customer_no'].isin(group2['customer_no'])==True].customer_no).tolist()
d = pd.DataFrame()
d['customer_no'] = l
d['allaccountdpd'] = [0 for i in range(len(d))]
d1 = pd.concat([d, group2], ignore_index=True)
d1.sort_values('customer_no',inplace=True)
d2 = d1.reset_index(drop=True)
df_derived['allaccountdpd'] = d2['allaccountdpd']
# len(d2)


# In[ ]:


inactiveaccount = newdf.ix[(newdf['last_paymt_dt'] > 0) & (newdf['diff_reporting_closed_dt'] < 0)]
group2 = inactiveaccount.groupby('customer_no').size().reset_index(name='inactiveaccountdpd')
l = (group1[~group1['customer_no'].isin(group2['customer_no'])==True].customer_no).tolist()
d = pd.DataFrame()
d['customer_no'] = l
d['inactiveaccountdpd'] = [0 for i in range(len(d))]
d1 = pd.concat([d, group2], ignore_index=True)
d1.sort_values('customer_no',inplace=True)
d2 = d1.reset_index(drop=True)
df_derived['inactiveaccountdpd'] = d2['inactiveaccountdpd']
# len(d2)


# In[ ]:


df_derived.isnull().sum(axis=0)
df_derived.replace(np.nan,0,inplace=True)


# In[ ]:


df_derived['meanCurBal_by_meanCreditLim'] = df_derived['cur_balance_amt_mean']/df_derived['creditlimit_mean']


# In[ ]:


df_derived.isnull().sum(axis=0)
df_derived.replace(np.nan,0,inplace=True)


# In[ ]:


df_derived['meanHighCrAmt_by_meanCreditLim'] = df_derived['high_credit_amt_mean']/df_derived['creditlimit_mean']


# In[ ]:


df_derived.isnull().sum(axis=0)
df_derived.replace(np.nan,0,inplace=True)


# In[ ]:


df_derived['meanHighCrAmt_by_meanCurBal'] = df_derived['high_credit_amt_mean']/df_derived['cur_balance_amt_mean']


# In[ ]:


# df_derived.isnull().sum(axis=0)
# df_derived.replace(np.nan,0,inplace=True)


# In[ ]:


df_derived['latestDPD_sum'] = group1[('latestDPD', 'sum')]
df_derived['latestDPD_mean'] = group1[('latestDPD', 'mean')]
# df_derived['latestDPD_min'] = group1[('latestDPD', 'min')]
# df_derived['latestDPD_max'] = group1[('latestDPD', 'max')]


# In[ ]:


df_derived['dayspastDPD_sum'] = group1[('dayspastDPD', 'sum')]
df_derived['dayspastDPD_mean'] = group1[('dayspastDPD', 'mean')]
# df_derived['dayspastDPD_min'] = group1[('dayspastDPD', 'min')]
# df_derived['dayspastDPD_max'] = group1[('dayspastDPD', 'max')]


# In[ ]:


df_derived['totalDPDdays_sum'] = group1[('totalDPDdays', 'sum')]
df_derived['totalDPDdays_mean'] = group1[('totalDPDdays', 'mean')]
# df_derived['totalDPDdays_min'] = group1[('totalDPDdays', 'min')]
# df_derived['totalDPDdays_max'] = group1[('totalDPDdays', 'max')]


# In[ ]:


df_derived.isnull().sum(axis=0)
# df_derived.replace(np.nan,0,inplace=True)


# In[ ]:


d_1 = newdf[newdf['high_credit_amt'] > newdf['cur_balance_amt']]
g_1 = d_1.groupby('customer_no').size().reset_index(name='count')
g_2 = newdf.groupby('customer_no').size().reset_index(name='count')
l = (g_2[~g_2['customer_no'].isin(g_1['customer_no'])==True].customer_no).tolist()
d = pd.DataFrame()
d['customer_no'] = l
d['count'] = [0 for i in range(len(d))]
d1 = pd.concat([g_1, d], ignore_index=True)
d1.sort_values('customer_no',inplace=True)
d1.reset_index(drop=True,inplace=True)


# In[ ]:


# df_derived.isnull().sum(axis=0)
# df_derived.replace(np.nan,0,inplace=True)


# In[ ]:


df_derived['expected_highCr_currBal'] = d1['count']*(d1['count']/g_2['count'])


# In[ ]:


df_derived.isnull().sum(axis=0)
# df_derived.replace(np.nan,0,inplace=True)


# In[ ]:


d_1 = newdf[newdf['cur_balance_amt'] > 0]
g_1 = d_1.groupby('customer_no').size().reset_index(name='count')
g_2 = newdf.groupby('customer_no').size().reset_index(name='count')
l = (g_2[~g_2['customer_no'].isin(g_1['customer_no'])==True].customer_no).tolist()
d = pd.DataFrame()
d['customer_no'] = l
d['count'] = [0 for i in range(len(d))]
d1 = pd.concat([g_1, d], ignore_index=True)
d1.sort_values('customer_no',inplace=True)
d1.reset_index(drop=True,inplace=True)


# In[ ]:


df_derived['expected_zeroCurrBal'] = d1['count']*(d1['count']/g_2['count'])


# In[ ]:


df_derived.isnull().sum(axis=0)
# df_derived.replace(np.nan,0,inplace=True)


# In[ ]:


d_1 = newdf[newdf['high_credit_amt'] > newdf['creditlimit']]
g_1 = d_1.groupby('customer_no').size().reset_index(name='count')
g_2 = newdf.groupby('customer_no').size().reset_index(name='count')
l = (g_2[~g_2['customer_no'].isin(g_1['customer_no'])==True].customer_no).tolist()
d = pd.DataFrame()
d['customer_no'] = l
d['count'] = [0 for i in range(len(d))]
d1 = pd.concat([g_1, d], ignore_index=True)
d1.sort_values('customer_no',inplace=True)
d1.reset_index(drop=True,inplace=True)


# In[ ]:


df_derived['expected_highCr_creditLim'] = d1['count']*(d1['count']/g_2['count'])


# In[ ]:


df_derived.isnull().sum(axis=0)
# df_derived.replace(np.nan,0,inplace=True)


# In[ ]:


df_derived['Class'] = newdf1['Bad_label']
df_derived.to_csv(r'C:\Users\Saurabh Banga\Desktop\Tookitaki\FinalData\Test2.csv')


# In[ ]:


print(len(df_data_train),len(df_derived))


# In[ ]:


# dff = pd.read_csv(r'C:\Users\Saurabh Banga\Desktop\Tookitaki\FinalData\Train2.csv', engine='python')


# In[ ]:


# dff[(dff.Class).isnull()==True].customer_no


# In[ ]:


# sum(axis=0)


# In[ ]:


# s2 = group1['customer_no']
# s2
# s = list(s2-s1)


# In[ ]:


# newdf[newdf['customer_no'].isin(s)].


# In[ ]:


# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.std(skipna=True))
# d[d.isnull()==True]


# In[ ]:


# df_account_train.customer_no == 3876


# In[ ]:


# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.min(skipna=True))
# d[d.isnull()==True]


# In[ ]:


# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))
# len(d)


# In[ ]:


# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))
# len(d)


# In[ ]:


# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))
# len(d)


# In[ ]:


# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))
# len(d)


# In[ ]:


# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))
# len(d)


# In[ ]:


# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))
# len(d)


# In[ ]:


# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))
# len(d)


# In[ ]:


# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))
# len(d)


# In[ ]:


# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))
# len(d)


# In[ ]:


# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))
# len(d)


# In[ ]:


# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))
# len(d)


# In[ ]:


# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))
# len(d)


# In[ ]:


# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))
# len(d)


# In[ ]:


# newdf['recentDPD'] = latestdpd(newdf['paymenthistory1'].values)
# newdf[newdf['customer_no'] == 10]
# newdf['recentDPD']


# In[ ]:


# newdf['recentDPD'] = latestdpd(newdf['paymenthistory1'].values)
# newdf[newdf['customer_no'] == 10]
# newdf['recentDPD']


# In[ ]:


# g1.columns


# In[ ]:


# largeGroup = df_account_train.groupby('customer_no', as_index=False).agg({'high_credit_amt': ['max','min','mean','sum']})


# In[ ]:


# d = df_account_train[df_account_train['cashlimit'].isnull()==False].groupby('customer_no').agg({'customer_no':['count']})
# len(d)


# In[ ]:


# d1 = df_account_train.groupby('customer_no').agg({'customer_no':['count']})


# In[ ]:


# d1.columns.values


# In[ ]:


# d = df_account_train[df_account_train['high_credit_amt'] == 0]


# In[ ]:


# group1 = df_account_train.groupby('customer_no', as_index=False).agg({'high_credit_amt': ['max','min','mean','sum','std'], 'cur_balance_amt': ['max','min','mean','sum','std'], 'creditlimit': ['max','min','mean','sum','std']})


# In[ ]:


# df_old = pd.read_csv(r'C:\Users\Saurabh Banga\Desktop\Tookitaki\FinalData\rashmiTrain1.csv', engine='python')


# In[ ]:


# df_old[df_old['mean_creditlimit'].isnull()==True]


# In[ ]:


# df_old.columns.values
# df_old.drop(['std_diff_curBal_highCredit','std_creditlimit','std_cur_balance_amt'],inplace=True, axis=1)


# In[ ]:


# df_old.columns.values


# In[ ]:


# group1[group1['high_credit_amt', 'std']==0].head()


# In[ ]:


# latestdpd()


# In[ ]:


# totalDpdDaysCount()


# In[ ]:


# totalDpdDays()


# In[ ]:


# totaldpdDaysDiff()


# In[ ]:


# # sum of count of number of days between two dpds
# def TwodpdDaysDiff(arr=np.array([['"""123sssttt030"""'],['"""fffhhh08989"""'],['']]),m=1,n=2):
#     value = []
#     for index, x in np.ndenumerate(arr):
#         dpdlist = []
#         count=0
#         x = x.strip('"""')
#         f = [x[i:i+3] for i in range(0,len(x),3)]
#         for i, item in enumerate(f):
#             try:
#                 val=int(item)
#                 dpdlist.append(i)
#                 continue
#             except(ValueError, Exception) as e:
#                 continue
#         try:
#             value.append((dpdlist[n-1]-dpdlist[m-1]-1)*30)
#         except IndexError as e:
#             value.append(0)
#             continue
#     print(value)


# In[ ]:


# TwodpdDaysDiff()


# In[ ]:


# if(int('gggg')):
#     print('hi')

