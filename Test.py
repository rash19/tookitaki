
# coding: utf-8

# In[1]:


import datetime
import pandas as pd
from scipy import stats
import numpy as np
import pickle


# In[2]:


# Function to compute the frequency of the dpd for a customer
def frequencyOfDPD(arr):
    value = []
    count=0
    for index, x in np.ndenumerate(arr):
        dpdlist=[]
        x = x.strip('"""')
        f = [x[i:i+3] for i in range(0,len(x),3)]
        for each in f:
            try:
                dpdlist.append(int(each))
                continue
            except(ValueError, Exception) as e:
                continue
        if dpdlist:
            value.append(len(dpdlist))
        else:
            value.append(0)
    return value


# In[3]:


# Return n recent dpd from payment history. If none then return the sum of all dpd for one history.
def latestNdpd(arr,freq=None):
    value = []
    for index, x in np.ndenumerate(arr):
        dpdlist=[]
        count=0
        x = x.strip('"""')
        f = [x[i:i+3] for i in range(0,len(x),3)]
        for each in f:
            try:
                dpdlist.append(int(each))
                continue
            except(ValueError, Exception) as e:
                dpdlist.append(0)
                continue
        if dpdlist:
            if freq==None:
                for dpd in dpdlist:
                    count+=dpd
                value.append(count)
            else:
                try:
                    if dpdlist[freq-1]!=0:
                        value.append(1)
                        continue
                    else:
                        value.append(0)
                except(IndexError) as e:
                    value.append(0)
                    continue
        else:
            value.append(0)
    return value


# In[4]:


def countCredit(x):
    count=0
    for index, item in np.ndenumerate(x.values):
        if item in [10,35,14,31]:
            count+=1
    return count/len(x)


# In[5]:


def countAccount(x):
    count=0
    for index, item in np.ndenumerate(x.values):
        if item in [1,2,3,4,5,6,7,8,9,13,15,17,32,33,34,41,41,42,51,54,59,60]:
            count+=1
    return count/len(x)


# In[6]:


def paymthistorylength(x):
    count=0
    for index, str in np.ndenumerate(x):
        str = str.strip('"""')
        count+=len(str)
    return count/len(x)


# In[7]:


def findMode(arr):
    return (stats.mode(arr)).mode


# In[8]:


df_data = pd.read_csv(r'C:\Users\Dell\Desktop\raw_data_30_new.csv',engine='python')
df_account = pd.read_csv(r'C:\Users\Dell\Desktop\raw_account_30_new.csv',engine='python', parse_dates = ['dt_opened','upload_dt', 'opened_dt', 'paymt_str_dt', 'paymt_end_dt','last_paymt_dt','closed_dt','reporting_dt'])
df_enquiry = pd.read_csv(r'C:\Users\Dell\Desktop\raw_enquiry_30_new.csv',engine='python', parse_dates = ['dt_opened','upload_dt', 'enquiry_dt'])


# In[9]:


newdf = df_account
newdf1 = df_enquiry
# df_enquiry.isnull().sum(axis=0)


# In[10]:


#  Assuming current date for null last_paymt_dt.
newdf.last_paymt_dt.fillna(datetime.datetime.now(),inplace=True)


# In[11]:


# Assuming value of 0.0001 for null high_credit_amt and then high_credit_amt is assumed for null creditlimit.
# newdf.high_credit_amt.fillna(0.0001, inplace=True)
# newdf.creditlimit.fillna(newdf.high_credit_amt, inplace=True)


# In[12]:


newdf['diff_opened_lastPaymt_dt'] = newdf['opened_dt'].sub(newdf['last_paymt_dt'], axis=0)
newdf['diff_opened_lastPaymt_dt'] = newdf['diff_opened_lastPaymt_dt'] / np.timedelta64(1, 'D')
# newdf.isnull().sum(axis=0)


# In[13]:


# ----------------------------------------------------Deriving Account Features--------------------------------------------------------
newdf['highcr-creditlimt'] = newdf['high_credit_amt'] - newdf['creditlimit']
newdf['currBal-highcr'] = newdf['cur_balance_amt'] - newdf['high_credit_amt']
newdf['currbal-creditlimit'] = newdf['cur_balance_amt'] - newdf['creditlimit']
# newdf.isnull().sum(axis=0)


# In[14]:


newdf['1DPDReported'] = latestNdpd(newdf['paymenthistory1'].values, freq=1)
newdf['2DPDReported'] = newdf['1DPDReported'] | latestNdpd(newdf['paymenthistory1'].values, freq=2)
newdf['3DPDReported'] = newdf['2DPDReported'] | latestNdpd(newdf['paymenthistory1'].values, freq=3)
newdf['totalDPD'] = latestNdpd(newdf['paymenthistory1'].values)
newdf['frequencyofDPDreported'] = frequencyOfDPD(newdf['paymenthistory1'].values)
# newdf.isnull().sum(axis=0)


# In[15]:


group1 = newdf.groupby('customer_no', as_index=False).agg({'acct_type': ['count'],'diff_opened_lastPaymt_dt': ['mean','sum'],'highcr-creditlimt': ['mean','sum'],'currBal-highcr': ['mean','sum'], 'currbal-creditlimit': ['mean','sum'], '1DPDReported': ['mean','sum'], '2DPDReported': ['mean','sum'], '3DPDReported': ['mean','sum'], 'totalDPD': ['mean','sum'], 'frequencyofDPDreported': ['mean','sum'], 'cur_balance_amt': ['mean','sum','std','count'], 'creditlimit': ['mean','sum','std'], 'high_credit_amt': ['mean','sum','std']})
# group1.columns.values


# In[16]:


df_derived = pd.DataFrame()
df_derived['customer_no'] = group1['customer_no']


# In[17]:


df_derived['diff_opened_lastPaymt_dt_sum'] = group1[('diff_opened_lastPaymt_dt','sum')]
# df_derived['diff_highcr_creditlim_sum'] = d1[('highcr-creditlimt','sum')]
df_derived['diff_highcr_creditlim_mean'] = group1[('highcr-creditlimt','mean')]
# df_derived['diff_highcr_currbal_sum'] = d3[('highcr-currBal','sum')]
df_derived['1DPDReported_mean'] = group1[('1DPDReported','mean')]
df_derived['2DPDReported_mean'] = group1[('2DPDReported','mean')]
df_derived['3DPDReported_mean'] = group1[('3DPDReported','mean')]
df_derived['totalDPD_sum'] = group1[('3DPDReported','sum')]
df_derived['frequencyofDPDreported_mean'] = group1[('frequencyofDPDreported','mean')]

df_derived['ratio_totalCurrbal_totalcrlim'] = group1[('cur_balance_amt','sum')]/group1[('creditlimit','sum')]
df_derived['ratio_totalCurrbal_totalhighCr'] = group1[('cur_balance_amt','sum')]/group1[('high_credit_amt','sum')]
df_derived['ratio_totalhighcr_totalcrlim'] = group1[('high_credit_amt','sum')]/group1[('creditlimit','sum')]
# df_derived.isnull().sum(axis=0)


# In[18]:


d1 = newdf.groupby('customer_no',as_index=False)['acct_type'].apply(countCredit).reset_index(name='creditcount')
df_derived['avg_creditcount'] = d1.creditcount


# In[19]:


d1 = newdf.groupby('customer_no',as_index=False)['acct_type'].apply(countAccount).reset_index(name='loancount')
df_derived['avg_loancount'] = d1.loancount/group1[('acct_type','count')]


# In[20]:


d1 = newdf.groupby('customer_no',as_index=False)['paymenthistory1'].apply(paymthistorylength).reset_index(name='historylength')
df_derived['avg_payhistlength'] = d1.historylength


# In[21]:


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
d1['meancount'] = d1['count']/group1[('acct_type','count')]
df_derived['meanAcctWithHighCrGreaterThanCreditLim'] = d1['meancount']


# In[22]:


d_1 = newdf[newdf['cur_balance_amt'] == 0]
g_1 = d_1.groupby('customer_no').size().reset_index(name='count')
g_2 = newdf.groupby('customer_no').size().reset_index(name='count')
l = (g_2[~g_2['customer_no'].isin(g_1['customer_no'])==True].customer_no).tolist()
d = pd.DataFrame()
d['customer_no'] = l
d['count'] = [0 for i in range(len(d))]
d1 = pd.concat([g_1, d], ignore_index=True)
d1.sort_values('customer_no',inplace=True)
d1.reset_index(drop=True,inplace=True)
d1['meancount'] = d1['count']/group1[('acct_type','count')]
# df_derived['meanAcctWithCurrBalEqualsZero'] = d1['meancount']
df_derived['meanAcctWithCurrBalEqualsZero'] = d1['count']


# In[23]:


#--------------------------------------------Deriving Enquiry Features-----------------------------------------------------------
d1 = newdf1.enquiry_dt.values
d2 = d1[1:]
d2 = np.append(d2,d1[-1])
d = (d2-d1)/np.timedelta64(1, 'D')
newdf1['GapEnquiryDates'] = abs(d)


# In[24]:


group2 = newdf1.groupby('customer_no', as_index=False).agg({'dt_opened': ['count'],'upload_dt': ['count'],'enquiry_dt': ['count'],'enq_purpose': ['count'], 'enq_amt': ['count'],'enq_purpose': findMode,'GapEnquiryDates': ['mean','sum']})


# In[25]:


df_derived['meanGapEnquiryDates'] = group2[('GapEnquiryDates','mean')]


# In[26]:


def findMode(arr):
    return (stats.mode(arr)).mode


# In[27]:


df_derived['mostFrequentEnquiryPorpose'] = group2[('enq_purpose','findMode')]


# In[28]:


values = []
for i in group2.customer_no:
    values.append(newdf1[newdf1.customer_no==i].enq_purpose.values[0])
df_derived['mostRecentEnquiryPorpose'] = values


# In[29]:


df_derived.replace(np.nan,0,inplace=True)


# In[30]:


newdf2 = df_data.select_dtypes(['float64', 'int64', 'float32','int32'])
for col in newdf2.columns.values.tolist():
    df_derived[col] = newdf2[col]


# In[31]:


# df_derived.replace(np.nan,0.000,inplace=True)
# df_derived.replace(np.nan,0,inplace=True)
# df_derived.replace(np.nan,0,inplace=True)
df_data.to_csv(r'C:\Users\Dell\Desktop\Test.csv')


# In[32]:


len(df_derived.columns.values)


# In[33]:


# df_derived.isnull().sum(axis=0)

