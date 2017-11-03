{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import datetime\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Count total number of dpds from payment history\n",
    "def tototalDpdDaysCount(arr):\n",
    "# def totalDpdDaysCount(arr=np.array([['\"\"\"123sssttt300\"\"\"'],['\"\"\"fffhhh089\"\"\"'],['']]),m=None):\n",
    "    value = []\n",
    "    for index, x in np.ndenumerate(arr):\n",
    "        count=0\n",
    "        x = x.strip('\"\"\"')\n",
    "        f = [x[i:i+3] for i in range(0,len(x),3)]\n",
    "        for item in f:\n",
    "            try:\n",
    "                val=int(item)\n",
    "                count+=1\n",
    "                continue\n",
    "            except(ValueError, Exception) as e:\n",
    "                count+=0\n",
    "                continue\n",
    "        value.append(count)\n",
    "    return value\n",
    "\n",
    "# Sum of num of days past between two dpds.\n",
    "def totaldpdDaysDiff(arr):\n",
    "# def totaldpdDaysDiff(arr=np.array([['\"\"\"STDSTDSTDXXXXXXXXXXXXXXXSTDXXXXXXXXXXXXXXXSTD\"\"\"'],['\"\"\"fffhhh089\"\"\"'],['ffffff']])):\n",
    "    value = []\n",
    "    for index, x in np.ndenumerate(arr):\n",
    "        dpdlist = []\n",
    "        count=0\n",
    "        x = x.strip('\"\"\"')\n",
    "        f = [x[i:i+3] for i in range(0,len(x),3)]\n",
    "        for i, item in enumerate(f):\n",
    "            try:\n",
    "                val=int(item)\n",
    "                dpdlist.append(i)\n",
    "                continue\n",
    "            except(ValueError, Exception) as e:\n",
    "                continue\n",
    "        k = [(dpdlist[i+1]-dpdlist[i]-1)*30 for i in range(len(dpdlist)-1)]\n",
    "        if k:\n",
    "            value.append(k[0])\n",
    "        else:\n",
    "            value.append(0)\n",
    "    return(value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Return Most recent dpd from payment history\n",
    "def latestdpd(arr):\n",
    "# def latestdpd(arr=np.array([['\"\"\"000150150150150150120090090060030015000015000000000000\"\"\"'],['\"\"\"fffhhh089\"\"\"'],['']]),m=None):\n",
    "    value = []\n",
    "    count=0\n",
    "    for index, x in np.ndenumerate(arr):\n",
    "        x = x.strip('\"\"\"')\n",
    "        f = [x[i:i+3] for i in range(0,len(x),3)]\n",
    "        try:\n",
    "            count=int(f[1])\n",
    "#             print(count)\n",
    "            value.append(count)\n",
    "            continue\n",
    "        except(ValueError, Exception) as e:\n",
    "            count=0\n",
    "            value.append(count)\n",
    "            continue\n",
    "    return value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Count total number of dpds in days from payment history\n",
    "def totalDpdDays(arr):\n",
    "# def totalDpdDays(arr=np.array([['\"\"\"123sssttt300\"\"\"'],['\"\"\"fffhhh089\"\"\"'],['']]),m=None):\n",
    "    value = []\n",
    "    for index, x in np.ndenumerate(arr):\n",
    "        count=0\n",
    "        x = x.strip('\"\"\"')\n",
    "        f = [x[i:i+3] for i in range(0,len(x),3)]\n",
    "        for item in f:\n",
    "            try:\n",
    "                val=int(item)\n",
    "                count+=val\n",
    "                continue\n",
    "            except(ValueError, Exception) as e:\n",
    "                count+=0\n",
    "                continue\n",
    "        value.append(count)\n",
    "    return value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "df_data_train = pd.read_csv(r'C:\\Users\\Saurabh Banga\\Desktop\\Tookitaki\\test data\\raw_data_70_new.csv',engine='python')\n",
    "df_account_train = pd.read_csv(r'C:\\Users\\Saurabh Banga\\Desktop\\Tookitaki\\test data\\raw_account_70_new.csv',engine='python', parse_dates = ['dt_opened','upload_dt', 'opened_dt', 'paymt_str_dt', 'paymt_end_dt','last_paymt_dt','closed_dt','reporting_dt'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "newdf = df_account_train\n",
    "newdf1 = df_data_train\n",
    "col_null_count = newdf.isnull().sum(axis=0)\n",
    "# col_null_count"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# newdf['new_closed_dt'] = newdf['closed_dt']\n",
    "# newdf.new_closed_dt.fillna(datetime.datetime.now(),inplace=True)\n",
    "newdf.last_paymt_dt.fillna(datetime.datetime.now(),inplace=True)\n",
    "newdf.opened_dt.fillna(datetime.datetime.now(),inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# newdf.loc[(newdf.opened_dt.isnull()==True) & (newdf.last_paymt_dt.isnull()==False), 'customer_no'].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "newdf.creditlimit.fillna(newdf.high_credit_amt, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "h_1 = newdf.groupby('customer_no', as_index=False).agg({'high_credit_amt': ['count','mean']})\n",
    "l_1 = (h_1[h_1[('high_credit_amt','count')] == 0].customer_no).tolist()\n",
    "for id in l_1:\n",
    "    newdf.loc[newdf['customer_no'] == id,'high_credit_amt'] = 0.1\n",
    "    \n",
    "h_1 = newdf.groupby('customer_no', as_index=False).agg({'cur_balance_amt': ['count','mean']})\n",
    "l_1 = (h_1[h_1[('cur_balance_amt','count')] == 0].customer_no).tolist()\n",
    "for id in l_1:\n",
    "    newdf.loc[newdf['customer_no'] == id,'cur_balance_amt'] = 0.1\n",
    "\n",
    "h_1 = newdf.groupby('customer_no', as_index=False).agg({'creditlimit': ['count','mean']})\n",
    "l_1 = (h_1[h_1[('creditlimit','count')] == 0].customer_no).tolist()\n",
    "for id in l_1:\n",
    "    newdf.loc[newdf['customer_no'] == id,'creditlimit'] = 0.1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# l = [3876,5822,10211,11850,15270,16373,16750,18722,19318,19543,20867,22453,22788]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "loan_account_list = [1,2,3,4,5,6,7,8,9,13,15,17,32,33,34,41,41,42,51,54,59,60]\n",
    "card_list = [10,14,31,35]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "newdf['diff_opened_lastPaymt_dt'] = newdf['opened_dt'].sub(newdf['last_paymt_dt'], axis=0)\n",
    "newdf['diff_opened_lastPaymt_dt'] = newdf['diff_opened_lastPaymt_dt'] / np.timedelta64(1, 'D')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "newdf['highcr-creditlimt'] = newdf['high_credit_amt'] - newdf['creditlimit']\n",
    "newdf['highcr-currBal'] = newdf['high_credit_amt'] - newdf['cur_balance_amt']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "newdf['latestDPD'] = latestdpd(newdf['paymenthistory1'].values)\n",
    "# newdf['recentDPD'].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "newdf['totalDPDdays'] = totalDpdDays(newdf['paymenthistory1'].values)\n",
    "# newdf['totalDPDdays'].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "newdf['dayspastDPD'] = totaldpdDaysDiff(newdf['paymenthistory1'].values)\n",
    "# newdf['dayspastDPD'].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "group1 = newdf.groupby('customer_no', as_index=False).agg({'dayspastDPD': ['mean','sum'],'latestDPD': ['mean','sum'],'totalDPDdays': ['mean','sum']})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# group1.isnull().sum(axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "23896"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_derived = pd.DataFrame()\n",
    "df_derived['customer_no'] = group1['customer_no']\n",
    "len(df_derived)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "d1 = newdf.groupby('customer_no', as_index=False).agg({'highcr-creditlimt': ['sum']})\n",
    "\n",
    "d2 = newdf.groupby('customer_no', as_index=False).agg({'highcr-creditlimt': ['mean']})\n",
    "\n",
    "d3 = newdf.groupby('customer_no', as_index=False).agg({'highcr-currBal': ['sum']})\n",
    "\n",
    "d4 = newdf.groupby('customer_no', as_index=False).agg({'highcr-currBal': ['mean']})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "df_derived['diff_highcr_creditlim_sum'] = d1[('highcr-creditlimt','sum')]\n",
    "df_derived['diff_highcr_creditlim_mean'] = d2[('highcr-creditlimt','mean')]\n",
    "df_derived['diff_highcr_currbal_sum'] = d3[('highcr-currBal','sum')]\n",
    "df_derived['diff_highcr_currbal_mean'] = d4[('highcr-currBal','mean')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "d1 = newdf.groupby('customer_no',as_index=False)['high_credit_amt'].apply(lambda x: x.sum(skipna=True))\n",
    "\n",
    "d2 = newdf.groupby('customer_no',as_index=False)['high_credit_amt'].apply(lambda x: x.mean(skipna=True))\n",
    "\n",
    "d3 = newdf.groupby('customer_no',as_index=False)['creditlimit'].apply(lambda x: x.sum(skipna=True))\n",
    "\n",
    "d4 = newdf.groupby('customer_no',as_index=False)['creditlimit'].apply(lambda x: x.mean(skipna=True))\n",
    "\n",
    "d5 = newdf.groupby('customer_no', as_index=False).agg({'cur_balance_amt': ['sum']})\n",
    "\n",
    "d6 = newdf.groupby('customer_no', as_index=False).agg({'cur_balance_amt': ['mean']})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "df_derived['high_credit_amt_mean'] = d2\n",
    "df_derived['creditlimit_mean'] = d4\n",
    "df_derived['cur_balance_amt_mean'] = d6[('cur_balance_amt','mean')]\n",
    "df_derived['ratio_totalhighcr_totalcrlim'] = d1/d3\n",
    "df_derived['ratio_totalCurrbal_totalcrlim'] = d5[('cur_balance_amt','sum')]/d3\n",
    "df_derived['utilisationTrend'] = df_derived['ratio_totalCurrbal_totalcrlim']/d4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "df_derived.replace(np.nan,0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "d1 = newdf.groupby('customer_no',as_index=False)['diff_opened_lastPaymt_dt'].apply(lambda x: x.sum(skipna=True))\n",
    "df_derived['sum_diff_opened_lastPaymt_dt'] = d1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def countCredit(x):\n",
    "    count=0\n",
    "    for index, item in np.ndenumerate(x.values):\n",
    "        if item in [10,35,14,31]:\n",
    "            count+=1\n",
    "    return count/len(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def countAccount(x):\n",
    "    count=0\n",
    "    for index, item in np.ndenumerate(x.values):\n",
    "        if item in [1,2,3,4,5,6,7,8,9,13,15,17,32,33,34,41,41,42,51,54,59,60]:\n",
    "            count+=1\n",
    "    return count/len(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def paymthistorylength(x):\n",
    "    count=0\n",
    "    for index, str in np.ndenumerate(x):\n",
    "        str = str.strip('\"\"\"')\n",
    "        count+=len(str)\n",
    "    return count/len(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "d1 = newdf.groupby('customer_no',as_index=False)['acct_type'].apply(countCredit).reset_index(name='creditcount')\n",
    "df_derived['avg_creditcount'] = d1.creditcount"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "d1 = newdf.groupby('customer_no',as_index=False)['acct_type'].apply(countAccount).reset_index(name='loancount')\n",
    "df_derived['avg_loancount'] = d1.loancount"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "d1 = newdf.groupby('customer_no',as_index=False)['paymenthistory1'].apply(paymthistorylength).reset_index(name='historylength')\n",
    "df_derived['avg_payhistlength'] = d1.historylength"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "collapsed": false,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# df_derived[df_derived['cur_balance_amt_mean'].isnull()==True]\n",
    "# newdf.columns.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "activeaccount = newdf.ix[(newdf['latestDPD'] > 0)]\n",
    "group2 = activeaccount.groupby('customer_no').size().reset_index(name='allaccountdpd')\n",
    "l = (group1[~group1['customer_no'].isin(group2['customer_no'])==True].customer_no).tolist()\n",
    "d = pd.DataFrame()\n",
    "d['customer_no'] = l\n",
    "d['allaccountdpd'] = [0 for i in range(len(d))]\n",
    "d1 = pd.concat([d, group2], ignore_index=True)\n",
    "d1.sort_values('customer_no',inplace=True)\n",
    "d2 = d1.reset_index(drop=True)\n",
    "df_derived['allaccountdpd'] = d2['allaccountdpd']\n",
    "# len(d2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "invalid type comparison",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-36-09c35f5a6de7>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0minactiveaccount\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnewdf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mix\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mnewdf\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'last_paymt_dt'\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m>\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m&\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mnewdf\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'diff_reporting_closed_dt'\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m<\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0mgroup2\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0minactiveaccount\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mgroupby\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'customer_no'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msize\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mreset_index\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mname\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'inactiveaccountdpd'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0ml\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mgroup1\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m~\u001b[0m\u001b[0mgroup1\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'customer_no'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0misin\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mgroup2\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'customer_no'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m==\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcustomer_no\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtolist\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0md\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mDataFrame\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0md\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'customer_no'\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0ml\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mC:\\Users\\Saurabh Banga\\Anaconda3\\lib\\site-packages\\pandas\\core\\ops.py\u001b[0m in \u001b[0;36mwrapper\u001b[0;34m(self, other, axis)\u001b[0m\n\u001b[1;32m    761\u001b[0m                 \u001b[0mother\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mother\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m    762\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m--> 763\u001b[0;31m             \u001b[0mres\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mna_op\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mvalues\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mother\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    764\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0misscalar\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mres\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m    765\u001b[0m                 raise TypeError('Could not compare %s type with Series' %\n",
      "\u001b[0;32mC:\\Users\\Saurabh Banga\\Anaconda3\\lib\\site-packages\\pandas\\core\\ops.py\u001b[0m in \u001b[0;36mna_op\u001b[0;34m(x, y)\u001b[0m\n\u001b[1;32m    690\u001b[0m             \u001b[1;31m# integer comparisons\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m    691\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mis_datetimelike_v_numeric\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m--> 692\u001b[0;31m                 \u001b[1;32mraise\u001b[0m \u001b[0mTypeError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"invalid type comparison\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    693\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m    694\u001b[0m             \u001b[1;31m# numpy does not like comparisons vs None\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mTypeError\u001b[0m: invalid type comparison"
     ]
    }
   ],
   "source": [
    "inactiveaccount = newdf.ix[(newdf['last_paymt_dt'] > 0) & (newdf['diff_reporting_closed_dt'] < 0)]\n",
    "group2 = inactiveaccount.groupby('customer_no').size().reset_index(name='inactiveaccountdpd')\n",
    "l = (group1[~group1['customer_no'].isin(group2['customer_no'])==True].customer_no).tolist()\n",
    "d = pd.DataFrame()\n",
    "d['customer_no'] = l\n",
    "d['inactiveaccountdpd'] = [0 for i in range(len(d))]\n",
    "d1 = pd.concat([d, group2], ignore_index=True)\n",
    "d1.sort_values('customer_no',inplace=True)\n",
    "d2 = d1.reset_index(drop=True)\n",
    "df_derived['inactiveaccountdpd'] = d2['inactiveaccountdpd']\n",
    "# len(d2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "df_derived.isnull().sum(axis=0)\n",
    "df_derived.replace(np.nan,0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "df_derived['meanCurBal_by_meanCreditLim'] = df_derived['cur_balance_amt_mean']/df_derived['creditlimit_mean']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "df_derived.isnull().sum(axis=0)\n",
    "df_derived.replace(np.nan,0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "df_derived['meanHighCrAmt_by_meanCreditLim'] = df_derived['high_credit_amt_mean']/df_derived['creditlimit_mean']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "df_derived.isnull().sum(axis=0)\n",
    "df_derived.replace(np.nan,0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "df_derived['meanHighCrAmt_by_meanCurBal'] = df_derived['high_credit_amt_mean']/df_derived['cur_balance_amt_mean']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# df_derived.isnull().sum(axis=0)\n",
    "# df_derived.replace(np.nan,0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "df_derived['latestDPD_sum'] = group1[('latestDPD', 'sum')]\n",
    "df_derived['latestDPD_mean'] = group1[('latestDPD', 'mean')]\n",
    "# df_derived['latestDPD_min'] = group1[('latestDPD', 'min')]\n",
    "# df_derived['latestDPD_max'] = group1[('latestDPD', 'max')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "df_derived['dayspastDPD_sum'] = group1[('dayspastDPD', 'sum')]\n",
    "df_derived['dayspastDPD_mean'] = group1[('dayspastDPD', 'mean')]\n",
    "# df_derived['dayspastDPD_min'] = group1[('dayspastDPD', 'min')]\n",
    "# df_derived['dayspastDPD_max'] = group1[('dayspastDPD', 'max')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "df_derived['totalDPDdays_sum'] = group1[('totalDPDdays', 'sum')]\n",
    "df_derived['totalDPDdays_mean'] = group1[('totalDPDdays', 'mean')]\n",
    "# df_derived['totalDPDdays_min'] = group1[('totalDPDdays', 'min')]\n",
    "# df_derived['totalDPDdays_max'] = group1[('totalDPDdays', 'max')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "df_derived.isnull().sum(axis=0)\n",
    "# df_derived.replace(np.nan,0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "d_1 = newdf[newdf['high_credit_amt'] > newdf['cur_balance_amt']]\n",
    "g_1 = d_1.groupby('customer_no').size().reset_index(name='count')\n",
    "g_2 = newdf.groupby('customer_no').size().reset_index(name='count')\n",
    "l = (g_2[~g_2['customer_no'].isin(g_1['customer_no'])==True].customer_no).tolist()\n",
    "d = pd.DataFrame()\n",
    "d['customer_no'] = l\n",
    "d['count'] = [0 for i in range(len(d))]\n",
    "d1 = pd.concat([g_1, d], ignore_index=True)\n",
    "d1.sort_values('customer_no',inplace=True)\n",
    "d1.reset_index(drop=True,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# df_derived.isnull().sum(axis=0)\n",
    "# df_derived.replace(np.nan,0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "df_derived['expected_highCr_currBal'] = d1['count']*(d1['count']/g_2['count'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "df_derived.isnull().sum(axis=0)\n",
    "# df_derived.replace(np.nan,0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "d_1 = newdf[newdf['cur_balance_amt'] > 0]\n",
    "g_1 = d_1.groupby('customer_no').size().reset_index(name='count')\n",
    "g_2 = newdf.groupby('customer_no').size().reset_index(name='count')\n",
    "l = (g_2[~g_2['customer_no'].isin(g_1['customer_no'])==True].customer_no).tolist()\n",
    "d = pd.DataFrame()\n",
    "d['customer_no'] = l\n",
    "d['count'] = [0 for i in range(len(d))]\n",
    "d1 = pd.concat([g_1, d], ignore_index=True)\n",
    "d1.sort_values('customer_no',inplace=True)\n",
    "d1.reset_index(drop=True,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "df_derived['expected_zeroCurrBal'] = d1['count']*(d1['count']/g_2['count'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "df_derived.isnull().sum(axis=0)\n",
    "# df_derived.replace(np.nan,0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "d_1 = newdf[newdf['high_credit_amt'] > newdf['creditlimit']]\n",
    "g_1 = d_1.groupby('customer_no').size().reset_index(name='count')\n",
    "g_2 = newdf.groupby('customer_no').size().reset_index(name='count')\n",
    "l = (g_2[~g_2['customer_no'].isin(g_1['customer_no'])==True].customer_no).tolist()\n",
    "d = pd.DataFrame()\n",
    "d['customer_no'] = l\n",
    "d['count'] = [0 for i in range(len(d))]\n",
    "d1 = pd.concat([g_1, d], ignore_index=True)\n",
    "d1.sort_values('customer_no',inplace=True)\n",
    "d1.reset_index(drop=True,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "df_derived['expected_highCr_creditLim'] = d1['count']*(d1['count']/g_2['count'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "df_derived.isnull().sum(axis=0)\n",
    "# df_derived.replace(np.nan,0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "df_derived['Class'] = newdf1['Bad_label']\n",
    "df_derived.to_csv(r'C:\\Users\\Saurabh Banga\\Desktop\\Tookitaki\\FinalData\\Train2.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "with open(r'C:\\Users\\Saurabh Banga\\Desktop\\Tookitaki\\FinalData\\Train2','wb') as f:\n",
    "    pickle.dump((df_derived.columns.difference(['customer_no'])).values,f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "df_derived.columns.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "df_derived.isnull().sum(axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "print(len(df_data_train),len(df_derived))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# dff = pd.read_csv(r'C:\\Users\\Saurabh Banga\\Desktop\\Tookitaki\\FinalData\\Train2.csv', engine='python')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# dff[(dff.Class).isnull()==True].customer_no"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# sum(axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# s2 = group1['customer_no']\n",
    "# s2\n",
    "# s = list(s2-s1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# newdf[newdf['customer_no'].isin(s)]."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.std(skipna=True))\n",
    "# d[d.isnull()==True]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# df_account_train.customer_no == 3876"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.min(skipna=True))\n",
    "# d[d.isnull()==True]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))\n",
    "# len(d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))\n",
    "# len(d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))\n",
    "# len(d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))\n",
    "# len(d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))\n",
    "# len(d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))\n",
    "# len(d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))\n",
    "# len(d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))\n",
    "# len(d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))\n",
    "# len(d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))\n",
    "# len(d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))\n",
    "# len(d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))\n",
    "# len(d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train.groupby('customer_no')['high_credit_amt'].apply(lambda x: x.sum(skipna=True))\n",
    "# len(d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# newdf['recentDPD'] = latestdpd(newdf['paymenthistory1'].values)\n",
    "# newdf[newdf['customer_no'] == 10]\n",
    "# newdf['recentDPD']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# newdf['recentDPD'] = latestdpd(newdf['paymenthistory1'].values)\n",
    "# newdf[newdf['customer_no'] == 10]\n",
    "# newdf['recentDPD']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# g1.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# largeGroup = df_account_train.groupby('customer_no', as_index=False).agg({'high_credit_amt': ['max','min','mean','sum']})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train[df_account_train['cashlimit'].isnull()==False].groupby('customer_no').agg({'customer_no':['count']})\n",
    "# len(d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d1 = df_account_train.groupby('customer_no').agg({'customer_no':['count']})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d1.columns.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# d = df_account_train[df_account_train['high_credit_amt'] == 0]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# group1 = df_account_train.groupby('customer_no', as_index=False).agg({'high_credit_amt': ['max','min','mean','sum','std'], 'cur_balance_amt': ['max','min','mean','sum','std'], 'creditlimit': ['max','min','mean','sum','std']})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# df_old = pd.read_csv(r'C:\\Users\\Saurabh Banga\\Desktop\\Tookitaki\\FinalData\\rashmiTrain1.csv', engine='python')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# df_old[df_old['mean_creditlimit'].isnull()==True]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# df_old.columns.values\n",
    "# df_old.drop(['std_diff_curBal_highCredit','std_creditlimit','std_cur_balance_amt'],inplace=True, axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# df_old.columns.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# group1[group1['high_credit_amt', 'std']==0].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# latestdpd()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# totalDpdDaysCount()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# totalDpdDays()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# totaldpdDaysDiff()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# # sum of count of number of days between two dpds\n",
    "# def TwodpdDaysDiff(arr=np.array([['\"\"\"123sssttt030\"\"\"'],['\"\"\"fffhhh08989\"\"\"'],['']]),m=1,n=2):\n",
    "#     value = []\n",
    "#     for index, x in np.ndenumerate(arr):\n",
    "#         dpdlist = []\n",
    "#         count=0\n",
    "#         x = x.strip('\"\"\"')\n",
    "#         f = [x[i:i+3] for i in range(0,len(x),3)]\n",
    "#         for i, item in enumerate(f):\n",
    "#             try:\n",
    "#                 val=int(item)\n",
    "#                 dpdlist.append(i)\n",
    "#                 continue\n",
    "#             except(ValueError, Exception) as e:\n",
    "#                 continue\n",
    "#         try:\n",
    "#             value.append((dpdlist[n-1]-dpdlist[m-1]-1)*30)\n",
    "#         except IndexError as e:\n",
    "#             value.append(0)\n",
    "#             continue\n",
    "#     print(value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# TwodpdDaysDiff()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# if(int('gggg')):\n",
    "#     print('hi')"
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [default]",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
