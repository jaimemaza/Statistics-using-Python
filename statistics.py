#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as mp
from scipy import special
from scipy import stats
from scipy.stats import binom
from pandas.api.types import is_numeric_dtype
from statsmodels.formula.api import ols


# In[2]:


fileName = "Police_Sentiment_Scores.csv"
def readFile(fileName):
    descriptiveData = pd.read_csv(fileName, usecols = ['AREA','SAFETY','TRUST','S_SEX_FEMALE','S_SEX_MALE'])
    return descriptiveData


# In[15]:


def descriptiveStatistics(descriptiveData):
    print("\nDescriptive statistics of how safe and trust worthy the people of Chicago feel in their areas(S_SEX_FEMALE = safety for females, S_SEX_MALE = safety for males):")
    for column in descriptiveData:
        if is_numeric_dtype(descriptiveData[column]):
            mean = descriptiveData[column].mean()
            median = descriptiveData[column].median()
            mode = descriptiveData[column].mode()[0]
            stdev = descriptiveData[column].std()
            variance = descriptiveData[column].var()
            minimum = descriptiveData[column].min()
            maximum = descriptiveData[column].max()
            therange = maximum - minimum
            print(f"\n\t{column}:")
            print(f"\t\t Mean = {mean:.2f}")
            print(f"\t\t Median = ", median)
            print(f"\t\t Mode = ", mode)
            print(f"\t\t Standart Deviation = {stdev:.2f}")
            print(f"\t\t Variance = {variance:.2f}")
            print(f"\t\t Range = {therange:.2f}")  
    
    print("\nMean safety and trust of each area of Chicago: ")
            
    mean2 = descriptiveData.groupby(['AREA'])[["SAFETY", "TRUST"]].mean()
    safest = mean2["SAFETY"].idxmax()
    trustest = mean2["TRUST"].idxmax()
    lessSafe = mean2["SAFETY"].idxmin()
    lessTrust = mean2["TRUST"].idxmin()
    print("\n",mean2)
    print("\nAfter analyze this table we can conclude that the safest area in Chicago is", safest, "and the most trusted is", trustest)
    print("However, we can also conclude that the least safest area is", lessSafe, "and the least trusted area is", lessTrust)


# In[16]:


def probablity(descriptiveData):
    for column in descriptiveData:
        if is_numeric_dtype(descriptiveData[column]):
            meanSafe = descriptiveData['SAFETY'].mean()
            meanTrust = descriptiveData['TRUST'].mean()            
            stdevSafe = descriptiveData['SAFETY'].std() 
            stdevTrust = descriptiveData['TRUST'].std()
    probSafe1 = stats.norm(meanSafe, stdevSafe)
    probSafe = probSafe1.cdf(75)
    probTrust1 = stats.norm(meanTrust, stdevTrust)
    probTrust = probTrust1.cdf(55)
    print(f"\nThe probability that a person has rated the trust of their area with less than 55 is:  {probTrust:.3f}")
    n = 100
    p = probSafe
    k = 100
    binomial = stats.binom.pmf(k,n,p)
    print(f"\nIf we ask 100 people from Chicago how safety they feel about their area, what is the probability that all of them vote less than 75? {binomial:.3f}")
    
    


# In[17]:


def hypothesis(descriptiveData):
    confidenceLevel = 0.95
    significanceLevel = 1- confidenceLevel
    
    sampleSize = len(descriptiveData)
    listData =[]
    numTotal = 0
    listData = descriptiveData['SAFETY'].tolist()
    for result in listData:
        if result>=70 and result <= 75 :
            numTotal += 1
    pHat = float(numTotal/sampleSize)
    
    pClaim = 0.06
    qClaim = 1-pClaim
    
    import math 
    zScore = (pHat - pClaim)/math.sqrt(pClaim*qClaim/sampleSize)
    
    import scipy.stats
    
    if zScore < 0:
        pValue = scipy.stats.norm.sf(abs(zScore))*2
        altPValue = (1 - scipy.stats.norm.cdf(abs(zScore)))*2
    else:
        altPValue = scipy.stats.norm.cdf(abs(zScore))*2
        
    print("\nClaim: 6% of the people of Chicago responded in the survey that their level of safety on their area is between 70 and 75")    
    print("\nNull hypothesis is that the true proportion is 6%")
    print("Alternative hypothesis is that the true proportion is <> 6%")
    
    
    print(f"\nGiven a z Score of {zScore: .4f} and ")
    print(f"a p-value of {pValue: .4f}")                
    print(f"a differently calculated p-value of {altPValue: .4f}")
    
    reject = pValue<=significanceLevel

    if reject:
        textReject = "reject"
    else:
        textReject ="fail to reject"

    print(f"We {textReject} the Null Hypothesis of p = {pClaim:.2f}")
    


# In[18]:


df = pd.DataFrame(['https://catalog.data.gov/dataset/police-sentiment-scores'])

def make_clickable(val):
    print("\n \nThe data used for this project has been collected from the next link:")
    return '<a href="{}">{}</a>'.format(val,val)


# In[19]:


def anova_test(descriptiveData):
    mod = ols('SAFETY ~ S_SEX_FEMALE', data=descriptiveData).fit()
    aov_table = mp.stats.anova_lm(mod, typ=2)
    print("\n", aov_table)
    
    mod2 = ols('SAFETY ~ S_SEX_MALE', data=descriptiveData).fit()
    aov_table2 = mp.stats.anova_lm(mod2, typ=2)
    print("\n", aov_table2)


# In[20]:


print("Welcome to my Introduction to Data Science - Midterm Project:")
print("\nIn this program I am going to read a CSV File and I am going to show you some information about it. The CSV File that I am readding is a survey carrie out to the inhabitants of different areas of Chicago, where they have to evaluate how safe and trust they feel living in their areas. Link: https://catalog.data.gov/dataset/police-sentiment-scores")

descriptiveData = readFile(fileName)
print("\n\n\nDESCRIPTIVE STATISTICS:")
descriptiveStatistics(descriptiveData)
print("\n\n\nPROBABLITY:")
probablity(descriptiveData)
print("\n\n\nHYPOTHESIS:")
hypothesis(descriptiveData)
print("\n\n\nANOVA:")
anova_test(descriptiveData)
df.style.format(make_clickable)




# In[ ]:




