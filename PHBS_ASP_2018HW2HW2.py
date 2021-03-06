
# coding: utf-8

# # HW 2: Corporate Bond Pricing (due by 9.21 Fri)

# We are going to compute the price of a corporate bond (subject to default) with Monte-Carlo simulation. Assume that 
# * the default time of a company follows the exponential distribution with intensity $\lambda=$__`def_rate`__. 
# * the riskfree interest rate is $r_f=$__`rf_rate`__ and the maturity of the bond is $T=$__`mat`__. 
# * in the case of default, you can recover some portion ($R=$__`recovery_rate`__) of the face value.
# * the coupon is 0%, i.e., it is a zero-coupon bond.
# * the face value of the bond is 1.0
# * use compound rate for discounting; the price of the default-free bond is $e^{-r_f T}$
# 
# The Problem 1 of the [2017 ASP Midterm Exam](../files/ASP2017_Midterm.pdf) will be helpful.
# 
# ### Instruction to upload your HW
# * Create a repository named __`PHBS_ASP`__ (and clone it to your PC)
# * Copy this file to __`PHBS_ASP_2018/HW2/HW2.ipynb`__  (Please use the same name for repository and ipynb file)
# * Adding more code.
# * Run your your code to make sure that there's no error.
# * Upload (commit and sync) your file.

# ### 1. First, let's create a pricing function and check the std 

# In[83]:


import numpy as np


# In[84]:


def_rate = 0.1
rf_rate = 0.03
recovery = 0.3
mat = 10


# In[85]:


# First generate exponential random numbers
# Although you can generate directly using fault_time = np.random.exponential(scale=), let's use uniform random numbers.
n_sample = 10000
U = np.random.uniform(size=n_sample)
default_time = -(1/def_rate)*np.log(U)

# You can check if the RNs are correct by comparing the means
(default_time.mean(), 1/def_rate)


# In[86]:


# Put your code here to price the corporate bond
def corp_bond(mat=1, def_rate=0.03, rf_rate=0.04, recovery=0.3, n_sample=1e4):
    U = np.random.uniform(size=int(n_sample))
    default_time = -(1/def_rate)*np.log(U)
    v1 = 0
    v2 = 0
    for df_time in default_time:
        if df_time <= mat:
            v1 += np.exp(-rf_rate * df_time)
        else:
            v2 += np.exp(-rf_rate * mat)
    def_value = v1 * recovery / int(n_sample)
    survive_value = v2 / int(n_sample)
    return def_value + survive_value


# Call your function
corp_bond()

# Find the mean and std by calling the function 100 times. 
def MC_corp_bond(repeat_time=100):
    vals = np.zeros(repeat_time)
    for k in range(repeat_time):
        vals[k] = corp_bond()
    return( [np.mean(vals), np.std(vals)] )


# In[87]:


MC_corp_bond()


# In[88]:


#Another way to calculate the expected payoff of bond：
def default_time_norm():
    U = np.random.uniform(size=10000)
    return -(1/0.03)*np.log(U)

def corp_bond(term, mat=1, def_rate=0.03, rf_rate=0.04, recovery=0.3, n_sample=1e4):
    v1 = 0
    v2 = 0
    for df_time in term:
        if df_time <= mat:
            v1 += np.exp(-rf_rate * df_time)
        else:
            v2 += np.exp(-rf_rate * mat)
    def_value = v1 * recovery / int(n_sample)
    survive_value = v2 / int(n_sample)
    return def_value + survive_value

corp_bond(default_time_norm())


# ### 2. Now, let's improve the function by reducing the MC variations.
# 1. Use antithetic method: If `U` is uniform random variable, so is `1-U`
# 2. Also shift the RNs to match the mean, `1/def_rate`

# In[92]:


# For example, antithetic method mean
n_sample = 10000
U = np.random.uniform(size=n_sample)
default_time = -(1/def_rate)*np.log(np.concatenate((U,1-U),axis=0))

# Mean-matching means
default_time += 1/def_rate-default_time.mean()
(default_time.mean(), 1/def_rate)


# In[93]:


# No include the two new features: `antithetic` and `mean_match`

def default_time_antithetic():
    U = np.random.uniform(size=5000)
    return -(1/0.03)*np.log(np.concatenate((U,1-U),axis=0))

def default_time_mean_match():
    U = np.random.uniform(size=10000)
    default_time = -(1/0.03)*np.log(U)
    default_time += 1/0.03-default_time.mean()
    return default_time

def default_time_both():
    U = np.random.uniform(size=5000)
    default_time = -(1/0.03)*np.log(np.concatenate((U,1-U),axis=0))
    default_time += 1/0.03-default_time.mean()
    return default_time
    


# In[94]:


#待删除
print(corp_bond(default_time_antithetic()))
print(corp_bond(default_time_mean_match()))
print(corp_bond(default_time_both()))


# In[95]:


def corp_bond_cv(mat=1, def_rate=0.03, rf_rate=0.04, recovery=0.3, n_sample=1e4, antithetic=True, mean_match=True):
    if(antithetic):
        return corp_bond(default_time_antithetic())
        
    if(mean_match):
        return corp_bond(default_time_antithetic())
    
    if (antithetic) and (mean_match):
        return corp_bond(default_time_both())


# In[97]:


# Find the mean and std by calling the function 100 times for (i) antithetic (ii) mean_match and (iii) both

vals1, vals2, vals3 = np.zeros(100), np.zeros(100), np.zeros(100)
for k in range(100):
    vals1[k] = corp_bond_cv(antithetic=True, mean_match=False)
    vals2[k] = mean_match = corp_bond_cv(antithetic=False, mean_match=True)
    vals3[k] = corp_bond_cv(antithetic=True, mean_match=True)

print("antithetic:","mean:",np.mean(vals1),",","std:",np.std(vals1))
print("mean_match:","mean:",np.mean(vals2),",","std:",np.std(vals2))  
print("both: ","mean:",np.mean(vals3),",","std:",np.std(vals3))


# ### 3. Finally, what is the analytic value of the corporate bond? How does it compare to your MC result above?

# ### Put the analytic expression for the corporate bond price
# $$\int_0^{mat} {e^{-rf}*recovery*\lambda*e^{-\lambda*t}}\,{\rm d}t + e^{(-rf-\lambda)*mat} $$

# In[98]:


mat = 1
def_rate = 0.03
rf_rate = 0.04
recovery = 0.3


# ### subsitiue those value into the analytical expression:
# $$\int_0^{1} {0.009*e^{-0.07*t}}\,{\rm d}t + e^{-0.07} $$
# ### then the analytical value will be:

# In[99]:


9/70*(1-np.exp(-0.07))+np.exp(-0.07)


# ## 4.Conclusion:

# ### The analytical value is 0.94108604306089783. After applying mean-matching method, the estimated values are  closer to the analytical value than others. 
# ### This may due to the process of mean matching method, when generating random numbers under this method, we can find the difference with true mean is very small.
# ### In fact, when running this code:
# (default_time.mean(), 1/def_rate)

# ### It only shows several results:
# (10.0, 10.0)<br />(10.000000000000002, 10.0)<br />(10.000000000000004, 10.0)<br />(9.9999999999999982, 10.0)
# ### and the fisrt case shows more frequent in my runing results. We can do this generating 100 times:

# In[114]:


# Mean-matching means
n_sample = 10000
def_rate = 0.1
U = np.random.uniform(size=n_sample)
default_time += 1/def_rate-default_time.mean()
print((default_time.mean(), 1/def_rate))

vals = np.zeros(100)
for k in range(100):
    vals[k] = default_time.mean()
    
print([np.mean(vals), np.std(vals)] )


# ### In this mean_match method, the random variables are much more closer to the true expotential distribution.
