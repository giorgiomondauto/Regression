
# Logistic regression

# Revision(?): logistic regression same as standard linear regression except for 
# response variable is binary and assumed Bernoulli distributed
# Model:
# y_i ~ Bernoulli(p_i)
# logit(p_i) = XB
# where logit(z) = log(z/(1-z))

# Let's load in a new dataset for this:
SA = pd.read_csv('http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data')

# Columns are 
# sbp - systolic blood pressure
# tobacco - cumulative tobacco (kg)
# ldl - low densiity lipoprotein cholesterol
# adiposity
# famhist - family history of heart disease (Present, Absent)
# typea - type-A behavior
# obesity
# alcohol - current alcohol consumption
# age - age at onset
# chd - response, coronary heart disease

# Lets describe the data
SA.describe()

# Missing famhist
SA.famhist.describe()

# Standardise
SA_numeric = SA.drop(['famhist','chd'],axis=1)
SA_std = (SA_numeric-SA_numeric.mean())/SA_numeric.std()
#AGG constant
SA_std_const = sm.add_constant(SA_std)

# Run the logistic regression
import statsmodels.api as sm
logit1 = sm.Logit(SA.chd, SA_std_const).fit()
# Note: this is usually fitted via maximum likelihood
print logit1.summary()

# As before loads of other stuff you can get
dir(logit1)

# Age seems to be the most important variable; perhaps re-fit and plot effect
new_age = DataFrame({'age':np.linspace(-3,3,100)})
new_preds = logit2.predict(new_age)
# Turn into a DataFrame
preds_df = DataFrame({'age':new_age.values[:,0],'preds':new_preds})

# Plot
fig = plt.figure()
preds_df.plot('age','preds')
# Add in data points
plt.plot(SA_std_const['age'],SA.chd,'r+')
plt.ylim(-0.05,1.05)
# Might need a bit of jitter too


