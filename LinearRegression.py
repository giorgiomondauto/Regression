# Let's now move on to statsmodels

# Revision: linear regression works by fitting the model 
# y = XB + e
# where e ~ N(0,s^2)
# We want to estimate B and s.
# The usual way to do this is Ordinary Least Squares. 
# Endogenous means the response variable y 
# Exogenous means the explanatory variables X

# Linear regression very simple to run
import statsmodels.api as sm
mod = sm.OLS(prostate_std.lpsa, prostate_std.drop('lpsa',axis=1))
res = mod.fit()
print res.summary()
# Lots of fitting detail - be careful no intercept!

#INTERPRETATION
#Coefficient:
#Since X1 is a continuous variable, B1 represents the difference in the predicted value of Y for each one-unit difference in X1, if X2 remains constant.
#This means that if X1 differed by one unit, and X2 did not differ, Y will differ by B1 units, on average.

#R-squared:
The definition of R-squared is fairly straight-forward; it is the percentage of the response variable variation that is explained by a linear model. Or:
R-squared = Explained variation / Total variation
R-squared is always between 0 and 100%:
0% indicates that the model explains none of the variability of the response data around its mean.
100% indicates that the model explains all the variability of the response data around its mean.
#AIC
#take the model with the lowest AIC


# Can add an intercept this way. INTERCEPT HAS NO REAL MEANING
X_with_const = sm.add_constant(prostate_std.drop('lpsa',axis=1))

# Fitting with R style formulae
import statsmodels.formula.api as smf
mod2 = smf.ols(formula='lpsa ~ lcavol + lweight + age', data=prostate_std)
res2 = mod2.fit()
print res2.summary() # Note that this includes an intercept

# It also allows to do things like simple interactions
mod3 = smf.ols(formula='lpsa ~ lcavol * lweight * age', data=prostate_std).fit()
print mod3.summary() # None of these interactions seem important
# There are added options in smf.ols for using subsets
# Also options in .fit() for fitting method: I think this might use a different fitting method than R - use Moore-Penrose rather than QR

# You can also add things in which are categorical
prostate_std['age_lt_65'] = prostate.age>65
mod4 = smf.ols(formula='lpsa ~ lcavol + lweight + C(age_lt_65)', data=prostate_std).fit()
print mod4.summary()
# Note that it dropped one of the categories in age_lt_65 so that the model was fully determined

# You can also drop the intercept
mod5 = smf.ols(formula='lpsa ~ lcavol + lweight + C(age_lt_65) -1', data=prostate_std).fit()
print mod5.summary() # Now it can use both age_lt_65 values

# Or add in your own functions
mod6 = smf.ols(formula='lpsa ~ lcavol + pow(lcavol,2)', data=prostate_std).fit()
print mod6.summary() # Now it can use both age_lt_65 values

# You can actually get loads more than just the summary
dir(mod6)
# Huge list

# Example: PLOT OF FITTED VALUES AND RESIDUALS 
mod_summary = DataFrame({'preds':mod6.predict(),'resids':mod6.resid})
mod_summary.plot('preds','resids',kind='scatter')
# Looks like a pretty random scatter
INTERPRETATIONS
#The plot is used to detect non-linearity, unequal error variances, and outliers.
#Any data point that falls directly on the estimated regression line has a residual of 0. 
#Therefore, the residual = 0 line corresponds to the estimated regression line.
#IF :
#The residuals "bounce randomly" around the 0 line. This suggests that the assumption that the relationship is linear is reasonable.
#The residuals roughly form a "horizontal band" around the 0 line. This suggests that the variances of the error terms are equal.
#No one residual "stands out" from the basic random pattern of residuals. This suggests that there are no outliers.

#Linear regression model con StatsModels
import statsmodels.api as sm  # No constant is added by the model unless you are using formulas. : import statsmodels.formula.api as smf
mod = sm.OLS(y,x).fit()
mod.summary()
# the coefficient value_coeff means that for every additional "grade" in x. the y, on average, increase of value_coeff "billion"

#Note that
mod.fittedvalues # sono i valori stimati quindi y cappello
mod.resid # sono i residui

plt.figure(figsize(4,5))
plt.subplot(2,2,2)
plt.scatter(x,y)
plt.xlabel("Maximum daily temperature in degrees")
plt.ylabel("the mean ozone in parts per billion")
plt.title("environment situation in New York")
plt.plot(plt.xlim(), plt.ylim(), ls="--") # for the fitted line overlaid
plt.subplot(2,2,1)
plt.scatter(mod.fittedvalues,mod.resid)  #fitted values are the values of response variable
plt.xlabel("fitted values")
plt.ylabel("resid")
plt.plot(plt.xlim(), plt.ylim(), ls="--")
