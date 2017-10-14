#dati considerati
from sklearn import datasets
diabetes = datasets.load_diabetes()

X_raw = DataFrame(diabetes.data,columns= ['age','sex','bmi','bp','S1','S2','S3', 'S4','S5','S6']) 
X_raw_std = (X_raw-X_raw.mean())/ X_raw.std() 
y = Series(diabetes.target)

To make things more interesting we will append 100 noise variables onto X:
X = X_raw.join(DataFrame(np.random.randn (len(y), 100)))

# Create a training and test set of size 332 and 110 (approx 75%/25% split)

#1 step: size of training set = 75% of row data
train_size = 332
#2 step: creating train data

# NO PERMUTATION
#attenzione faccio la permutazione solo se mi e' richiesto altrimenti salto il passaggio
# altrimenti :
#X_train=X.ix[:train_size,:].reset_index(drop=True) 
#X_test=X.ix[train_size:,:].reset_index(drop=True)
#y_train=y.ix[:train_size].reset_index(drop=True)
#y_test=y.ix[train_size:].reset_index(drop=True)

#YES PERMUTATION

np.random.seed(123)
train_size=332 #75% dei dati. 
train_select=np.random.permutation(range(len(y)))
X_train=X.ix[train_select[:train_size],:].reset_index(drop=True) ## Note that reset_index here is used so that we can access the first row of X_train with X_train.ix[0,:] for example
X_test=X.ix[train_select[train_size:],:].reset_index(drop=True)  

# creating train and test for y response variable 
y_train=y.ix[train_select[:train_size]].reset_index(drop=True)
y_test=y.ix[train_select[train_size:]].reset_index(drop=True)
# Remember what the above does: creates a test data set and drops the original index


#LINEAR REGRESSION MODEL
# Let's start with just doing a LINEAR REGRESSION MODEL in SCIKIT-LEARN on these data to get the hang(la pendenza) of things
from sklearn import linear_model
# Set up the linear regression
reg=linear_model.LinearRegression()
# Fit it
reg.fit(X_train,y_train)
# Examine the coefs
reg.coef_
# Look at how well it predicts on test data
reg_test_pred = reg.predict(X_test) #len = 110

# See how well they agree in a simple plot
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(y_test,reg_test_pred,'kx')
plt.plot(plt.xlim(), plt.ylim(), ls="--")
# Doesn't look too bad
# Compute residual sum of squares as well
RSS_reg = np.mean(pow((reg_test_pred - y_test),2)) #https://en.wikipedia.org/wiki/Residual_sum_of_squares  #mean squared error
# 4274.3

#LASSO REGRESSION : standard least squares linear regression
# Note that the lasso is still a linear model, and it does 
#not include any interactions between variables unless 
#specifically included in the X matrix I Fitting methods
# for the lasso are very fast; it can handle hundreds 
#of thousands of features with ease

lasso = linear_model.LassoCV(cv=10)
lasso.fit(X_train, y_train)
lasso.coef_ # Has set some of them to zero for the reason see the slide 14/28 lecture 11
# Predict
lasso_test_pred = lasso.predict(X_test)
# Plot
fig = plt.figure()
plt.plot(y_test,lasso_test_pred,'kx')
plt.plot(plt.xlim(), plt.ylim(), ls="--")
# Doesn't look too bad
# Compute residual sum of squares as well
RSS_lasso = np.mean(pow((lasso_test_pred - y_test),2))   
# 3978.99 - better than linear regression
# In my code lasso.coef displayed that some of the coe?cients (the ßs) 
#had been set to zero, and the RSS was slightly smaller than that of
# standard linear regression
NOTE
#The elastic net is an extension of the lasso. The details are here:
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html#sklearn.linear_model.ElasticNetCV
#It can be fitted with, e.g.
el_net = linear_model.ElasticNetCV(l1_ratio=0.5,cv=10)
el_net.fit(X_train,y_train)
el_net_test_pred=el_net.predict(X_test)
RRS_el_net=np.mean(pow((el_net_test_pred-y_test),2))
#where if l1_ratio = 1 we have lasso and l1_ratio=0 we have ridge regression (another type of shrinkage model). Fit both a lasso and elastic net regression (with 10-fold cross validation) on the training data and give your answer as the lowest mean squared error to 1dp (you can use the elastic net options as given in the code above)

#SUPPORT VECTOR REGRESSION/MACHINES
#The method is somewhat similar to the lasso but uses a di?erent 
#loss function, where only those points which contribute a large amount
#to the residual sum of squares are used in the prediction; these are 
#the so-called support vectors
from sklearn import svm
my_svm = svm.SVR()
my_svm.fit(X_train, y_train) 
# Predict
svm_test_pred = my_svm.predict(X_test)
# Plot
fig = plt.figure()
plt.plot(y_test,svm_test_pred,'kx')
plt.plot(plt.xlim(), plt.ylim(), ls="--")
# Performance 
RSS_svm = np.mean(pow((svm_test_pred - y_test),2))
# 5808.52 - Shocking!

# RANDOM FOREST REGRESSION
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train) 
# Predict
rf_test_pred = rf.predict(X_test)
# Plot
fig = plt.figure()
plt.plot(y_test,rf_test_pred,'kx')
plt.plot(plt.xlim(), plt.ylim(), ls="--")
# Performance 
RSS_rf = np.mean(pow((rf_test_pred - y_test),2))
# 3820.78 - The winner!

#An simpler alternative to random forests is Decision Tree Regression (sometimes found under the name Classification And Regression Trees (CART). For examples, see
#http://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#example-tree-plot-tree-regression-py
#Fit a decision tree model with max_depth 2 and 5 and give the mean squared error of the smallest to 1dp. 
#(Hint: use the extra argument random_state=123 to make sure your results are repeatable)
from sklearn.tree import DecisionTreeRegressor
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X_train,y_train)
regr_2.fit(X_train,y_train)
regr_1_test_pred = regr_1.predict(X_test)
regr_2_test_pred = regr_2.predict(X_test)
RSS_regr_1 = np.mean(pow((regr_1_test_pred - y_test),2))
RSS_regr_2 = np.mean(pow((regr_2_test_pred - y_test),2))

# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train) 
# Predict classes
logreg_test_pred = logreg.predict(X_test)
# Mis-classification
logreg_cross = pd.crosstab(logreg_test_pred,y_test).astype('float')
(logreg_cross.ix[0,1]+logreg_cross.ix[1,0])/np.sum(logreg_cross.values)
# 0.365
# Predict probabilities if required
logreg_test_prob = logreg.predict_proba(X_test) # Bizarrely in two columns

#USING STATS.MODELS
#Create a misclassification table for the training data and discuss your results. 
#Which variables seem important in determining whether a congressman is republican or democrat?

logit1 = sm.Logit(z_train,X_train).fit()
print logit1.summary()

logit1_test_pred = logit1.predict(X_test)
# Mis-classification
logit1_cross = pd.crosstab(logit1_test_pred,z_test).astype('float')
(logit1_cross.ix[0,1]+logit1_cross.ix[1,0])/np.sum(logit1_cross.values)
#n predictive analytics, a table of confusion 
(sometimes also called a confusion matrix), is a table with two rows 
and two columns that reports the number of false positives, false negatives, true positives, and true negatives

## LASSO LOGISTIC REGRESSION  C is the lasso penalty
lassologreg = LogisticRegression(penalty='l1',C=0.1)
lassologreg.fit(X_train, y_train) 
# Predict classes
lassologreg_test_pred = lassologreg.predict(X_test)
# Mis-classification
lassologreg_cross = pd.crosstab(lassologreg_test_pred,y_test).astype('float')
(lassologreg_cross.ix[0,1]+lassologreg_cross.ix[1,0])/np.sum(lassologreg_cross.values)
# 0.339
# Predict probabilities
lassologreg_test_prob = lassologreg.predict_proba(X_test) # Bizarrely in two columns


# Create ROC curve and AUC (area under curve) for each of the above
#I USE AUC SE VOGLIO COMPARARE DIVERSI METODI DI CLASSIFICAZIONE. QUELLO CHE PRESENTA L ARIA PIU GRANDE E' IL MIGLIOR MODELLO
from sklearn.metrics import roc_curve, auc

#roc curve and auc for LOGISTIC REGRESSION
roc_lr = roc_curve(y_test, logreg_test_prob[:,1]) # Returns fpr, tpr, cutoffs
lr_auc = auc(roc_lr[0],roc_lr[1]) #calcola l area sotto la curva auc(x,y)

#roc curve and auc for LASSO logistic regression
roc_lassolr = roc_curve(y_test, lassologreg_test_prob[:,1])
lassolr_auc = auc(roc_lassolr[0],roc_lassolr[1])

#roc curve and auc for RANDOM FOREST
roc_rf = roc_curve(y_test, rf_test_prob[:,1])
rf_auc = auc(roc_rf[0],roc_rf[1])

#roc curve and auc for SVM classification
roc_svm = roc_curve(y_test, svm_test_prob[:,1])
svm_auc = auc(roc_svm[0],roc_svm[1])

# Now plot and label:
plt.figure()
plt.plot(roc_lr[0],roc_lr[1],label='Logistic regression (AUC = {0:0.2f})'''.format(lr_auc))
plt.plot(roc_lassolr[0],roc_lassolr[1],label='Lasso logistic regression (AUC = {0:0.2f})'''.format(lassolr_auc))
plt.plot(roc_rf[0],roc_rf[1],label='Random forest classification (AUC = {0:0.2f})'''.format(rf_auc))
#plt.plot(roc_svm[0],roc_svm[1],label='Support vector machine classification (AUC = {0:0.2f})'
#               ''.format(svm_auc))
plt.legend(loc='lower right')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves for three different models applied to SA heart data')
