THE ACCURACY OF THE CLASSIFIER

#Use the following code to import the SVM, kNN and RF classifiers
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Run the three classifiers on the training set with their default parameter values.
#Each fitted object will then have an attribute score which gives
#the accuracy of the classifier (i.e. 1 - misclassification rate). Which has the highest score on the test data?
#(write knn, rf or svm)

#so in this way I calculate the accuracy of the classifier

knn_scores = RandomForestClassifier().fit(X_train,y_train).score(X_val,y_val)#0.89532293986636968
knn_scores = [KNeighborsClassifier().fit(X_train,y_train).score(X_val,y_val) ] #0.95768374164810688 
knn_scores = LinearSVC().fit(X_train,y_train).score(X_val,y_val) #0.9265033407572382
--------------------------------------------------------------------------------------------------------------------------
STIMAREI I PARAMETRI PER KNN,RANDOM_FOREST AND SVM PER VEDERE QUAL E' IL MIGLIORE
we might want to estimate the 'best' number of neighbours 
Repeat the above code but for the RF and SVM models. 
For the RF model change the n_estimators parameter on the grid 5, 10, ... 30. 
For the SVM model change the C parameter on the values 0.001, 0.01, 0.1, 1, 10 100. Write the best value of n_estimators and C separated by a space. If you get two values with the same score write the smallest.
 

#Each of the above models has key parameters which we might like to estimate. For example, we might want to estimate the 'best' number of neighbours to use in kNN. To do this, we fit kNN with differing values of k to the training set and compare performance on the validation test. Some code to do this is below:
#kvals = range(1,11)
#knn_scores = [KNeighborsClassifier(n_neighbors=kval).fit(X_train,y_train).score(X_val,y_val) for kval in kvals]
#You should see that the best score is when k=1 and acheives 96.9% accuracy. Repeat the above code but for the RF and SVM models. For the RF model change the n_estimators parameter on the grid 5, 10, ... 30. For the SVM model change the C parameter on the values 0.001, 0.01, 0.1, 1, 10 100. Write the best value of n_estimators and C separated by a space. If you get two values with the same score write the smallest.
 
-----------------------------------------------------------------------------------------------------------------------
#Use your best knn, rf and svm models to PREDICT FOR THE TEST DATA SET. Which model performs best? (write knn, rf, or svm)
LinearSVC_scores = LinearSVC(C=0.001).fit(X_train,y_train).score(X_val,y_val)

rf_scores = RandomForestClassifier(n_estimators=25,random_state=123,).fit(X_train,y_train).score(X_val,y_val)

knn_scores = KNeighborsClassifier(n_neighbors=1).fit(X_train,y_train).score(X_val,y_val) 

#(Hint: use random_state=123 as an extra argument to your RF classifier to get repeatable results).
n_estimators=range(5,35,5)
rf_scores = [RandomForestClassifier(n_estimators=i,random_state=123,).fit(X_train,y_train).score(X_val,y_val) for i in n_estimators]  #25

C=[0.001,0.01,0.1,1,10]
LinearSVC_scores = [LinearSVC(C=i).fit(X_train,y_train).score(X_val,y_val) for i in C] #0.001
