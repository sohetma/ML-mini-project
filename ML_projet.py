# -*- coding: utf-8 -*-
"""
@author: Sohet Maxime et Poelman Simon
"""

from ML_functions import *
from Multi_linear_regression import *

testSize = 0.2

#Load data
X1,y1,X2,feature_names,l,c = load_data()

#split the data
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = testSize, random_state = 0)

#Linear model
X1_train, X1_test, y1_train,y1_test,model,regressor,y_pred = ml_linear_reg(X1,y1,X1_train, X1_test, y1_train, y1_test)
print(model.coef_, model.intercept_) 

#Linear model after backwardElimination
model_opt,regressor_opt,y_pred_opt =  ml_linear_reg_opt(X1,y1,testSize)
print(model_opt.coef_, model_opt.intercept_) 

#Knn model
knn = KNeighborsRegressor(n_neighbors = 5)  
X1_train,X1_test = Normer(X1_train,X1_test)
knn.fit(X1_train,y1_train)
y_pred_knn = knn.predict(X1_test)

#fitting Kernel SVM to the training set
classifier = svm.SVC(kernel = 'rbf', random_state =0)
y1_train=y1_train.astype('int')
classifier.fit(X1_train,y1_train)
y_pred_svc = classifier.predict(X1_test)

#plot_Linear_prediction(y1_test,y_pred_svc,'red','svc_prediction')
"""
#confusion matrix
cm = confusion_matrix(y1_test,y_pred_svc)
print("Confusion Matrix : ", cm)
"""

#applying k-Fold Cross Validation
result_linear = cross_val_score(regressor, X1, y1,cv=10)
result_knn = cross_val_score(knn, X1, y1,cv=10)
y1=y1.astype('int')
result_SVC = cross_val_score(classifier, X1, y1,cv=10)
print("Accuracy linear : %0.2f (+/- %0.2f)" % (result_linear.mean(), result_linear.std() * 2))
print("Accuracy KNN : %0.2f (+/- %0.2f)" % (result_knn.mean(), result_knn.std() * 2))
print("Accuracy SVC : %0.2f (+/- %0.2f)" % (result_SVC.mean(), result_SVC.std() * 2))

#computing mean_squared_error
err = mean_squared_error(y1_test, y_pred)
err_new = mean_squared_error(y1_test, y_pred_opt)
err_knn = mean_squared_error(y1_test, y_pred_knn)
err_svc = mean_squared_error(y1_test, y_pred_svc)
print('RMSE Linear = ',err)
print('New RMSE Linear = ',err_new)
print('RMSE knn = ',err_knn)
print('RMSE svc = ',err_svc)

plot_Linear_prediction(y1_test,y_pred,'blue','Model with all features')
plot_Linear_prediction(y1_test,y_pred_opt,'red','Model with backwardElimination')

plot_prediction(y1_test,y_pred,y_pred_knn,y_pred_opt,y_pred_svc)

for i in range(0,8): 
  plt.clf() 
  analyze_feature(X1,y1,feature_names,i) 
  
#Mutual information  
plot_mutual_inf(X1,c,feature_names)  




# Analysis of k's influence

# for a range of neighbors k
neighbors = list(range(1,50)) 

cv_score = []
Rmse_score = []

# finds the k that gives the best cross validation score
for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors = k)   
    scores = cross_val_score(knn,StandardScaler().fit_transform(X1), y1.ravel(), cv=5)
    cv_score.append(scores.mean())

# the best k is the one that returns the best cross_validation_score
opti_K = neighbors[cv_score.index(max(cv_score))]

#validation curve plot 
#validation curve for the parameter k, 
figvc = plt.gcf()
param_range = list(range(1,50))
figvc = plot_vc(KNeighborsRegressor(),StandardScaler().fit_transform(X1),y1.ravel(),param_name = "n_neighbors",param_range = param_range,cv=5,scoring=score,
                title = 'Validation curve with Knn',xlabel='Parameter K')
  
plt.legend()
plt.show()
#plt.draw()
#figvc.tight_layout()
figvc.savefig('vcK')