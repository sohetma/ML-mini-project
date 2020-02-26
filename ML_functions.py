# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 17:10:19 2018

@author: user
"""
from ML_imports import *

# -*- chrono -*-
import time

_global_start_time = 0
def tic():
    global _global_start_time
    _global_start_time = time.time()

def toc():
    return time.time() - _global_start_time


def load_data():
  dataset1 = pd.read_csv('X1_t1.csv')
  dataset2 = pd.read_csv('X2.csv')
  l,c = dataset1.shape
  X1 = dataset1.iloc[:, :-1].values
  y1 = dataset1.iloc[:, c-1].values
  X2 = dataset2.iloc[:, :].values
  feature_names = list(dataset1.columns.values)
  
  return X1,y1,X2,feature_names,l,c

def backwardElimination(x,y1,SL,l,c):
    numVars = len(x[0])
    useless = np.zeros((1,c)).astype(int)
    temp = np.zeros((l,c)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y1, x).fit()
        #print(regressor_OLS.summary())
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    useless[:,j] = 1; 
                    tmp_regressor = sm.OLS(y1, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    print(regressor_OLS.summary())
    return x

def normalize_ML(X1_train,X1_test,y1_train,y1_test):     
    scaler = StandardScaler()
    X1_train = scaler.fit_transform(X1_train)
    X1_test = scaler.transform(X1_test)
    y1_train = scaler.fit_transform(y1_train)
    y1_test = scaler.transform(y1_test)
    return X1_train,X1_test,y1_train,y1_test
 
def Normer(X_train,X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train,X_test    
    
def adapt_dataset(X):
  l,c = X.shape
  #X = X[:,1:]
  X = np.append(arr = np.ones((l, 1)).astype(int), values = X, axis = 1)
  X.reshape(1, -1)
  return X
    
def score_function(model, X, Y):
    'norm-2 criterion for optimization of models'
    return np.sqrt(np.mean(((model.predict(X) - Y)) **2, axis=0)).sum()   
    
    
def plot_Linear_prediction(y1_test,y_pred,col,titre): 
  #plt.scatter(X1_train[:,0],y1_train,color='red')
  plt.scatter(y1_test,y_pred,color=col)
  #plt.plot(X1_train[:,0], regressor.predict(X1_train[:,0]),color='blue')
  plt.title(titre)
  plt.xlabel('True values')
  plt.ylabel('Predictions')
  plt.show()

def plot_prediction(y_test,y_pred_linear,y_pred_knn,y_pred_opt,y_pred_svc,i=0):
  plt.scatter(y_test,y_pred_linear,color='blue')    
  plt.hold(True)
  plt.scatter(y_test,y_pred_knn,color='red')
  plt.hold(True)
  plt.scatter(y_test,y_pred_opt,color='green')
  plt.title('Analyse prediction')
  plt.xlabel('True values')
  plt.ylabel('Predictions')
  plt.legend(('linear prediction','knn prediction','linear with BL prediction'),loc='best')
  plt.show()
  plt.savefig('analyse_pred_%d' % i,dpi=300)
  """
  plt.clf()
  plt.close()
  
  plt.plot(y_test,color='blue')    
  plt.hold(True)
  plt.plot(y_pred_opt,color='green')
  plt.hold(True)
  plt.plot(y_pred_knn,color='red')
  plt.hold(True)
  plt.plot(y_pred_svc,color='black')
  plt.title('Analyse des predictions')
  plt.legend(('True values','linear prediction','knn prediction','svc prediction'),loc='best')
  plt.show()
  plt.savefig('analyse_des_prediction_%d' % i,dpi=300)
  """
  
# simple function for plot analysis
def analyze_feature(X,y, feature_names,i=0, plot_target=True):
    plt1 = plt.subplot(2,1,1)
    plt.hist(X[:,i],bins = 50)
    plt.title(feature_names[i])
    if plot_target:
        plt2 = plt.subplot(2,1,2,sharex=plt1)
        plt.scatter(X[:,i],y)
    print('spearson coefficient [CSS]:', scipy.stats.pearsonr(X[:,i].ravel(), y.ravel())[0])
    # save image
    plt.savefig('feature_analysis_%d' % i, dpi=300)

# estimation of mutual information
def distribution(X, *columns):
    'compute distribution with explicit bucketing'
    nd, _ = X.shape
    count = {}
    for i in range(nd):
        entry = tuple(X[i, columns])
        if entry not in count:
            count[entry] = 0
        count[entry] += 1
    return {k:v/nd for (k,v) in count.items()}

def entropy(distr):
    p = np.array([v for v in distr.values()])
    return - (p * np.log2(p)).sum()

def normalized_mi(X, c1, c2):
    d12 = distribution(X, c1, c2)
    d1  = distribution(X, c1)
    d2  = distribution(X, c2)
    E   = entropy(d1) + entropy(d2)
    return (E - entropy(d12)) / entropy(d12)
    
def score(estimator, X_test, y_test):
    return np.sqrt(np.mean(((estimator.predict(X_test)-y_test))**2))   
####################### PLOT #######################

# Functions from scikit-learn.org    

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 20), scoring=score):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt_fig = plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt_fig

def plot_vc(model,X,Y,param_name,param_range,cv=10,scoring=score,title=None,xlabel=None,ylim=None,semilog=False):
    train_scores, test_scores = validation_curve(model, X, Y,param_name=param_name,
                                                   param_range = param_range, cv = cv,scoring=score)


    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fig = plt.figure()
    if ylim is not None:
        plt.ylim(*ylim)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if title is not None:
        plt.title(title)
    plt.ylabel("Score")
    #mp.ylim(0.8, 1.1)
    lw = 2
    if semilog is False:
        plt.plot(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
    else:
        plt.semilogx(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)        
    plt.plot([param_range[np.argmin(test_scores_mean)]],[test_scores_mean.min()],'or')
    plt.axvline(x=param_range[np.argmin(test_scores_mean)], linewidth=2, color='red',linestyle='--')
    plt.legend(loc="best")
    plt.show()
    print(param_range[np.argmin(test_scores_mean)])
    return fig

   
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
   
def plot_mutual_inf(X1,c,feature_names) :    
  nf = c-1
  corr = np.zeros((nf,nf))
  
  for i in range(nf):
      for j in range(nf):
          corr[i,j] = normalized_mi(X1, i, j)
          
  labels = feature_names
  labels = ['#%d' % i for i in range(0, nf)]
  
  seaborn.clustermap(corr, annot=True, xticklabels = labels, yticklabels = labels);
  plt.savefig('mutual_information_MI.png', dpi=300)
    