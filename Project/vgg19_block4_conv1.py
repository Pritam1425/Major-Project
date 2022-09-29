#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
from glob import glob
import itertools
import fnmatch
import random
import matplotlib.pylab as plt
import seaborn as sns
import cv2
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.applications import VGG16
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
from sklearn.metrics import auc,roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import Model
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn import metrics


# In[2]:


layers_list = ["block4_conv1"]


# In[3]:
image_signal =['viterbi']

for Layer_Featue in layers_list:
    for gname in image_signal:

        try:
            print('$$$$$$$$$$$$$$$$   ',Layer_Featue,'$$$$$$$$$$$$   ',gname)
            default = "/scratch/sks.cse.iitbhu/edf/"
            #train
            directory1 = default+gname+"/train"
            directory2 = default+gname+"/eval"

            X_train = np.load(directory1+'/X_train.npy')
            Y_train = np.load(directory1+'/Y_train.npy')

            X_val = np.load(directory1+'/X_val.npy')
            Y_val = np.load(directory1+'/Y_val.npy')

            Y_train1 = np.load(directory1+'/Y_train_1dim.npy')
            Y_val1 = np.load(directory1+'/Y_val_1dim.npy')

            Y_test1 = np.load(directory2+'/Y_test_1dim.npy')
            X_test = np.load(directory2+'/X_test.npy')
            Y_test = np.load(directory2+'/Y_test.npy')



            print("Training Features:", X_train.shape)
            print("Training Labels:", Y_train.shape)
            print("Test Features:", X_test.shape)

            print("Validation Features:", X_val.shape)
            print("Validation Labels:", Y_val.shape)
            print("Validation Labels 1 dim:", Y_val1.shape)
            print("Validation Labels 1 dim:", Y_train1.shape)
            n_length, n_features, n_outputs = X_train.shape[0], X_train.shape[2], Y_train.shape[1]
            print(X_train.shape[0])
            
            
            ################### Model + Classifier ######################



            X_train_temp = np.concatenate((X_train,X_val), axis=0)
            Y_train_temp = np.concatenate((Y_train,Y_val), axis=0)
            Y_train1_temp = np.concatenate((Y_train1,Y_val1), axis=0)

            X_train = X_train_temp
            Y_train = Y_train_temp
            Y_train1 = Y_train1_temp





            # Model
            print("Model")

            from keras.applications.vgg19 import VGG19 #downloading model for transfer learning
            model_vgg19 = VGG19(include_top=False,weights='imagenet',input_shape=(256,256,3),classes=2)
            optimizer = Adam(lr=0.0001)

            arg_model= Model(inputs=model_vgg19.input, outputs=model_vgg19.get_layer(Layer_Featue).output)
            arg_model.compile(loss='categorical_crossentropy',
                              optimizer=optimizer,
                              metrics=['accuracy'])

            from keras.applications.vgg19 import preprocess_input #preprocessing the input so that it could work with the downloaded model
            bottleneck_train1=arg_model.predict(preprocess_input(X_train),batch_size=50,verbose=1) #calculating bottleneck features, this inshure that we hold the weights of bottom layers
            #bottleneck_val1=arg_model.predict(preprocess_input(X_val),batch_size=50,verbose=1)
            bottleneck_test1=arg_model.predict(preprocess_input(X_test),batch_size=50,verbose=1)


            print(bottleneck_test1.shape)
            print(bottleneck_train1.shape)
            #print(bottleneck_val1.shape)


            train1,train2,train3,train4 = bottleneck_train1.shape
            trainshape = train2*train3*train4
            print('train shape',trainshape)

            test1,test2,test3,test4 = bottleneck_test1.shape
            testshape = test2*test3*test4
            print('test shape',testshape)

            bottleneck_train=bottleneck_train1.reshape(train1,trainshape) 
            #bottleneck_val=bottleneck_val1.reshape(6,featureshape)
            bottleneck_test=bottleneck_test1.reshape(test1,trainshape)

            #bottleneck_train=bottleneck_train1.flatten() 
            #bottleneck_val=bottleneck_val1.flatten()
            #bottleneck_test=bottleneck_test1.flatten()



            print(bottleneck_test.shape)
            print(bottleneck_train.shape)
            



            print('###### Random Forest #########')

            from sklearn.ensemble import RandomForestClassifier
            #Change the directory

            classifer_rf_pred = np.load('/home/sks.cse.iitbhu/tuh/final/rf_viterbiblock4_conv1.npy')


            from sklearn.metrics import confusion_matrix
            import pandas as pd

            print(pd.DataFrame(confusion_matrix(Y_test1, classifer_rf_pred), index = ['0', '1'], columns = ['0', '1']))

            # Accuracy

            from sklearn.metrics import accuracy_score

            print(accuracy_score(Y_test1, classifer_rf_pred))


            #print(Y_train)
            #arg_model.summary()
            cm1 = confusion_matrix(Y_test1, classifer_rf_pred)
            print('Confusion Matrix : \n', cm1)

            total1=sum(sum(cm1))

            #####from confusion matrix calculate accuracy
            accuracy1=(cm1[0,0]+cm1[1,1])/total1
            print ('Accuracy : ', accuracy1)

            sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
            print('Sensitivity : ', sensitivity1 )

            specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
            print('Specificity : ', specificity1)

            # Classification Report

            from sklearn.metrics import classification_report
            print(classification_report(Y_test1, classifer_rf_pred))

            from pycm import ConfusionMatrix
            cm = ConfusionMatrix(actual_vector=list(Y_test1),predict_vector=list(classifer_rf_pred))
            print(cm)
            print('Geometric means')
            from imblearn.metrics import geometric_mean_score
            g = geometric_mean_score(Y_test1, classifer_rf_pred, average='weighted')
            print(g)
            print('End Geometric means')

            # In[ ]:

            print('###### SVM linear #########')
            from sklearn import svm
            kernel = 'linear'
            classifer_svmrbf = svm.SVC(kernel=kernel, gamma=2)

            #Train the model
            classifer_svmrbf.fit(bottleneck_train, Y_train1)

            #Predict the values on test data
            classifer_svmrbf_pred = classifer_svmrbf.predict(bottleneck_test)


            from sklearn.metrics import confusion_matrix
            import pandas as pd

            print(pd.DataFrame(confusion_matrix(Y_test1, classifer_svmrbf_pred), index = ['0', '1'], columns = ['0', '1']))

            # Accuracy

            from sklearn.metrics import accuracy_score

            print(accuracy_score(Y_test1, classifer_svmrbf_pred))

            from sklearn.metrics import roc_curve, auc,roc_auc_score

            np.save('./svm_'+gname+''+Layer_Featue+'.npy',classifer_svmrbf_pred)

            #print(Y_train)
            #arg_model.summary()


            # Classification Report

            from sklearn.metrics import classification_report
            print(classification_report(Y_test1, classifer_svmrbf_pred))

            from pycm import ConfusionMatrix
            cm = ConfusionMatrix(actual_vector=list(Y_test1),predict_vector=list(classifer_svmrbf_pred))
            print(cm)


            print('######Logistic Regression #########')
            from sklearn.linear_model import LogisticRegression

            # Paramters setting for the model
            classifer_LR = LogisticRegression(C=1.0, max_iter=1000, solver='newton-cg')


            #Train the model
            classifer_LR.fit(bottleneck_train, Y_train1)

            #Predict the values on test data
            logistic_regression_pred = classifer_LR.predict(bottleneck_test)


            from sklearn.metrics import confusion_matrix
            import pandas as pd

            print(pd.DataFrame(confusion_matrix(Y_test1, logistic_regression_pred), index = ['0', '1'], columns = ['0', '1']))

            # Accuracy

            from sklearn.metrics import accuracy_score

            print(accuracy_score(Y_test1, logistic_regression_pred))
            from sklearn.metrics import roc_curve, auc,roc_auc_score

            np.save('./logistic_'+gname+''+Layer_Featue+'.npy',logistic_regression_pred)

            #print(Y_train)
            #arg_model.summary()


            # Classification Report

            from sklearn.metrics import classification_report
            print(classification_report(Y_test1, logistic_regression_pred))

            from pycm import ConfusionMatrix
            cm = ConfusionMatrix(actual_vector=list(Y_test1),predict_vector=list(logistic_regression_pred))
            print(cm)




            print('$$$$$$$$$$$$$$$$   ',Layer_Featue,'   $$$$$$$$$$$$   ',gname)

        except Exception as e:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@  errror@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print(e)
            print('$$$$$$$$$$$$$$$$   ',Layer_Featue,'   $$$$$$$$$$$$   ',gname)

