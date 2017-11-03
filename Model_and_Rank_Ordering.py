
# coding: utf-8


from collections import Counter
import csv
from csv import DictReader
#import seaborn
import imblearn
from sklearn.metrics import classification_report
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import feature_selection
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.datasets import make_classification
import pickle
from imblearn.ensemble import EasyEnsemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy


def Model_Building():

    X_train  = numpy.load('./combine_raw_enquiry_Train_X.npy')
    Y_train = numpy.load('./combine_raw_enquiry_Train_Y.npy')
    X_test  = numpy.load('./combine_raw_enquiry_Test_X.npy')
    Y_test = numpy.load('./combine_raw_enquiry_Test_Y.npy')



    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)



    print(X_train.shape)

    #numpy.set_printoptions(suppress=True)
    bbc = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=100)\
        ,ratio='auto',replacement=False,random_state=0, bootstrap_features=False)

    clf = SelectKBest(mutual_info_classif, k=100)
    X_train = clf.fit_transform(X_train, Y_train)
    X_test = clf.transform(X_test)
    bbc.fit(X_train, Y_train)
    y_pred = bbc.predict(X_test)
    # print('selectKBESt:',clf.get_support())
    # for  each in clf.get_support():
    #   print(each)
        


    '''
    END
    '''


    '''
    Feature selection using Pipeline
    '''
    # transform = feature_selection.SelectPercentile(feature_selection.f_classif)


    # clf = Pipeline([('anova', transform), ('classification', bbc )])
    # clf.set_params(anova__percentile=48)
    # clf.fit(X_train, Y_train) 

    # #print(bbc.coef_)

    # y_pred = clf.predict(X_test)
    '''
    END
    '''

    #feat_list = numpy.mean([est.steps[1][1].feature_importances_ for est in bbc.estimators_], axis=0)
    # print('BEFORE',feat_list)
    # feat_list.sort()
    # print('AFTER',feat_list)

    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test,y_pred))


    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred, pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    print('auc score =',auc_score)
    print('gini score =',2*auc_score-1)


def Rank_Ordering():
    X_train  = numpy.load('./combine_raw_enquiry_Train_X.npy')
    Y_train = numpy.load('./combine_raw_enquiry_Train_Y.npy')
    X_test  = numpy.load('./combine_raw_enquiry_Test_X.npy')
    Y_test = numpy.load('./combine_raw_enquiry_Test_Y.npy')

    feature_importance = {0: 0.03421852,  1: 0.00715693,  2: 0.05181835,  3: 0.01027565,  4: 0.00838411,  5: 0.0059313,  6: 0.00223466,  7: 0.00310911,  8: 0.00964993,  9: 0.01483778,  10: 0.0066048,  11: 0.00522294,  12: 0.02071259,  13: 0.00220598,  14: 0.03072879,  15: 0.00625178,  16: 0.01470893,  17: 0.02050969,  18: 0.00235906,  19: 0.00072533,  20: 0.00457309,  21: 0.01256962,  22: 0.00559733,  23: 0.0392486,  24: 0.0060228,  25: 0.00908848,  26: 0.01150884,  27: 0.01115191,  28: 0.01161009,  29: 0.01280016,  30: 0.01341943,  31: 0.01367287,  32: 0.01357383,  33: 0.01372479,  34: 0.01486814,  35: 0.01548993,  36: 0.01722397,  37: 0.01798768,  38: 0.00627185,  39: 0.00506561,  40: 0.00597463,  41: 0.00543547,  42: 0.00499733,  43: 0.00452328,  44: 0.00458949,  45: 0.0042845,  46: 0.00425256,  47: 0.00428415,  48: 0.00432716,  49: 0.00439585,  50: 0.00450958,  51: 0.00575839,  52: 0.00739246,  53: 0.00502165,  54: 0.0095109,  55: 0.0069166,  56: 0.0071989,  57: 0.00776389,  58: 0.00814988,  59: 0.0067113,  60: 0.00803184,  61: 0.00822991,  62: 0.00502434,  63: 0.00571724,  64: 0.00985963,  65: 0.00549805,  66: 0.00979168,  67: 0.01288979,  68: 0.01062375,  69: 0.00566553,  70: 0.00503578,  71: 0.01169688,  72: 0.00579325,  73: 0.01161525,  74: 0.01194471,  75: 0.00419361,  76: 0.00416345,  77: 0.0049956,  78: 0.01476468,  79: 0.01368045,  80: 0.00522837,  81: 0.01272689,  82: 0.00422461,  83: 0.01271347,  84: 0.00527942,  85: 0.01348907,  86: 0.01594457,  87: 0.01447534,  88: 0.00454526,  89: 0.00494497,  90: 0.01684405,  91: 0.00459594,  92: 0.00517932,  93: 0.00542252,  94: 0.01753486,  95: 0.00552958,  96: 0.00558431,  97: 0.01830974,  98: 0.01795993,  99: 0.01113913}

    sorted_feature_importance = [(k, feature_importance[k]) for k in sorted(feature_importance, key=feature_importance.get, reverse=True)]

    print(type(sorted_feature_importance))
    for each in sorted_feature_importance:
        print(each)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)



    print(X_train.shape)

    #numpy.set_printoptions(suppress=True)
    bbc = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=100)\
        ,ratio='auto',replacement=False,random_state=0, bootstrap_features=False)



    for i in range(10):
        temp_feat = []
        for j in range(10):
            temp_feat.append(sorted_feature_importance[(i*10)+j][0])
        X_train_new = X_train[:,temp_feat]
        X_test_new =  X_test[:,temp_feat]
        bbc.fit(X_train_new, Y_train)
        #print("For Rank Order",i+1, X_train_new.shape)
        y_pred = bbc.predict(X_test_new)

        #print(confusion_matrix(Y_test, y_pred))
        #print(classification_report(Y_test,y_pred))


        fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred, pos_label=1)
        auc_score = metrics.auc(fpr, tpr)
        #print('auc score =',auc_score)
        print(2*auc_score-1)




Rank_Ordering()


