from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import random
import pickle
import os

#Getting features
methods = ['WHATSAPP_WEB_MAC', 'WHATSAPP_APP_WIN', 'WHATSAPP_WEB_IPAD', 'WHATSAPP_APP_MAC', 'WHATSAPP_IPHONE', 'WHATSAPP_WEB_WIN', 'WHATSAPP_ANDROID']
classes = ['APP-DESKTOP', 'IPHONE', 'WEB-SAFARI', 'WEB-WIN', 'ANDROID']
separated_classes = ['APP-MAC', 'APP-WIN' ,'IPHONE', 'WEB-IPAD', 'WEB-MAC' ,'WEB-WIN', 'ANDROID']

dct = "Results feature extraction/extractor_DCT/"
header = "Results feature extraction/extractor_HEADER/"
meta = "Results feature extraction/extractor_META/"

features = []

for method in methods:
    quality_factors_dct = os.listdir(dct + method + "/")
    quality_factors_header = os.listdir(header + method + "/")
    quality_factors_meta = os.listdir(meta + method + "/")

    #Remove .DS_Store
    quality_factors_dct.pop(0)
    quality_factors_header.pop(0)
    quality_factors_meta.pop(0)

    #Fusing DCT feature with HEADER and META in one vector
    for i in range(6):
        path_dct = dct + method + "/" + quality_factors_dct[i]
        path_header = header + method + "/" + quality_factors_header[i]
        path_meta = meta + method + "/" + quality_factors_meta[i]

        dct_features = os.listdir(path_dct)
        header_features = os.listdir(path_header)
        meta_features = os.listdir(path_meta)

        for j in range(150):
            if dct_features[j].startswith('.'):
                dct_features.pop(j)
            
            if header_features[j].startswith('.'):
                header_features.pop(j)

            if meta_features[j].startswith('.'):
                meta_features.pop(j)

            with open(path_dct + "/" + dct_features[j], "rb") as f1:
                res1 = pickle.load(f1)[0]
            
            with open(path_header + "/" + header_features[j], "rb") as f2:
                res2 = pickle.load(f2)[0]

            with open(path_meta + "/" + meta_features[j], "rb") as f3:
                res3 = pickle.load(f3)[0]

            tmp = np.hstack((res1, res2))
            feature = np.hstack((tmp, res3))
            features.append((feature, method))

features = np.asarray(features, dtype='object')

#Contructing training and testing datasets
whatsapp_web_mac_feature = [i[0] for i in features[0:900]]
whatsapp_web_mac_method = [i[1] for i in features[0:900]]
whatsapp_app_win_feature = [i[0] for i in features[900:1800]]
whatsapp_app_win_method = [i[1] for i in features[900:1800]]
whatsapp_web_ipad_feature = [i[0] for i in features[1800:2700]]
whatsapp_web_ipad_method = [i[1] for i in features[1800:2700]]
whatsapp_app_mac_feature = [i[0] for i in features[2700:3600]]
whatsapp_app_mac_method = [i[1] for i in features[2700:3600]]
whatsapp_iphone_feature = [i[0] for i in features[3600:4500]]
whatsapp_iphone_method = [i[1] for i in features[3600:4500]]
whatsapp_web_win_feature = [i[0] for i in features[4500:5400]]
whatsapp_web_win_method = [i[1] for i in features[4500:5400]]
whatsapp_android_feature = [i[0] for i in features[5400:6300]]
whatsapp_android_method = [i[1] for i in features[5400:6300]]

random.shuffle(whatsapp_web_mac_feature)
random.shuffle(whatsapp_app_win_feature)
random.shuffle(whatsapp_web_ipad_feature)
random.shuffle(whatsapp_app_mac_feature)
random.shuffle(whatsapp_iphone_feature)
random.shuffle(whatsapp_web_win_feature)
random.shuffle(whatsapp_android_feature)

#training_set_x holds the training samples of shape (n_samples, n_features)
training_set_x = []
training_set_x = whatsapp_web_mac_feature[0:450]
training_set_x.extend(whatsapp_app_win_feature[0:450])
training_set_x.extend(whatsapp_web_ipad_feature[0:450])
training_set_x.extend(whatsapp_app_mac_feature[0:450])
training_set_x.extend(whatsapp_iphone_feature[0:450])
training_set_x.extend(whatsapp_web_win_feature[0:450])
training_set_x.extend(whatsapp_android_feature[0:450])
training_set_x = np.asarray(training_set_x)

#training_set_y holds the class labels of shape (n_samples)
training_set_y = []
training_set_y = whatsapp_web_mac_method[0:450]
training_set_y.extend(whatsapp_app_win_method[0:450])
training_set_y.extend(whatsapp_web_ipad_method[0:450])
training_set_y.extend(whatsapp_app_mac_method[0:450])
training_set_y.extend(whatsapp_iphone_method[0:450])
training_set_y.extend(whatsapp_web_win_method[0:450])
training_set_y.extend(whatsapp_android_method[0:450])
training_set_y = np.asarray(training_set_y)

#clf = svm.SVC(kernel = 'rbf', C = 1, gamma = 0.1)
#clf = svm.LinearSVC(penalty = 'l2', dual = False, C = 1)
clf = RandomForestClassifier(n_estimators = 100, min_samples_split = 2, min_samples_leaf = 4, max_features = 'sqrt', max_depth = 10, bootstrap = True)
clf.fit(training_set_x, training_set_y)

#testing_set_x holds the testing samples of shape (n_samples, n_features)
testing_set_x = []
testing_set_x = whatsapp_web_mac_feature[450:900]
testing_set_x.extend(whatsapp_app_win_feature[450:900])
testing_set_x.extend(whatsapp_web_ipad_feature[450:900])
testing_set_x.extend(whatsapp_app_mac_feature[450:900])
testing_set_x.extend(whatsapp_iphone_feature[450:900])
testing_set_x.extend(whatsapp_web_win_feature[450:900])
testing_set_x.extend(whatsapp_android_feature[450:900])

#testing_set_y holds the class labels of shape (n_samples)
testing_set_y = []
testing_set_y = whatsapp_web_mac_method[450:900]
testing_set_y.extend(whatsapp_app_win_method[450:900])
testing_set_y.extend(whatsapp_web_ipad_method[450:900])
testing_set_y.extend(whatsapp_app_mac_method[450:900])
testing_set_y.extend(whatsapp_iphone_method[450:900])
testing_set_y.extend(whatsapp_web_win_method[450:900])
testing_set_y.extend(whatsapp_android_method[450:900])

classifier_predictions = clf.predict(testing_set_x)

#Compute accuracy for each sharing method
web_mac = 0
app_win = 0
web_ipad = 0
app_mac = 0
iphone = 0
web_win = 0
android = 0

for i in range(3150):
    if i < 450:
        if testing_set_y[i] == classifier_predictions[i]:
            web_mac = web_mac + 1
    
    elif i >= 450 and i < 900:
        if testing_set_y[i] == classifier_predictions[i]:
            app_win = app_win + 1

    elif i >= 900 and i < 1350:
        if testing_set_y[i] == classifier_predictions[i]:
            web_ipad = web_ipad + 1

    elif i >= 1350 and i < 1800:
        if testing_set_y[i] == classifier_predictions[i]:
            app_mac = app_mac + 1

    elif i >= 1800 and i < 2250:
        if testing_set_y[i] == classifier_predictions[i]:
            iphone = iphone + 1

    elif i >= 2250 and i < 2700:
        if testing_set_y[i] == classifier_predictions[i]:
            web_win = web_win + 1

    elif i >= 2700 and i < 3150:
        if testing_set_y[i] == classifier_predictions[i]:
            android = android + 1

#Overall accuracy
#print(accuracy_score(testing_set_y, classifier_predictions) * 100)

#Accuracy for separated classes
#print("Accuracy APP-DESKTOP: ", ((app_mac / 450) * 100) + ((app_win / 450) * 100))
#print("Accuracy IPHONE: ", (iphone / 450) * 100)
#print("Accuracy WEB-SAFARI: ", ((web_mac / 450) * 100) + ((web_ipad / 450) * 100))
#print("Accuracy WEB-WIN: ", (web_win / 450) * 100)
#print("Accuracy ANDROID: ", (android / 450) * 100)

#Renaiming classes
def renaiming_list_elements(arr, dim, merge):
    if merge:
        for i in range(dim):
            if arr[i] == 'WHATSAPP_WEB_MAC':
                arr[i] = 'WEB-SAFARI'
            elif arr[i] == 'WHATSAPP_WEB_IPAD':
                arr[i] = 'WEB-SAFARI'
            elif arr[i] == 'WHATSAPP_APP_MAC':
                arr[i] = 'APP-DESKTOP'
            elif arr[i] == 'WHATSAPP_APP_WIN':
                arr[i] = 'APP-DESKTOP'
            elif arr[i] == 'WHATSAPP_IPHONE':
                arr[i] = 'IPHONE'
            elif arr[i] == 'WHATSAPP_WEB_WIN':
                arr[i] = 'WEB-WIN'
            elif arr[i] == 'WHATSAPP_ANDROID':
                arr[i] = 'ANDROID'
        
    else:
        for i in range(dim):
            if arr[i] == 'WHATSAPP_WEB_MAC':
                arr[i] = 'WEB-MAC'
            elif arr[i] == 'WHATSAPP_WEB_IPAD':
                arr[i] = 'WEB-IPAD'
            elif arr[i] == 'WHATSAPP_APP_MAC':
                arr[i] = 'APP-MAC'
            elif arr[i] == 'WHATSAPP_APP_WIN':
                arr[i] = 'APP-WIN'
            elif arr[i] == 'WHATSAPP_IPHONE':
                arr[i] = 'IPHONE'
            elif arr[i] == 'WHATSAPP_WEB_WIN':
                arr[i] = 'WEB-WIN'
            elif arr[i] == 'WHATSAPP_ANDROID':
                arr[i] = 'ANDROID'

    return arr

testing_set_y_separated_classes = testing_set_y.copy()
testing_set_y_separated_classes = renaiming_list_elements(testing_set_y_separated_classes, 3150, False)
testing_set_y = renaiming_list_elements(testing_set_y, 3150, True)

classifier_predictions_separated_classes = classifier_predictions.copy()
classifier_predictions_separated_classes = renaiming_list_elements(classifier_predictions_separated_classes, 3150, False)
classifier_predictions = renaiming_list_elements(classifier_predictions, 3150, True)

#Confusion matrix with merged classes
cm = confusion_matrix(testing_set_y, classifier_predictions, labels = classes, normalize = 'true')
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)

fig, ax = plt.subplots(figsize = (11,9))
plt.rcParams.update({'font.size': 21}) #Update font for the entire confusion matrix
disp.plot(ax = ax, cmap = plt.cm.Blues, colorbar = False)

plt.setp(ax.get_yticklabels(), fontsize = 18) #Update font for y axis
plt.setp(ax.get_xticklabels(), fontsize = 18, rotation = 10) #Update font for x axis
plt.ylabel('True label', fontsize = 18, weight = 'bold') #Update font for "True label"
plt.xlabel('Predicted label', fontsize = 18, weight = 'bold') #Update font for "Predicted label"
plt.savefig("Results SVM - RF classification/confusion_matrix_RF.pdf")
plt.show()

#Confusion matrix with un-merged classes
cm = confusion_matrix(testing_set_y_separated_classes, classifier_predictions_separated_classes, labels = separated_classes, normalize = 'true')
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = separated_classes)

fig, ax = plt.subplots(figsize = (12,11))
plt.rcParams.update({'font.size': 21}) #Update font for the entire confusion matrix
disp.plot(ax = ax, cmap = plt.cm.Blues, colorbar = False)

plt.setp(ax.get_yticklabels(), fontsize = 18) #Update font for y axis
plt.setp(ax.get_xticklabels(), fontsize = 18, rotation = 15) #Update font for x axis
plt.ylabel('True label', fontsize = 18, weight = 'bold') #Update font for "True label"
plt.xlabel('Predicted label', fontsize = 18, weight = 'bold') #Update font for "Predicted label"
plt.savefig("Results SVM - RF classification/confusion_matrix_RF_sep_class.pdf")
plt.show()

def QF_50():
    qf_50_train_x = []
    qf_50_train_y = []
    qf_50_test_x = []
    qf_50_test_y = []

    qf_50_web_mac = features[150:300]
    qf_50_app_win = features[1050:1200]
    qf_50_web_ipad = features[1950:2100]
    qf_50_app_mac = features[2850:3000]
    qf_50_iphone = features[3750:3900]
    qf_50_web_win = features[4650:4800]
    qf_50_android = features[5550:5700]

    random.shuffle(qf_50_web_mac)
    random.shuffle(qf_50_app_win)
    random.shuffle(qf_50_web_ipad)
    random.shuffle(qf_50_app_mac)
    random.shuffle(qf_50_iphone)
    random.shuffle(qf_50_web_win)
    random.shuffle(qf_50_android)

    qf_50_train_x.extend([i[0] for i in qf_50_web_mac[0:75]])
    qf_50_train_y.extend([i[1] for i in qf_50_web_mac[0:75]])
    qf_50_train_x.extend([i[0] for i in qf_50_app_win[0:75]])
    qf_50_train_y.extend([i[1] for i in qf_50_app_win[0:75]])
    qf_50_train_x.extend([i[0] for i in qf_50_web_ipad[0:75]])
    qf_50_train_y.extend([i[1] for i in qf_50_web_ipad[0:75]])
    qf_50_train_x.extend([i[0] for i in qf_50_app_mac[0:75]])
    qf_50_train_y.extend([i[1] for i in qf_50_app_mac[0:75]])
    qf_50_train_x.extend([i[0] for i in qf_50_iphone[0:75]])
    qf_50_train_y.extend([i[1] for i in qf_50_iphone[0:75]])
    qf_50_train_x.extend([i[0] for i in qf_50_web_win[0:75]])
    qf_50_train_y.extend([i[1] for i in qf_50_web_win[0:75]])
    qf_50_train_x.extend([i[0] for i in qf_50_android[0:75]])
    qf_50_train_y.extend([i[1] for i in qf_50_android[0:75]])

    qf_50_test_x.extend([i[0] for i in qf_50_web_mac[75:150]])
    qf_50_test_y.extend([i[1] for i in qf_50_web_mac[75:150]])
    qf_50_test_x.extend([i[0] for i in qf_50_app_win[75:150]])
    qf_50_test_y.extend([i[1] for i in qf_50_app_win[75:150]])
    qf_50_test_x.extend([i[0] for i in qf_50_web_ipad[75:150]])
    qf_50_test_y.extend([i[1] for i in qf_50_web_ipad[75:150]])
    qf_50_test_x.extend([i[0] for i in qf_50_app_mac[75:150]])
    qf_50_test_y.extend([i[1] for i in qf_50_app_mac[75:150]])
    qf_50_test_x.extend([i[0] for i in qf_50_iphone[75:150]])
    qf_50_test_y.extend([i[1] for i in qf_50_iphone[75:150]])
    qf_50_test_x.extend([i[0] for i in qf_50_web_win[75:150]])
    qf_50_test_y.extend([i[1] for i in qf_50_web_win[75:150]])
    qf_50_test_x.extend([i[0] for i in qf_50_android[75:150]])
    qf_50_test_y.extend([i[1] for i in qf_50_android[75:150]])

    #clf = svm.SVC(kernel = 'rbf', C = 1, gamma = 0.1)
    #clf = svm.LinearSVC(penalty = 'l2', dual = False, C = 1)
    clf = RandomForestClassifier(n_estimators = 100, min_samples_split = 2, min_samples_leaf = 4, max_features = 'sqrt', max_depth = 10, bootstrap = True)
    clf.fit(qf_50_train_x, qf_50_train_y)
    classifier_predictions = clf.predict(qf_50_test_x)

    #Renaiming classes
    qf_50_test_y_separated_classes = qf_50_test_y.copy()
    qf_50_test_y_separated_classes = renaiming_list_elements(qf_50_test_y_separated_classes, 525, False)
    qf_50_test_y = renaiming_list_elements(qf_50_test_y, 525, True)

    classifier_predictions_separated_classes = classifier_predictions.copy()
    classifier_predictions_separated_classes = renaiming_list_elements(classifier_predictions_separated_classes, 525, False)
    classifier_predictions = renaiming_list_elements(classifier_predictions, 525, True)

    #Confusion matrix with merged classes
    cm = confusion_matrix(qf_50_test_y, classifier_predictions, labels = classes, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)

    fig, ax = plt.subplots(figsize = (11,9))
    plt.rcParams.update({'font.size': 21}) #Update font for the entire confusion matrix
    ax.set_title("QF-50")
    disp.plot(ax = ax, cmap = plt.cm.Blues, colorbar = False)

    plt.setp(ax.get_yticklabels(), fontsize = 18) #Update font for y axis
    plt.setp(ax.get_xticklabels(), fontsize = 18, rotation = 10) #Update font for x axis
    plt.ylabel('True label', fontsize = 18, weight = 'bold') #Update font for "True label"
    plt.xlabel('Predicted label', fontsize = 18, weight = 'bold') #Update font for "Predicted label"
    plt.savefig("Results SVM - RF classification/confusion_matrix_RF_QF-50.pdf")
    plt.show()

    #Confusion matrix with un-merged classes
    cm = confusion_matrix(qf_50_test_y_separated_classes, classifier_predictions_separated_classes, labels = separated_classes, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = separated_classes)

    fig, ax = plt.subplots(figsize = (12,11))
    plt.rcParams.update({'font.size': 21}) #Update font for the entire confusion matrix
    ax.set_title("QF-50")
    disp.plot(ax = ax, cmap = plt.cm.Blues, colorbar = False)

    plt.setp(ax.get_yticklabels(), fontsize = 18) #Update font for y axis
    plt.setp(ax.get_xticklabels(), fontsize = 18, rotation = 15) #Update font for x axis
    plt.ylabel('True label', fontsize = 18, weight = 'bold') #Update font for "True label"
    plt.xlabel('Predicted label', fontsize = 18, weight = 'bold') #Update font for "Predicted label"
    plt.savefig("Results SVM - RF classification/confusion_matrix_RF_sep_class_QF-50.pdf")
    plt.show()

def QF_60():
    qf_60_train_x = []
    qf_60_train_y = []
    qf_60_test_x = []
    qf_60_test_y = []

    qf_60_web_mac = features[300:450]
    qf_60_app_win = features[1200:1350]
    qf_60_web_ipad = features[2100:2250]
    qf_60_app_mac = features[3000:3150]
    qf_60_iphone = features[3900:4050]
    qf_60_web_win = features[4800:4950]
    qf_60_android = features[5700:5850]

    random.shuffle(qf_60_web_mac)
    random.shuffle(qf_60_app_win)
    random.shuffle(qf_60_web_ipad)
    random.shuffle(qf_60_app_mac)
    random.shuffle(qf_60_iphone)
    random.shuffle(qf_60_web_win)
    random.shuffle(qf_60_android)

    qf_60_train_x.extend([i[0] for i in qf_60_web_mac[0:75]])
    qf_60_train_y.extend([i[1] for i in qf_60_web_mac[0:75]])
    qf_60_train_x.extend([i[0] for i in qf_60_app_win[0:75]])
    qf_60_train_y.extend([i[1] for i in qf_60_app_win[0:75]])
    qf_60_train_x.extend([i[0] for i in qf_60_web_ipad[0:75]])
    qf_60_train_y.extend([i[1] for i in qf_60_web_ipad[0:75]])
    qf_60_train_x.extend([i[0] for i in qf_60_app_mac[0:75]])
    qf_60_train_y.extend([i[1] for i in qf_60_app_mac[0:75]])
    qf_60_train_x.extend([i[0] for i in qf_60_iphone[0:75]])
    qf_60_train_y.extend([i[1] for i in qf_60_iphone[0:75]])
    qf_60_train_x.extend([i[0] for i in qf_60_web_win[0:75]])
    qf_60_train_y.extend([i[1] for i in qf_60_web_win[0:75]])
    qf_60_train_x.extend([i[0] for i in qf_60_android[0:75]])
    qf_60_train_y.extend([i[1] for i in qf_60_android[0:75]])

    qf_60_test_x.extend([i[0] for i in qf_60_web_mac[75:150]])
    qf_60_test_y.extend([i[1] for i in qf_60_web_mac[75:150]])
    qf_60_test_x.extend([i[0] for i in qf_60_app_win[75:150]])
    qf_60_test_y.extend([i[1] for i in qf_60_app_win[75:150]])
    qf_60_test_x.extend([i[0] for i in qf_60_web_ipad[75:150]])
    qf_60_test_y.extend([i[1] for i in qf_60_web_ipad[75:150]])
    qf_60_test_x.extend([i[0] for i in qf_60_app_mac[75:150]])
    qf_60_test_y.extend([i[1] for i in qf_60_app_mac[75:150]])
    qf_60_test_x.extend([i[0] for i in qf_60_iphone[75:150]])
    qf_60_test_y.extend([i[1] for i in qf_60_iphone[75:150]])
    qf_60_test_x.extend([i[0] for i in qf_60_web_win[75:150]])
    qf_60_test_y.extend([i[1] for i in qf_60_web_win[75:150]])
    qf_60_test_x.extend([i[0] for i in qf_60_android[75:150]])
    qf_60_test_y.extend([i[1] for i in qf_60_android[75:150]])

    #clf = svm.SVC(kernel = 'rbf', C = 1, gamma = 0.1)
    #clf = svm.LinearSVC(penalty = 'l2', dual = False, C = 1)
    clf = RandomForestClassifier(n_estimators = 100, min_samples_split = 2, min_samples_leaf = 4, max_features = 'sqrt', max_depth = 10, bootstrap = True)
    clf.fit(qf_60_train_x, qf_60_train_y)
    classifier_predictions = clf.predict(qf_60_test_x)

    #Renaiming classes
    qf_60_test_y_separated_classes = qf_60_test_y.copy()
    qf_60_test_y_separated_classes = renaiming_list_elements(qf_60_test_y_separated_classes, 525, False)
    qf_60_test_y = renaiming_list_elements(qf_60_test_y, 525, True)

    classifier_predictions_separated_classes = classifier_predictions.copy()
    classifier_predictions_separated_classes = renaiming_list_elements(classifier_predictions_separated_classes, 525, False)
    classifier_predictions = renaiming_list_elements(classifier_predictions, 525, True)

    #Confusion matrix with merged classes
    cm = confusion_matrix(qf_60_test_y, classifier_predictions, labels = classes, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)

    fig, ax = plt.subplots(figsize = (11,9))
    plt.rcParams.update({'font.size': 21}) #Update font for the entire confusion matrix
    ax.set_title("QF-60")
    disp.plot(ax = ax, cmap = plt.cm.Blues, colorbar = False)

    plt.setp(ax.get_yticklabels(), fontsize = 18) #Update font for y axis
    plt.setp(ax.get_xticklabels(), fontsize = 18, rotation = 10) #Update font for x axis
    plt.ylabel('True label', fontsize = 18, weight = 'bold') #Update font for "True label"
    plt.xlabel('Predicted label', fontsize = 18, weight = 'bold') #Update font for "Predicted label"
    plt.savefig("Results SVM - RF classification/confusion_matrix_RF_QF-60.pdf")
    plt.show()

    #Confusion matrix with un-merged classes
    cm = confusion_matrix(qf_60_test_y_separated_classes, classifier_predictions_separated_classes, labels = separated_classes, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = separated_classes)

    fig, ax = plt.subplots(figsize = (12,11))
    plt.rcParams.update({'font.size': 21}) #Update font for the entire confusion matrix
    ax.set_title("QF-60")
    disp.plot(ax = ax, cmap = plt.cm.Blues, colorbar = False)

    plt.setp(ax.get_yticklabels(), fontsize = 18) #Update font for y axis
    plt.setp(ax.get_xticklabels(), fontsize = 18, rotation = 15) #Update font for x axis
    plt.ylabel('True label', fontsize = 18, weight = 'bold') #Update font for "True label"
    plt.xlabel('Predicted label', fontsize = 18, weight = 'bold') #Update font for "Predicted label"
    plt.savefig("Results SVM - RF classification/confusion_matrix_RF_sep_class_QF-60.pdf")
    plt.show()

def QF_70():
    qf_70_train_x = []
    qf_70_train_y = []
    qf_70_test_x = []
    qf_70_test_y = []

    qf_70_web_mac = features[450:600]
    qf_70_app_win = features[1350:1500]
    qf_70_web_ipad = features[2250:2400]
    qf_70_app_mac = features[3150:3300]
    qf_70_iphone = features[4050:4200]
    qf_70_web_win = features[4950:5100]
    qf_70_android = features[5850:6000]

    random.shuffle(qf_70_web_mac)
    random.shuffle(qf_70_app_win)
    random.shuffle(qf_70_web_ipad)
    random.shuffle(qf_70_app_mac)
    random.shuffle(qf_70_iphone)
    random.shuffle(qf_70_web_win)
    random.shuffle(qf_70_android)

    qf_70_train_x.extend([i[0] for i in qf_70_web_mac[0:75]])
    qf_70_train_y.extend([i[1] for i in qf_70_web_mac[0:75]])
    qf_70_train_x.extend([i[0] for i in qf_70_app_win[0:75]])
    qf_70_train_y.extend([i[1] for i in qf_70_app_win[0:75]])
    qf_70_train_x.extend([i[0] for i in qf_70_web_ipad[0:75]])
    qf_70_train_y.extend([i[1] for i in qf_70_web_ipad[0:75]])
    qf_70_train_x.extend([i[0] for i in qf_70_app_mac[0:75]])
    qf_70_train_y.extend([i[1] for i in qf_70_app_mac[0:75]])
    qf_70_train_x.extend([i[0] for i in qf_70_iphone[0:75]])
    qf_70_train_y.extend([i[1] for i in qf_70_iphone[0:75]])
    qf_70_train_x.extend([i[0] for i in qf_70_web_win[0:75]])
    qf_70_train_y.extend([i[1] for i in qf_70_web_win[0:75]])
    qf_70_train_x.extend([i[0] for i in qf_70_android[0:75]])
    qf_70_train_y.extend([i[1] for i in qf_70_android[0:75]])

    qf_70_test_x.extend([i[0] for i in qf_70_web_mac[75:150]])
    qf_70_test_y.extend([i[1] for i in qf_70_web_mac[75:150]])
    qf_70_test_x.extend([i[0] for i in qf_70_app_win[75:150]])
    qf_70_test_y.extend([i[1] for i in qf_70_app_win[75:150]])
    qf_70_test_x.extend([i[0] for i in qf_70_web_ipad[75:150]])
    qf_70_test_y.extend([i[1] for i in qf_70_web_ipad[75:150]])
    qf_70_test_x.extend([i[0] for i in qf_70_app_mac[75:150]])
    qf_70_test_y.extend([i[1] for i in qf_70_app_mac[75:150]])
    qf_70_test_x.extend([i[0] for i in qf_70_iphone[75:150]])
    qf_70_test_y.extend([i[1] for i in qf_70_iphone[75:150]])
    qf_70_test_x.extend([i[0] for i in qf_70_web_win[75:150]])
    qf_70_test_y.extend([i[1] for i in qf_70_web_win[75:150]])
    qf_70_test_x.extend([i[0] for i in qf_70_android[75:150]])
    qf_70_test_y.extend([i[1] for i in qf_70_android[75:150]])

    #clf = svm.SVC(kernel = 'rbf', C = 1, gamma = 0.1)
    #clf = svm.LinearSVC(penalty = 'l2', dual = False, C = 1)
    clf = RandomForestClassifier(n_estimators = 100, min_samples_split = 2, min_samples_leaf = 4, max_features = 'sqrt', max_depth = 10, bootstrap = True)
    clf.fit(qf_70_train_x, qf_70_train_y)
    classifier_predictions = clf.predict(qf_70_test_x)

    #Renaiming classes
    qf_70_test_y_separated_classes = qf_70_test_y.copy()
    qf_70_test_y_separated_classes = renaiming_list_elements(qf_70_test_y_separated_classes, 525, False)
    qf_70_test_y = renaiming_list_elements(qf_70_test_y, 525, True)

    classifier_predictions_separated_classes = classifier_predictions.copy()
    classifier_predictions_separated_classes = renaiming_list_elements(classifier_predictions_separated_classes, 525, False)
    classifier_predictions = renaiming_list_elements(classifier_predictions, 525, True)

    #Confusion matrix with merged classes
    cm = confusion_matrix(qf_70_test_y, classifier_predictions, labels = classes, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)

    fig, ax = plt.subplots(figsize = (11,9))
    plt.rcParams.update({'font.size': 21}) #Update font for the entire confusion matrix
    ax.set_title("QF-70")
    disp.plot(ax = ax, cmap = plt.cm.Blues, colorbar = False)

    plt.setp(ax.get_yticklabels(), fontsize = 18) #Update font for y axis
    plt.setp(ax.get_xticklabels(), fontsize = 18, rotation = 10) #Update font for x axis
    plt.ylabel('True label', fontsize = 18, weight = 'bold') #Update font for "True label"
    plt.xlabel('Predicted label', fontsize = 18, weight = 'bold') #Update font for "Predicted label"
    plt.savefig("Results SVM - RF classification/confusion_matrix_RF_QF-70.pdf")
    plt.show()

    #Confusion matrix with un-merged classes
    cm = confusion_matrix(qf_70_test_y_separated_classes, classifier_predictions_separated_classes, labels = separated_classes, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = separated_classes)

    fig, ax = plt.subplots(figsize = (12,11))
    plt.rcParams.update({'font.size': 21}) #Update font for the entire confusion matrix
    ax.set_title("QF-70")
    disp.plot(ax = ax, cmap = plt.cm.Blues, colorbar = False)

    plt.setp(ax.get_yticklabels(), fontsize = 18) #Update font for y axis
    plt.setp(ax.get_xticklabels(), fontsize = 18, rotation = 15) #Update font for x axis
    plt.ylabel('True label', fontsize = 18, weight = 'bold') #Update font for "True label"
    plt.xlabel('Predicted label', fontsize = 18, weight = 'bold') #Update font for "Predicted label"
    plt.savefig("Results SVM - RF classification/confusion_matrix_RF_sep_class_QF-70.pdf")
    plt.show()

def QF_80():
    qf_80_train_x = []
    qf_80_train_y = []
    qf_80_test_x = []
    qf_80_test_y = []

    qf_80_web_mac = features[0:150]
    qf_80_app_win = features[900:1050]
    qf_80_web_ipad = features[1800:1950]
    qf_80_app_mac = features[2700:2850]
    qf_80_iphone = features[3600:3750]
    qf_80_web_win = features[4500:4650]
    qf_80_android = features[5400:5550]

    random.shuffle(qf_80_web_mac)
    random.shuffle(qf_80_app_win)
    random.shuffle(qf_80_web_ipad)
    random.shuffle(qf_80_app_mac)
    random.shuffle(qf_80_iphone)
    random.shuffle(qf_80_web_win)
    random.shuffle(qf_80_android)

    qf_80_train_x.extend([i[0] for i in qf_80_web_mac[0:75]])
    qf_80_train_y.extend([i[1] for i in qf_80_web_mac[0:75]])
    qf_80_train_x.extend([i[0] for i in qf_80_app_win[0:75]])
    qf_80_train_y.extend([i[1] for i in qf_80_app_win[0:75]])
    qf_80_train_x.extend([i[0] for i in qf_80_web_ipad[0:75]])
    qf_80_train_y.extend([i[1] for i in qf_80_web_ipad[0:75]])
    qf_80_train_x.extend([i[0] for i in qf_80_app_mac[0:75]])
    qf_80_train_y.extend([i[1] for i in qf_80_app_mac[0:75]])
    qf_80_train_x.extend([i[0] for i in qf_80_iphone[0:75]])
    qf_80_train_y.extend([i[1] for i in qf_80_iphone[0:75]])
    qf_80_train_x.extend([i[0] for i in qf_80_web_win[0:75]])
    qf_80_train_y.extend([i[1] for i in qf_80_web_win[0:75]])
    qf_80_train_x.extend([i[0] for i in qf_80_android[0:75]])
    qf_80_train_y.extend([i[1] for i in qf_80_android[0:75]])

    qf_80_test_x.extend([i[0] for i in qf_80_web_mac[75:150]])
    qf_80_test_y.extend([i[1] for i in qf_80_web_mac[75:150]])
    qf_80_test_x.extend([i[0] for i in qf_80_app_win[75:150]])
    qf_80_test_y.extend([i[1] for i in qf_80_app_win[75:150]])
    qf_80_test_x.extend([i[0] for i in qf_80_web_ipad[75:150]])
    qf_80_test_y.extend([i[1] for i in qf_80_web_ipad[75:150]])
    qf_80_test_x.extend([i[0] for i in qf_80_app_mac[75:150]])
    qf_80_test_y.extend([i[1] for i in qf_80_app_mac[75:150]])
    qf_80_test_x.extend([i[0] for i in qf_80_iphone[75:150]])
    qf_80_test_y.extend([i[1] for i in qf_80_iphone[75:150]])
    qf_80_test_x.extend([i[0] for i in qf_80_web_win[75:150]])
    qf_80_test_y.extend([i[1] for i in qf_80_web_win[75:150]])
    qf_80_test_x.extend([i[0] for i in qf_80_android[75:150]])
    qf_80_test_y.extend([i[1] for i in qf_80_android[75:150]])

    #clf = svm.SVC(kernel = 'rbf', C = 1, gamma = 0.1)
    #clf = svm.LinearSVC(penalty = 'l2', dual = False, C = 1)
    clf = RandomForestClassifier(n_estimators = 100, min_samples_split = 2, min_samples_leaf = 4, max_features = 'sqrt', max_depth = 10, bootstrap = True)
    clf.fit(qf_80_train_x, qf_80_train_y)
    classifier_predictions = clf.predict(qf_80_test_x)

    #Renaiming classes
    qf_80_test_y_separated_classes = qf_80_test_y.copy()
    qf_80_test_y_separated_classes = renaiming_list_elements(qf_80_test_y_separated_classes, 525, False)
    qf_80_test_y = renaiming_list_elements(qf_80_test_y, 525, True)

    classifier_predictions_separated_classes = classifier_predictions.copy()
    classifier_predictions_separated_classes = renaiming_list_elements(classifier_predictions_separated_classes, 525, False)
    classifier_predictions = renaiming_list_elements(classifier_predictions, 525, True)

    #Confusion matrix with merged classes
    cm = confusion_matrix(qf_80_test_y, classifier_predictions, labels = classes, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)

    fig, ax = plt.subplots(figsize = (11,9))
    plt.rcParams.update({'font.size': 21}) #Update font for the entire confusion matrix
    ax.set_title("QF-80")
    disp.plot(ax = ax, cmap = plt.cm.Blues, colorbar = False)

    plt.setp(ax.get_yticklabels(), fontsize = 18) #Update font for y axis
    plt.setp(ax.get_xticklabels(), fontsize = 18, rotation = 10) #Update font for x axis
    plt.ylabel('True label', fontsize = 18, weight = 'bold') #Update font for "True label"
    plt.xlabel('Predicted label', fontsize = 18, weight = 'bold') #Update font for "Predicted label"
    plt.savefig("Results SVM - RF classification/confusion_matrix_RF_QF-80.pdf")
    plt.show()

    #Confusion matrix with un-merged classes
    cm = confusion_matrix(qf_80_test_y_separated_classes, classifier_predictions_separated_classes, labels = separated_classes, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = separated_classes)

    fig, ax = plt.subplots(figsize = (12,11))
    plt.rcParams.update({'font.size': 21}) #Update font for the entire confusion matrix
    ax.set_title("QF-80")
    disp.plot(ax = ax, cmap = plt.cm.Blues, colorbar = False)

    plt.setp(ax.get_yticklabels(), fontsize = 18) #Update font for y axis
    plt.setp(ax.get_xticklabels(), fontsize = 18, rotation = 15) #Update font for x axis
    plt.ylabel('True label', fontsize = 18, weight = 'bold') #Update font for "True label"
    plt.xlabel('Predicted label', fontsize = 18, weight = 'bold') #Update font for "Predicted label"
    plt.savefig("Results SVM - RF classification/confusion_matrix_RF_sep_class_QF-80.pdf")
    plt.show()

def QF_90():
    qf_90_train_x = []
    qf_90_train_y = []
    qf_90_test_x = []
    qf_90_test_y = []

    qf_90_web_mac = features[750:900]
    qf_90_app_win = features[1650:1800]
    qf_90_web_ipad = features[2550:2700]
    qf_90_app_mac = features[3450:3600]
    qf_90_iphone = features[4350:4500]
    qf_90_web_win = features[5250:5400]
    qf_90_android = features[6150:6300]

    random.shuffle(qf_90_web_mac)
    random.shuffle(qf_90_app_win)
    random.shuffle(qf_90_web_ipad)
    random.shuffle(qf_90_app_mac)
    random.shuffle(qf_90_iphone)
    random.shuffle(qf_90_web_win)
    random.shuffle(qf_90_android)

    qf_90_train_x.extend([i[0] for i in qf_90_web_mac[0:75]])
    qf_90_train_y.extend([i[1] for i in qf_90_web_mac[0:75]])
    qf_90_train_x.extend([i[0] for i in qf_90_app_win[0:75]])
    qf_90_train_y.extend([i[1] for i in qf_90_app_win[0:75]])
    qf_90_train_x.extend([i[0] for i in qf_90_web_ipad[0:75]])
    qf_90_train_y.extend([i[1] for i in qf_90_web_ipad[0:75]])
    qf_90_train_x.extend([i[0] for i in qf_90_app_mac[0:75]])
    qf_90_train_y.extend([i[1] for i in qf_90_app_mac[0:75]])
    qf_90_train_x.extend([i[0] for i in qf_90_iphone[0:75]])
    qf_90_train_y.extend([i[1] for i in qf_90_iphone[0:75]])
    qf_90_train_x.extend([i[0] for i in qf_90_web_win[0:75]])
    qf_90_train_y.extend([i[1] for i in qf_90_web_win[0:75]])
    qf_90_train_x.extend([i[0] for i in qf_90_android[0:75]])
    qf_90_train_y.extend([i[1] for i in qf_90_android[0:75]])

    qf_90_test_x.extend([i[0] for i in qf_90_web_mac[75:150]])
    qf_90_test_y.extend([i[1] for i in qf_90_web_mac[75:150]])
    qf_90_test_x.extend([i[0] for i in qf_90_app_win[75:150]])
    qf_90_test_y.extend([i[1] for i in qf_90_app_win[75:150]])
    qf_90_test_x.extend([i[0] for i in qf_90_web_ipad[75:150]])
    qf_90_test_y.extend([i[1] for i in qf_90_web_ipad[75:150]])
    qf_90_test_x.extend([i[0] for i in qf_90_app_mac[75:150]])
    qf_90_test_y.extend([i[1] for i in qf_90_app_mac[75:150]])
    qf_90_test_x.extend([i[0] for i in qf_90_iphone[75:150]])
    qf_90_test_y.extend([i[1] for i in qf_90_iphone[75:150]])
    qf_90_test_x.extend([i[0] for i in qf_90_web_win[75:150]])
    qf_90_test_y.extend([i[1] for i in qf_90_web_win[75:150]])
    qf_90_test_x.extend([i[0] for i in qf_90_android[75:150]])
    qf_90_test_y.extend([i[1] for i in qf_90_android[75:150]])

    #clf = svm.SVC(kernel = 'rbf', C = 1, gamma = 0.1)
    #clf = svm.LinearSVC(penalty = 'l2', dual = False, C = 1)
    clf = RandomForestClassifier(n_estimators = 100, min_samples_split = 2, min_samples_leaf = 4, max_features = 'sqrt', max_depth = 10, bootstrap = True)
    clf.fit(qf_90_train_x, qf_90_train_y)
    classifier_predictions = clf.predict(qf_90_test_x)

    #Renaiming classes
    qf_90_test_y_separated_classes = qf_90_test_y.copy()
    qf_90_test_y_separated_classes = renaiming_list_elements(qf_90_test_y_separated_classes, 525, False)
    qf_90_test_y = renaiming_list_elements(qf_90_test_y, 525, True)

    classifier_predictions_separated_classes = classifier_predictions.copy()
    classifier_predictions_separated_classes = renaiming_list_elements(classifier_predictions_separated_classes, 525, False)
    classifier_predictions = renaiming_list_elements(classifier_predictions, 525, True)

    #Confusion matrix with merged classes
    cm = confusion_matrix(qf_90_test_y, classifier_predictions, labels = classes, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)

    fig, ax = plt.subplots(figsize = (11,9))
    plt.rcParams.update({'font.size': 21}) #Update font for the entire confusion matrix
    ax.set_title("QF-90")
    disp.plot(ax = ax, cmap = plt.cm.Blues, colorbar = False)

    plt.setp(ax.get_yticklabels(), fontsize = 18) #Update font for y axis
    plt.setp(ax.get_xticklabels(), fontsize = 18, rotation = 10) #Update font for x axis
    plt.ylabel('True label', fontsize = 18, weight = 'bold') #Update font for "True label"
    plt.xlabel('Predicted label', fontsize = 18, weight = 'bold') #Update font for "Predicted label"
    plt.savefig("Results SVM - RF classification/confusion_matrix_RF_QF-90.pdf")
    plt.show()

    #Confusion matrix with un-merged classes
    cm = confusion_matrix(qf_90_test_y_separated_classes, classifier_predictions_separated_classes, labels = separated_classes, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = separated_classes)

    fig, ax = plt.subplots(figsize = (12,11))
    plt.rcParams.update({'font.size': 21}) #Update font for the entire confusion matrix
    ax.set_title("QF-90")
    disp.plot(ax = ax, cmap = plt.cm.Blues, colorbar = False)

    plt.setp(ax.get_yticklabels(), fontsize = 18) #Update font for y axis
    plt.setp(ax.get_xticklabels(), fontsize = 18, rotation = 15) #Update font for x axis
    plt.ylabel('True label', fontsize = 18, weight = 'bold') #Update font for "True label"
    plt.xlabel('Predicted label', fontsize = 18, weight = 'bold') #Update font for "Predicted label"
    plt.savefig("Results SVM - RF classification/confusion_matrix_RF_sep_class_QF-90.pdf")
    plt.show()

def QF_100():
    qf_100_train_x = []
    qf_100_train_y = []
    qf_100_test_x = []
    qf_100_test_y = []

    qf_100_web_mac = features[600:750]
    qf_100_app_win = features[1500:1650]
    qf_100_web_ipad = features[2400:2550]
    qf_100_app_mac = features[3300:3450]
    qf_100_iphone = features[4200:4350]
    qf_100_web_win = features[5100:5250]
    qf_100_android = features[6000:6150]

    random.shuffle(qf_100_web_mac)
    random.shuffle(qf_100_app_win)
    random.shuffle(qf_100_web_ipad)
    random.shuffle(qf_100_app_mac)
    random.shuffle(qf_100_iphone)
    random.shuffle(qf_100_web_win)
    random.shuffle(qf_100_android)

    qf_100_train_x.extend([i[0] for i in qf_100_web_mac[0:75]])
    qf_100_train_y.extend([i[1] for i in qf_100_web_mac[0:75]])
    qf_100_train_x.extend([i[0] for i in qf_100_app_win[0:75]])
    qf_100_train_y.extend([i[1] for i in qf_100_app_win[0:75]])
    qf_100_train_x.extend([i[0] for i in qf_100_web_ipad[0:75]])
    qf_100_train_y.extend([i[1] for i in qf_100_web_ipad[0:75]])
    qf_100_train_x.extend([i[0] for i in qf_100_app_mac[0:75]])
    qf_100_train_y.extend([i[1] for i in qf_100_app_mac[0:75]])
    qf_100_train_x.extend([i[0] for i in qf_100_iphone[0:75]])
    qf_100_train_y.extend([i[1] for i in qf_100_iphone[0:75]])
    qf_100_train_x.extend([i[0] for i in qf_100_web_win[0:75]])
    qf_100_train_y.extend([i[1] for i in qf_100_web_win[0:75]])
    qf_100_train_x.extend([i[0] for i in qf_100_android[0:75]])
    qf_100_train_y.extend([i[1] for i in qf_100_android[0:75]])

    qf_100_test_x.extend([i[0] for i in qf_100_web_mac[75:150]])
    qf_100_test_y.extend([i[1] for i in qf_100_web_mac[75:150]])
    qf_100_test_x.extend([i[0] for i in qf_100_app_win[75:150]])
    qf_100_test_y.extend([i[1] for i in qf_100_app_win[75:150]])
    qf_100_test_x.extend([i[0] for i in qf_100_web_ipad[75:150]])
    qf_100_test_y.extend([i[1] for i in qf_100_web_ipad[75:150]])
    qf_100_test_x.extend([i[0] for i in qf_100_app_mac[75:150]])
    qf_100_test_y.extend([i[1] for i in qf_100_app_mac[75:150]])
    qf_100_test_x.extend([i[0] for i in qf_100_iphone[75:150]])
    qf_100_test_y.extend([i[1] for i in qf_100_iphone[75:150]])
    qf_100_test_x.extend([i[0] for i in qf_100_web_win[75:150]])
    qf_100_test_y.extend([i[1] for i in qf_100_web_win[75:150]])
    qf_100_test_x.extend([i[0] for i in qf_100_android[75:150]])
    qf_100_test_y.extend([i[1] for i in qf_100_android[75:150]])

    #clf = svm.SVC(kernel = 'rbf', C = 1, gamma = 0.1)
    #clf = svm.LinearSVC(penalty = 'l2', dual = False, C = 1)
    clf = RandomForestClassifier(n_estimators = 100, min_samples_split = 2, min_samples_leaf = 4, max_features = 'sqrt', max_depth = 10, bootstrap = True)
    clf.fit(qf_100_train_x, qf_100_train_y)
    classifier_predictions = clf.predict(qf_100_test_x)

    #Renaiming classes
    qf_100_test_y_separated_classes = qf_100_test_y.copy()
    qf_100_test_y_separated_classes = renaiming_list_elements(qf_100_test_y_separated_classes, 525, False)
    qf_100_test_y = renaiming_list_elements(qf_100_test_y, 525, True)

    classifier_predictions_separated_classes = classifier_predictions.copy()
    classifier_predictions_separated_classes = renaiming_list_elements(classifier_predictions_separated_classes, 525, False)
    classifier_predictions = renaiming_list_elements(classifier_predictions, 525, True)

    #Confusion matrix with merged classes
    cm = confusion_matrix(qf_100_test_y, classifier_predictions, labels = classes, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)

    fig, ax = plt.subplots(figsize = (11,9))
    plt.rcParams.update({'font.size': 21}) #Update font for the entire confusion matrix
    ax.set_title("QF-100")
    disp.plot(ax = ax, cmap = plt.cm.Blues, colorbar = False)

    plt.setp(ax.get_yticklabels(), fontsize = 18) #Update font for y axis
    plt.setp(ax.get_xticklabels(), fontsize = 18, rotation = 10) #Update font for x axis
    plt.ylabel('True label', fontsize = 18, weight = 'bold') #Update font for "True label"
    plt.xlabel('Predicted label', fontsize = 18, weight = 'bold') #Update font for "Predicted label"
    plt.savefig("Results SVM - RF classification/confusion_matrix_RF_QF-100.pdf")
    plt.show()

    #Confusion matrix with un-merged classes
    cm = confusion_matrix(qf_100_test_y_separated_classes, classifier_predictions_separated_classes, labels = separated_classes, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = separated_classes)

    fig, ax = plt.subplots(figsize = (12,11))
    plt.rcParams.update({'font.size': 21}) #Update font for the entire confusion matrix
    ax.set_title("QF-100")
    disp.plot(ax = ax, cmap = plt.cm.Blues, colorbar = False)

    plt.setp(ax.get_yticklabels(), fontsize = 18) #Update font for y axis
    plt.setp(ax.get_xticklabels(), fontsize = 18, rotation = 15) #Update font for x axis
    plt.ylabel('True label', fontsize = 18, weight = 'bold') #Update font for "True label"
    plt.xlabel('Predicted label', fontsize = 18, weight = 'bold') #Update font for "Predicted label"
    plt.savefig("Results SVM - RF classification/confusion_matrix_RF_sep_class_QF-100.pdf")
    plt.show()

QF_50()
QF_60()
QF_70()
QF_80()
QF_90()
QF_100()