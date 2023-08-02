from sklearn import manifold
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

#Getting features
methods = ['WHATSAPP_WEB_MAC', 'WHATSAPP_APP_WIN', 'WHATSAPP_WEB_IPAD', 'WHATSAPP_APP_MAC', 'WHATSAPP_IPHONE', 'WHATSAPP_WEB_WIN']

classes = ['APP-DESKTOP', 'IPHONE', 'WEB-SAFARI', 'WEB-WIN']
separated_classes = ['APP-MAC', 'APP-WIN' ,'IPHONE', 'WEB-IPAD', 'WEB-MAC' ,'WEB-WIN']

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
            features.append(feature)

features = np.asarray(features)

#Dimensionality reduction
iso = manifold.Isomap(n_components = 2, n_neighbors = 20, eigen_solver = 'dense')
iso.fit(features)
feature_transformed = iso.transform(features)

qf_80 = []
qf_80.extend(features[0:150])
qf_80.extend(features[900:1050])
qf_80.extend(features[1800:1950])
qf_80.extend(features[2700:2850])
qf_80.extend(features[3600:3750])
qf_80.extend(features[4500:4650])
qf_80_transformed = iso.transform(qf_80)

qf_50 = []
qf_50.extend(features[150:300])
qf_50.extend(features[1050:1200])
qf_50.extend(features[1950:2100])
qf_50.extend(features[2850:3000])
qf_50.extend(features[3750:3900])
qf_50.extend(features[4650:4800])
qf_50_transformed = iso.transform(qf_50)

qf_60 = []
qf_60.extend(features[300:450])
qf_60.extend(features[1200:1350])
qf_60.extend(features[2100:2250])
qf_60.extend(features[3000:3150])
qf_60.extend(features[3900:4050])
qf_60.extend(features[4800:4950])
qf_60_transformed = iso.transform(qf_60)

qf_70 = []
qf_70.extend(features[450:600])
qf_70.extend(features[1350:1500])
qf_70.extend(features[2250:2400])
qf_70.extend(features[3150:3300])
qf_70.extend(features[4050:4200])
qf_70.extend(features[4950:5100])
qf_70_transformed = iso.transform(qf_70)

qf_100 = []
qf_100.extend(features[600:750])
qf_100.extend(features[1500:1650])
qf_100.extend(features[2400:2550])
qf_100.extend(features[3300:3450])
qf_100.extend(features[4200:4350])
qf_100.extend(features[5100:5250])
qf_100_transformed = iso.transform(qf_100)

qf_90 = []
qf_90.extend(features[750:900])
qf_90.extend(features[1650:1800])
qf_90.extend(features[2550:2700])
qf_90.extend(features[3450:3600])
qf_90.extend(features[4350:4500])
qf_90.extend(features[5250:5400])
qf_90_transformed = iso.transform(qf_90)

#Merge classes
app = feature_transformed[900:1800]
np.concatenate((app, feature_transformed[2700:3600]), axis = 0)

web_safari = feature_transformed[0:900]
np.concatenate((web_safari, feature_transformed[1800:2700]), axis = 0)

#Un-merged classes
web_mac = feature_transformed[0:900]
app_win = feature_transformed[900:1800]
web_ipad = feature_transformed[1800:2700]
app_mac = feature_transformed[2700:3600]
iphone = feature_transformed[3600:4500]
web_win = feature_transformed[4500:5400]



#Plotting reduced vectors
#Isomap of merged classes
plt.rcParams.update({'font.size': 13}) #Update font size for all the isomap
fig = plt.figure()
ax = fig.add_subplot()
ax.tick_params(axis = 'y', labelsize = 13) #Update font for x axis
ax.tick_params(axis = 'x', labelsize = 13) #Update font for y axis
ax.set_xlabel('X')
ax.set_ylabel('Y') 
ax.scatter(app[:, 0], app[:, 1], marker = '.', alpha = 0.7, c = 'blue', label = classes[0], s = 80)
ax.scatter(iphone[:, 0], iphone[:, 1], marker = '.', alpha = 0.7, c = 'green', label = classes[1], s = 80)
ax.scatter(web_safari[:, 0], web_safari[:, 1], marker = '.', alpha = 0.7, c = 'magenta', label = classes[2], s = 80)
ax.scatter(web_win[:, 0], web_win[:, 1], marker = '.', alpha = 0.7, c = 'cyan', label = classes[3], s = 80)
ax.legend()
f = plt.gcf()
f.set_size_inches(8, 6)
plt.xlim([-2, 5])
plt.ylim([-1.5, 2])
plt.savefig("Results dimensionality reduction isomap/isomap.pdf", format = "pdf", dpi = 500)
plt.show()

#Isomap of un-merged classes
fig = plt.figure()
ax = fig.add_subplot()
plt.rcParams.update({'font.size': 13}) #Update font size for all the isomap
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.scatter(web_mac[:, 0], web_mac[:, 1], marker = '.', alpha = 0.7, c = 'magenta', label = separated_classes[4], s = 80)
ax.scatter(app_win[:, 0], app_win[:, 1], marker = '.', alpha = 0.7, c = 'blue', label = separated_classes[1], s = 80)
ax.scatter(web_ipad[:, 0], web_ipad[:, 1], marker = '.', alpha = 0.7, c = 'red', label = separated_classes[3], s = 80)
ax.scatter(app_mac[:, 0], app_mac[:, 1], marker = '.', alpha = 0.7, c = 'black', label = separated_classes[0], s = 80)
ax.scatter(iphone[:, 0], iphone[:, 1], marker = '.', alpha = 0.7, c = 'green', label = separated_classes[2], s = 80)
ax.scatter(web_win[:, 0], web_win[:, 1], marker = '.', alpha = 0.7, c = 'cyan', label = separated_classes[5], s = 80)
ax.legend()
f = plt.gcf()
f.set_size_inches(8, 6)
plt.xlim([-2, 5])
plt.ylim([-1.5, 2])
plt.savefig("Results dimensionality reduction isomap/isomap_sep_class.pdf", format = "pdf", dpi = 500)
plt.show()

#Isomaps of merged classes for all quality factors
fig = plt.figure()
ax1 = fig.add_subplot()
plt.rcParams.update({'font.size': 13}) #Update font size for all the isomap
ax1.set_title('QF-50')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.scatter(np.concatenate((qf_50_transformed[150:300, 0], qf_50_transformed[450:600, 0]), axis = 0), np.concatenate((qf_50_transformed[150:300, 1], qf_50_transformed[450:600, 1]), axis = 0), marker = '.', alpha = 0.7, c = 'blue', label = classes[0], s = 80)
ax1.scatter(qf_50_transformed[600:750, 0], qf_50_transformed[600:750, 1], marker = '.', alpha = 0.7, c = 'green', label = classes[1], s = 80)
ax1.scatter(np.concatenate((qf_50_transformed[0:150, 0], qf_50_transformed[300:450, 0]), axis = 0), np.concatenate((qf_50_transformed[0:150, 1], qf_50_transformed[300:450, 1]), axis = 0), marker = '.', alpha = 0.7, c = 'magenta', label = classes[2], s = 80)
ax1.scatter(qf_50_transformed[750:900, 0], qf_50_transformed[750:900, 1], marker = '.', alpha = 0.7, c = 'cyan', label = classes[3], s = 80)
ax1.legend()
f = plt.gcf()
f.set_size_inches(8, 6)
plt.xlim([-2, 5])
plt.ylim([-1.5, 2])
plt.savefig("Results dimensionality reduction isomap/isomap_QF-50.pdf", format = "pdf", dpi = 500)

fig = plt.figure()
ax2 = fig.add_subplot()
plt.rcParams.update({'font.size': 13}) #Update font size for all the isomap
ax2.set_title('QF-60')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.scatter(np.concatenate((qf_60_transformed[150:300, 0], qf_60_transformed[450:600, 0]), axis = 0), np.concatenate((qf_60_transformed[150:300, 1], qf_60_transformed[450:600, 1]), axis = 0), marker = '.', alpha = 0.7, c = 'blue', label = classes[0], s = 80)
ax2.scatter(qf_60_transformed[600:750, 0], qf_60_transformed[600:750, 1], marker = '.', alpha = 0.7, c = 'green', label = classes[1], s = 80)
ax2.scatter(np.concatenate((qf_60_transformed[0:150, 0], qf_60_transformed[300:450, 0]), axis = 0), np.concatenate((qf_60_transformed[0:150, 1], qf_60_transformed[300:450, 1]), axis = 0), marker = '.', alpha = 0.7, c = 'magenta', label = classes[2], s = 80)
ax2.scatter(qf_60_transformed[750:900, 0], qf_60_transformed[750:900, 1], marker = '.', alpha = 0.7, c = 'cyan', label = classes[3], s = 80)
ax2.legend()
f = plt.gcf()
f.set_size_inches(8, 6)
plt.xlim([-2, 5])
plt.ylim([-1.5, 2])
plt.savefig("Results dimensionality reduction isomap/isomap_QF-60.pdf", format = "pdf", dpi = 500)

fig = plt.figure()
ax3 = fig.add_subplot()
plt.rcParams.update({'font.size': 13}) #Update font size for all the isomap
ax3.set_title('QF-70')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.scatter(np.concatenate((qf_70_transformed[150:300, 0], qf_70_transformed[450:600, 0]), axis = 0), np.concatenate((qf_70_transformed[150:300, 1], qf_70_transformed[450:600, 1]), axis = 0), marker = '.', alpha = 0.7, c = 'blue', label = classes[0], s = 80)
ax3.scatter(qf_70_transformed[600:750, 0], qf_70_transformed[600:750, 1], marker = '.', alpha = 0.7, c = 'green', label = classes[1], s = 80)
ax3.scatter(np.concatenate((qf_70_transformed[0:150, 0], qf_70_transformed[300:450, 0]), axis = 0), np.concatenate((qf_70_transformed[0:150, 1], qf_70_transformed[300:450, 1]), axis = 0), marker = '.', alpha = 0.7, c = 'magenta', label = classes[2], s = 80)
ax3.scatter(qf_70_transformed[750:900, 0], qf_70_transformed[750:900, 1], marker = '.', alpha = 0.7, c = 'cyan', label = classes[3], s = 80)
ax3.legend()
f = plt.gcf()
f.set_size_inches(8, 6)
plt.xlim([-2, 5])
plt.ylim([-1.5, 2])
plt.savefig("Results dimensionality reduction isomap/isomap_QF-70.pdf", format = "pdf", dpi = 500)

fig = plt.figure()
ax4 = fig.add_subplot()
plt.rcParams.update({'font.size': 13}) #Update font size for all the isomap
ax4.set_title('QF-80')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.scatter(np.concatenate((qf_80_transformed[150:300, 0], qf_80_transformed[450:600, 0]), axis = 0), np.concatenate((qf_80_transformed[150:300, 1], qf_80_transformed[450:600, 1]), axis = 0), marker = '.', alpha = 0.7, c = 'blue', label = classes[0], s = 80)
ax4.scatter(qf_80_transformed[600:750, 0], qf_80_transformed[600:750, 1], marker = '.', alpha = 0.7, c = 'green', label = classes[1], s = 80)
ax4.scatter(np.concatenate((qf_80_transformed[0:150, 0], qf_80_transformed[300:450, 0]), axis = 0), np.concatenate((qf_80_transformed[0:150, 1], qf_80_transformed[300:450, 1]), axis = 0), marker = '.', alpha = 0.7, c = 'magenta', label = classes[2], s = 80)
ax4.scatter(qf_80_transformed[750:900, 0], qf_80_transformed[750:900, 1], marker = '.', alpha = 0.7, c = 'cyan', label = classes[3], s = 80)
ax4.legend()
f = plt.gcf()
f.set_size_inches(8, 6)
plt.xlim([-2, 5])
plt.ylim([-1.5, 2])
plt.savefig("Results dimensionality reduction isomap/isomap_QF-80.pdf", format = "pdf", dpi = 500)

fig = plt.figure()
ax5 = fig.add_subplot()
plt.rcParams.update({'font.size': 13}) #Update font size for all the isomap
ax5.set_title('QF-90')
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.scatter(np.concatenate((qf_90_transformed[150:300, 0], qf_90_transformed[450:600, 0]), axis = 0), np.concatenate((qf_90_transformed[150:300, 1], qf_90_transformed[450:600, 1]), axis = 0), marker = '.', alpha = 0.7, c = 'blue', label = classes[0], s = 80)
ax5.scatter(qf_90_transformed[600:750, 0], qf_90_transformed[600:750, 1], marker = '.', alpha = 0.7, c = 'green', label = classes[1], s = 80)
ax5.scatter(np.concatenate((qf_90_transformed[0:150, 0], qf_90_transformed[300:450, 0]), axis = 0), np.concatenate((qf_90_transformed[0:150, 1], qf_90_transformed[300:450, 1]), axis = 0), marker = '.', alpha = 0.7, c = 'magenta', label = classes[2], s = 80)
ax5.scatter(qf_90_transformed[750:900, 0], qf_90_transformed[750:900, 1], marker = '.', alpha = 0.7, c = 'cyan', label = classes[3], s = 80)
ax5.legend()
f = plt.gcf()
f.set_size_inches(8, 6)
plt.xlim([-2, 5])
plt.ylim([-1.5, 2])
plt.savefig("Results dimensionality reduction isomap/isomap_QF-90.pdf", format = "pdf", dpi = 500)

fig = plt.figure()
ax6 = fig.add_subplot()
plt.rcParams.update({'font.size': 13}) #Update font size for all the isomap
ax6.set_title('QF-100')
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.scatter(np.concatenate((qf_100_transformed[150:300, 0], qf_100_transformed[450:600, 0]), axis = 0), np.concatenate((qf_100_transformed[150:300, 1], qf_100_transformed[450:600, 1]), axis = 0), marker = '.', alpha = 0.7, c = 'blue', label = classes[0], s = 80)
ax6.scatter(qf_100_transformed[600:750, 0], qf_100_transformed[600:750, 1], marker = '.', alpha = 0.7, c = 'green', label = classes[1], s = 80)
ax6.scatter(np.concatenate((qf_100_transformed[0:150, 0], qf_100_transformed[300:450, 0]), axis = 0), np.concatenate((qf_100_transformed[0:150, 1], qf_100_transformed[300:450, 1]), axis = 0), marker = '.', alpha = 0.7, c = 'magenta', label = classes[2], s = 80)
ax6.scatter(qf_100_transformed[750:900, 0], qf_100_transformed[750:900, 1], marker = '.', alpha = 0.7, c = 'cyan', label = classes[3], s = 80)
ax6.legend()
f = plt.gcf()
f.set_size_inches(8, 6)
plt.xlim([-2, 5])
plt.ylim([-1.5, 2])
plt.savefig("Results dimensionality reduction isomap/isomap_QF-100.pdf", format = "pdf", dpi = 500)

#0:150 -> WEB-MAC
#150:300 -> APP-WIN
#300:450 -> WEB-IPAD
#450:600 -> APP-MAC
#Isomaps of un-merged classes for all quality factors
fig = plt.figure()
ax7 = fig.add_subplot()
plt.rcParams.update({'font.size': 13}) #Update font size for all the isomap
ax7.set_title('QF-50')
ax7.set_xlabel('X')
ax7.set_ylabel('Y')
ax7.scatter(qf_50_transformed[0:150, 0], qf_50_transformed[0:150, 1], marker = '.', alpha = 0.7, c = 'magenta', label = separated_classes[4], s = 80)
ax7.scatter(qf_50_transformed[150:300, 0], qf_50_transformed[150:300, 1], marker = '.', alpha = 0.7, c = 'blue', label = separated_classes[1], s = 80)
ax7.scatter(qf_50_transformed[300:450, 0], qf_50_transformed[300:450, 1], marker = '.', alpha = 0.7, c = 'red', label = separated_classes[3], s = 80)
ax7.scatter(qf_50_transformed[450:600, 0], qf_50_transformed[450:600, 1], marker = '.', alpha = 0.7, c = 'black', label = separated_classes[0], s = 80)
ax7.scatter(qf_50_transformed[600:750, 0], qf_50_transformed[600:750, 1], marker = '.', alpha = 0.7, c = 'green', label = separated_classes[2], s = 80)
ax7.scatter(qf_50_transformed[750:900, 0], qf_50_transformed[750:900, 1], marker = '.', alpha = 0.7, c = 'cyan', label = separated_classes[5], s = 80)
ax7.legend()
f = plt.gcf()
f.set_size_inches(8, 6)
plt.xlim([-2, 5])
plt.ylim([-1.5, 2])
plt.savefig("Results dimensionality reduction isomap/isomap_sep_class_QF-50.pdf", format = "pdf", dpi = 500)

fig = plt.figure()
ax8 = fig.add_subplot()
plt.rcParams.update({'font.size': 13}) #Update font size for all the isomap
ax8.set_title('QF-60')
ax8.set_xlabel('X')
ax8.set_ylabel('Y')
ax8.scatter(qf_60_transformed[0:150, 0], qf_60_transformed[0:150, 1], marker = '.', alpha = 0.7, c = 'magenta', label = separated_classes[4], s = 80)
ax8.scatter(qf_60_transformed[150:300, 0], qf_60_transformed[150:300, 1], marker = '.', alpha = 0.7, c = 'blue', label = separated_classes[1], s = 80)
ax8.scatter(qf_60_transformed[300:450, 0], qf_60_transformed[300:450, 1], marker = '.', alpha = 0.7, c = 'red', label = separated_classes[3], s = 80)
ax8.scatter(qf_60_transformed[450:600, 0], qf_60_transformed[450:600, 1], marker = '.', alpha = 0.7, c = 'black', label = separated_classes[0], s = 80)
ax8.scatter(qf_60_transformed[600:750, 0], qf_60_transformed[600:750, 1], marker = '.', alpha = 0.7, c = 'green', label = separated_classes[2], s = 80)
ax8.scatter(qf_60_transformed[750:900, 0], qf_60_transformed[750:900, 1], marker = '.', alpha = 0.7, c = 'cyan', label = separated_classes[5], s = 80)
ax8.legend()
f = plt.gcf()
f.set_size_inches(8, 6)
plt.xlim([-2, 5])
plt.ylim([-1.5, 2])
plt.savefig("Results dimensionality reduction isomap/isomap_sep_class_QF-60.pdf", format = "pdf", dpi = 500)

fig = plt.figure()
ax9 = fig.add_subplot()
plt.rcParams.update({'font.size': 13}) #Update font size for all the isomap
ax9.set_title('QF-70')
ax9.set_xlabel('X')
ax9.set_ylabel('Y')
ax9.scatter(qf_70_transformed[0:150, 0], qf_70_transformed[0:150, 1], marker = '.', alpha = 0.7, c = 'magenta', label = separated_classes[4], s = 80)
ax9.scatter(qf_70_transformed[150:300, 0], qf_70_transformed[150:300, 1], marker = '.', alpha = 0.7, c = 'blue', label = separated_classes[1], s = 80)
ax9.scatter(qf_70_transformed[300:450, 0], qf_70_transformed[300:450, 1], marker = '.', alpha = 0.7, c = 'red', label = separated_classes[3], s = 80)
ax9.scatter(qf_70_transformed[450:600, 0], qf_70_transformed[450:600, 1], marker = '.', alpha = 0.7, c = 'black', label = separated_classes[0], s = 80)
ax9.scatter(qf_70_transformed[600:750, 0], qf_70_transformed[600:750, 1], marker = '.', alpha = 0.7, c = 'green', label = separated_classes[2], s = 80)
ax9.scatter(qf_70_transformed[750:900, 0], qf_70_transformed[750:900, 1], marker = '.', alpha = 0.7, c = 'cyan', label = separated_classes[5], s = 80)
ax9.legend()
f = plt.gcf()
f.set_size_inches(8, 6)
plt.xlim([-2, 5])
plt.ylim([-1.5, 2])
plt.savefig("Results dimensionality reduction isomap/isomap_sep_class_QF-70.pdf", format = "pdf", dpi = 500)

fig = plt.figure()
ax10 = fig.add_subplot()
plt.rcParams.update({'font.size': 13}) #Update font size for all the isomap
ax10.set_title('QF-80')
ax10.set_xlabel('X')
ax10.set_ylabel('Y')
ax10.scatter(qf_80_transformed[0:150, 0], qf_80_transformed[0:150, 1], marker = '.', alpha = 0.7, c = 'magenta', label = separated_classes[4], s = 80)
ax10.scatter(qf_80_transformed[150:300, 0], qf_80_transformed[150:300, 1], marker = '.', alpha = 0.7, c = 'blue', label = separated_classes[1], s = 80)
ax10.scatter(qf_80_transformed[300:450, 0], qf_80_transformed[300:450, 1], marker = '.', alpha = 0.7, c = 'red', label = separated_classes[3], s = 80)
ax10.scatter(qf_80_transformed[450:600, 0], qf_80_transformed[450:600, 1], marker = '.', alpha = 0.7, c = 'black', label = separated_classes[0], s = 80)
ax10.scatter(qf_80_transformed[600:750, 0], qf_80_transformed[600:750, 1], marker = '.', alpha = 0.7, c = 'green', label = separated_classes[2], s = 80)
ax10.scatter(qf_80_transformed[750:900, 0], qf_80_transformed[750:900, 1], marker = '.', alpha = 0.7, c = 'cyan', label = separated_classes[5], s = 80)
ax10.legend()
f = plt.gcf()
f.set_size_inches(8, 6)
plt.xlim([-2, 5])
plt.ylim([-1.5, 2])
plt.savefig("Results dimensionality reduction isomap/isomap_sep_class_QF-80.pdf", format = "pdf", dpi = 500)

fig = plt.figure()
ax11 = fig.add_subplot()
plt.rcParams.update({'font.size': 13}) #Update font size for all the isomap
ax11.set_title('QF-90')
ax11.set_xlabel('X')
ax11.set_ylabel('Y')
ax11.scatter(qf_90_transformed[0:150, 0], qf_90_transformed[0:150, 1], marker = '.', alpha = 0.7, c = 'magenta', label = separated_classes[4], s = 80)
ax11.scatter(qf_90_transformed[150:300, 0], qf_90_transformed[150:300, 1], marker = '.', alpha = 0.7, c = 'blue', label = separated_classes[1], s = 80)
ax11.scatter(qf_90_transformed[300:450, 0], qf_90_transformed[300:450, 1], marker = '.', alpha = 0.7, c = 'red', label = separated_classes[3], s = 80)
ax11.scatter(qf_90_transformed[450:600, 0], qf_90_transformed[450:600, 1], marker = '.', alpha = 0.7, c = 'black', label = separated_classes[0], s = 80)
ax11.scatter(qf_90_transformed[600:750, 0], qf_90_transformed[600:750, 1], marker = '.', alpha = 0.7, c = 'green', label = separated_classes[2], s = 80)
ax11.scatter(qf_90_transformed[750:900, 0], qf_90_transformed[750:900, 1], marker = '.', alpha = 0.7, c = 'cyan', label = separated_classes[5], s = 80)
ax11.legend()
f = plt.gcf()
f.set_size_inches(8, 6)
plt.xlim([-2, 5])
plt.ylim([-1.5, 2])
plt.savefig("Results dimensionality reduction isomap/isomap_sep_class_QF-90.pdf", format = "pdf", dpi = 500)

fig = plt.figure()
ax12 = fig.add_subplot()
plt.rcParams.update({'font.size': 13}) #Update font size for all the isomap
ax12.set_title('QF-100')
ax12.set_xlabel('X')
ax12.set_ylabel('Y')
ax12.scatter(qf_100_transformed[0:150, 0], qf_100_transformed[0:150, 1], marker = '.', alpha = 0.7, c = 'magenta', label = separated_classes[4], s = 80)
ax12.scatter(qf_100_transformed[150:300, 0], qf_100_transformed[150:300, 1], marker = '.', alpha = 0.7, c = 'blue', label = separated_classes[1], s = 80)
ax12.scatter(qf_100_transformed[300:450, 0], qf_100_transformed[300:450, 1], marker = '.', alpha = 0.7, c = 'red', label = separated_classes[3], s = 80)
ax12.scatter(qf_100_transformed[450:600, 0], qf_100_transformed[450:600, 1], marker = '.', alpha = 0.7, c = 'black', label = separated_classes[0], s = 80)
ax12.scatter(qf_100_transformed[600:750, 0], qf_100_transformed[600:750, 1], marker = '.', alpha = 0.7, c = 'green', label = separated_classes[2], s = 80)
ax12.scatter(qf_100_transformed[750:900, 0], qf_100_transformed[750:900, 1], marker = '.', alpha = 0.7, c = 'cyan', label = separated_classes[5], s = 80)
ax12.legend()
f = plt.gcf()
f.set_size_inches(8, 6)
plt.xlim([-2, 5])
plt.ylim([-1.5, 2])
plt.savefig("Results dimensionality reduction isomap/isomap_sep_class_QF-100.pdf", format = "pdf", dpi = 500)