from math import isnan, log10, sqrt
import matplotlib.pyplot as plt
import numpy as np
import os

#Resolutions
res_1 = '337x600'
res_2 = '1012x1800'
res_3 = '1687x3000'

methods = ['WHATSAPP_WEB_MAC', 'WHATSAPP_APP_WIN', 'WHATSAPP_WEB_IPAD', 'WHATSAPP_APP_MAC', 'WHATSAPP_WEB_WIN']

web_mac = []
app_win = []
web_ipad = []
app_mac = []
web_win = []

def MSE(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return mse

def PSNR(mse):
    max_pixel = 255.0

    if mse != 0:
        p = 20 * log10(max_pixel / sqrt(mse))
        return p
    else:
        return np.NaN

def load_images(elem, path):
    arr = []

    for i in range(150):
        image = plt.imread('Dataset/' + path + '/QF-100/' + elem[i], 1)
        arr.append(image)

    return arr

def compute_psnr(met1, met2):
    arr = []

    for i in range(150):
        m = MSE(met1[i], met2[i])
        p = PSNR(m)
        arr.insert(i, p)

    return arr

def plot(name, arr):

    arr = [x for x in arr if str(x) != 'nan']

    plt.boxplot(arr)
    plt.xticks([1, 2, 3, 4], ["APP MAC - APP WIN", "APP MAC - WEB IPAD", "APP MAC - WEB MAC", "APP MAC - WEB WIN"], size = 8)
    plt.ylim([32, 62])
    plt.savefig('Results PSNR/' + name + '.png', dpi = 500)
    plt.close()

def main():
    for method in methods:
        if method == 'WHATSAPP_WEB_MAC':
            web_mac = load_images(os.listdir('Dataset/' + method + '/QF-100'), 'WHATSAPP_WEB_MAC')
        elif method == 'WHATSAPP_APP_WIN':
            app_win = load_images(os.listdir('Dataset/' + method + '/QF-100'), 'WHATSAPP_APP_WIN')
        elif method == 'WHATSAPP_WEB_IPAD':
            web_ipad = load_images(os.listdir('Dataset/' + method + '/QF-100'), 'WHATSAPP_WEB_IPAD')
        elif method == 'WHATSAPP_APP_MAC':
            app_mac = load_images(os.listdir('Dataset/' + method + '/QF-100'), 'WHATSAPP_APP_MAC')
        elif method == 'WHATSAPP_WEB_WIN':
            web_win = load_images(os.listdir('Dataset/' + method + '/QF-100'), 'WHATSAPP_WEB_WIN')

    app_mac_app_win = compute_psnr(app_mac, app_win)
    app_mac_web_ipad = compute_psnr(app_mac, web_ipad)
    app_mac_web_mac = compute_psnr(app_mac, web_mac)
    app_mac_web_win = compute_psnr(app_mac, web_win)
    app_win_web_win = compute_psnr(app_win, web_win)
    web_ipad_app_win = compute_psnr(web_ipad, app_win)
    web_ipad_web_win = compute_psnr(web_ipad, web_win)
    web_mac_app_win = compute_psnr(web_mac, app_win)
    web_mac_web_ipad = compute_psnr(web_mac, web_ipad)
    web_mac_web_win = compute_psnr(web_mac, web_win)

    app_vs_web_safari = []
    app_vs_web_safari.extend(app_mac_web_ipad)
    app_vs_web_safari.extend(app_mac_web_mac)
    app_vs_web_safari.extend(web_ipad_app_win)
    app_vs_web_safari.extend(web_mac_app_win)

    web_safari_vs_web_win = []
    web_safari_vs_web_win.extend(web_ipad_web_win)
    web_safari_vs_web_win.extend(web_mac_web_win)

    app_vs_web_win = []
    app_vs_web_win.extend(app_mac_web_win)
    app_vs_web_win.extend(app_win_web_win)

    app_mac_app_win = [x for x in app_mac_app_win if str(x) != 'nan']
    app_mac_web_ipad = [x for x in app_mac_web_ipad if str(x) != 'nan']
    app_mac_web_mac = [x for x in app_mac_web_mac if str(x) != 'nan']
    app_mac_web_win = [x for x in app_mac_web_win if str(x) != 'nan']

    #plot('APP DESKTOP - WEB SAFARI', app_vs_web_safari)
    #plot('WEB SAFARI - WEB WIN', web_safari_vs_web_win)
    #plot('APP DESKTOP - WEB WIN', app_vs_web_win)
    plot("Immagine1", [app_mac_app_win, app_mac_web_ipad, app_mac_web_mac, app_mac_web_win])

if __name__ == '__main__':
    main()