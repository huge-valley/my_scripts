import os
import numpy as np                    
import pandas as pd                   #csv読み込みに必要
import matplotlib.pylab as plt        #plot用
import smplotlib                      #plotをいい感じにするためのライブラリ
import iminuit                        #chi squareフィッティングを行うためのライブラリ
import sys                            #コマンドライン引数を読み取るライブラリ
from matplotlib import gridspec       #残差を下につけたりできるライブラリ


#コマンドラインの第一引数からファイルネームを取得してfilenameに格納
filename=sys.argv[1]

#フォントの設定TeX風(グラフ描画前におく)
plt.rcParams['text.usetex'] = True

#データをデータフレームに詰める
df = pd.read_csv(filename,index_col=False,
  names=['minute','sec','1/10000 sec','channel'],
  dtype={'minute':np.uint16,'sec':np.uint16,'1/10000 sec':np.uint16,'channel':np.uint16})
df['total_sec'] = df['minute']*60+df['sec']
print(df)


##########################関数を定義しておく##########################
num_of_binning = 20


#histgramを生成
def const_hist(x_ax,num_of_binning):
    time_total_file = max(df['total_sec'])+1      #time_total_fileは時間の単位に合わせて変更する    
    #ヒストグラムにデータdfを詰めていく
    # if  98.23< df['channel'] <　#ここにif文入れればエネルギーで切れる？
    hist_y, edges = np.histogram(x_ax,
                                bins=int(time_total_file/num_of_binning),
                                range=(-0.5,time_total_file-0.5))
    hist_yerr = np.sqrt(hist_y) #Poisson分布を仮定したときの統計誤差
    hist_x = (edges[:-1] + edges[1:]) / 2.
    hist_xerr = (-edges[:-1] + edges[1:]) / 2.
    return hist_x,hist_y,hist_xerr,hist_yerr



#lightcurveの描画
def plot_lightcurve(lc_x,lc_y,lc_xe,lc_ye):
    #histogramの描画
    fig, ax = plt.subplots(1,1,figsize=(7.00,4.33)) #台紙と付箋が作られる．subplots(行，列，sharex)sharex="row"で同じ列のx軸を共有．#グラフの縦横比はここで設定

    #plotの体裁を整える
    ax.ticklabel_format(axis='x', style='plain', useOffset=False) 
    ax.minorticks_on()                                            #補助目盛をつける
    ax.tick_params(axis="both", which='major', direction='in', length=5) #主軸を内向き(in)にして長さ5
    ax.tick_params(axis="both", which='minor', direction='in', length=3) #補助目盛を内向きにして長さ3
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.xlabel('time(sec)', fontsize=15)              #x軸ラベル
    plt.ylabel('Counts/sec', fontsize=15)           #y軸ラベル
    plt.title(filename.replace('.csv',''),fontsize=15)       #グラフタイトルはファイル名から.csvを削除したもの

    # plt.yscale('log')
    # plt.xlim([0,40])                             #x軸範囲
    # plt.ylim([240,300])                           #y軸範囲

    plt.errorbar(lc_x,lc_y/num_of_binning,xerr=lc_xe,yerr=lc_ye/num_of_binning,marker='',drawstyle='steps-mid',capthick=0) #線でヒストグラムを表示する
    # plt.errorbar(lc_x,lc_y,xerr=lc_xe,yerr=lc_ye,marker='',drawstyle='steps-mid',capthick=0) #線でヒストグラムを表示する
    # plt.plot(xt, yt, "r--",label="guess")
    plt.savefig(filename.replace('.csv','.png'))
    plt.show()



#gaussianを定義する
def mygauss(x,area,mu,sigma):
    return area/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5 * ((x-mu)**2/(sigma**2)))
##########################関数を定義しておく##########################


lc_x,lc_y,lc_xe,lc_ye=const_hist(df['total_sec'],num_of_binning)          #時間の単位に合わせて変更する
plot_lightcurve(lc_x,lc_y,lc_xe,lc_ye)

# ####################################手動でGaussianを合わせる########################################
# xt = np.linspace(0, NUMBER_OF_BIT_PHA, NUMBER_OF_BIT_PHA)
# yt = mygauss(xt,area=3e+5,mu=60,sigma=5)
# plt.errorbar(hist_x,hist_y,yerr=hist_yerr,marker='',drawstyle='steps-mid',color='k',capthick=0)
# plt.plot(xt, yt, "r--",label="guess")
# ############################################ここまで###############################################



# ###########################################ここからフィッティング###########################################
# #80<x<120付近のピークをフィットするためにフラグを立てておく
# flag = np.logical_and(hist_x > 55, hist_x<65) 
# #最小化する関数を定義する
# def mycostfunc(area,mu,sigma):
#     return sum((y-mygauss(x,area,mu,sigma))**2/y for x,y in zip(hist_x[flag],hist_y[flag]))

# myminuit = iminuit.Minuit(mycostfunc,area=3e+5,mu=60,sigma=5)
# myminuit.limits["area"] = (0,1e+6)
# myminuit.limits["mu","sigma"] = (0.0,None) 
# myminuit.migrad()

# area_bestfit=myminuit.values["area"]
# area_error=myminuit.errors["area"]
# mu_bestfit=myminuit.values["mu"]
# mu_error=myminuit.errors["mu"]
# sigma_bestfit=myminuit.values["sigma"]
# sigma_error=myminuit.errors["sigma"]
# #gaussianを再定義
# xt = np.linspace(0, NUMBER_OF_BIT_PHA, NUMBER_OF_BIT_PHA)
# yt = mygauss(xt,area=area_bestfit,mu=mu_bestfit,sigma=sigma_bestfit)
# #フィッティングの結果を出力
# print("area=",area_bestfit,"+-",area_error)
# print("mu=",mu_bestfit,"+-",mu_error)
# print("sigma=",sigma_bestfit,"+-",sigma_error)
# ###########################################ここまでフィッティング###########################################



# ###########################################ここから残差付きグラフの描画############################################
# fig = plt.figure(figsize=(8, 6)) #台紙を用意
# gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)  # hspace=0 で縦の隙間をなくす

# ax1 = fig.add_subplot(gs[0])                #gs[0]=メインのax
# ax1.minorticks_on()
# ax1.set_title('Cogamo ${}^{137}\mathrm{Cs}$ 0512',fontsize=15)
# ax1.hist(df["channel"], bins=1024, range=(0, 1024), color="c")
# ax1.plot(xt, yt, "r--", label="fit")
# ax1.set_ylabel("Counts/bin")
# ax1.tick_params(labelbottom=False)  # x軸のラベルは下側 (ax2) だけに表示
# ax1.set_xlim(55,65)
# # ax1.set_yscale('log')

# ax2 = fig.add_subplot(gs[1], sharex=ax1)
# ax2.minorticks_on()
# ax2.set_xlabel("Channel")
# ax2.set_ylabel("Residual")
# # 残差の描画（例）
# residual = (hist_y - mygauss(hist_x, area_bestfit, mu_bestfit, sigma_bestfit))/hist_yerr
# ax2.errorbar(hist_x, residual, yerr=hist_yerr, fmt="o", markersize=2, color="k")


# fig.savefig("cogamo_spec_0512_residual.pdf", facecolor=fig.get_facecolor())
# ###########################################ここまでグラフの描画############################################

