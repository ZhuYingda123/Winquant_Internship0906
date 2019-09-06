import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import scipy.stats as stats
path = '/home/yingda/zyd/preprocessing'
path2 = '/home/yingda/Documents/fft_fig/'

def read_chafen_data(tag):
    file_path = path + '/' + tag + '.txt'
    file = open(file_path,'r')
    chafen_data = np.loadtxt(file,delimiter=',')
    return chafen_data

def smooth_data(chafen_data):
    num_codes, num_ticks = chafen_data.shape
    # set the window size
    win_size = 10
    smooth_chafen_data = np.zeros((num_codes, num_ticks-int((num_ticks+1)/2401)*(win_size-1)))
    for i in range(num_codes):
        for j in range(int((num_ticks+1)/2401)):
            if j!=int((num_ticks+1)/2401)-1 or num_ticks%2401==0:
                for k in range(2401-win_size+1):
                    smooth_chafen_data[i,(2401-win_size)*j+k]=np.mean(chafen_data[i,2401*j+k:2401*j+k+win_size])
            else:
                for k in range(2400-win_size+1):
                    smooth_chafen_data[i,(2401-win_size)*j+k]=np.mean(chafen_data[i,2401*j+k:2401*j+k+win_size])

    return smooth_chafen_data

def fullfill_data(chafen_data):
    num_codes, num_ticks = chafen_data.shape
    for i in range(num_codes):
        num_nan = len(np.nonzero(np.isnan(chafen_data[i,:])==True)[0])
        if num_nan < num_ticks/10:
            meanVal = np.mean(chafen_data[i,np.nonzero(np.isnan(chafen_data[i,:])==False)[0]])
            chafen_data[i,np.nonzero(np.isnan(chafen_data[i,:]))[0]] = meanVal
    return 0

def fft_for_chafen_data(tag, chafen_data, blocksize = 2401):
    num_codes, num_ticks = chafen_data.shape
    fft_data = []
    x = np.arange(int(blocksize/2))
    for i2 in  range(1):
        fft_data1 = np.zeros((int((num_ticks+1)/blocksize),int(blocksize/2)))
        for i in range(int((num_ticks+1)/blocksize)):
            data_in_two_hours = chafen_data[i2,i*blocksize:(i+1)*blocksize]-np.mean(chafen_data[0,i*blocksize:(i+1)*blocksize])
            # print(data_in_two_hours)
            fft_data_in_two_hours = fft(data_in_two_hours)
            # print(fft_data_in_two_hours)
            real = fft_data_in_two_hours.real
            imag = fft_data_in_two_hours.imag
            abs_fft_data_in_two_hours=abs(fft_data_in_two_hours)
            fft_y_norm = abs_fft_data_in_two_hours/(blocksize-1)
            fft_data1[i,:]=fft_y_norm[0:int(blocksize/2)]
            # print(fft_data1)
            ave = np.mean(fft_y_norm)
            max_index = np.argmax(fft_y_norm[0:int(blocksize/2)])
            print(max_index)
            fft_data_in_two_hours[np.nonzero(fft_y_norm[:]<1.2*ave)[0]]=0
            # print(fft_data_in_two_hours)
            y_extra = ifft(fft_data_in_two_hours).real
            plt.figure(1)
            plt.plot(x,fft_y_norm[0:int(blocksize/2)],'r')
            plt.title('fft'+tag,fontsize=9,color='b')
            plt.savefig(path2+tag+'_'+'chafen'+str(i))
            plt.close(1)
        fft_data.append(fft_data1)
    return fft_data

def cal_corr(ticks_tag1, ticks_tag2):
    pr = stats.pearsonr(ticks_tag1, ticks_tag2)[0]
    spr = stats.spearmanr(ticks_tag1, ticks_tag2)[0]
    return pr, spr

if __name__ == '__main__':
    tag='last'
    tag2='ask_vol1'
    # a= np.arange(100)
    # b= 2*np.arange(100)
    # print(cal_corr(a, b))
    chafen_data = read_chafen_data(tag)
    chafen_data2 = read_chafen_data(tag2)
    fullfill_data(chafen_data)
    fullfill_data(chafen_data2)
    len_min = min(chafen_data.shape[1],chafen_data2.shape[1])
    print(cal_corr(chafen_data[0,:len_min-1],chafen_data2[0,:len_min-1]))
    # smooth_chafen_data = smooth_data(chafen_data)
    fft_for_chafen_data(tag, chafen_data, 2401)
