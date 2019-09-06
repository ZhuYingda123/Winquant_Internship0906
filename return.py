#!/usr/bin/python
import sys, os
# import commands
import re

# sys.path.append('/home/sharedFold/WinQ/')
# from IndexData import IndexData

path1 = '/data/stock/newSystemData/rawdata/wind/stock_eod'
path2 = '/data/nvme/yingda/2'


def estab_day_list(day_begin, day_end):
    date_list = []
    year_begin = day_begin // 10000
    month_begin = day_begin // 100 % 100
    date_begin = day_begin % 100
    year_end = day_end // 10000
    month_end = day_end // 100 % 100
    date_end = day_end % 100
    if year_end > year_begin:
        interval = (year_end - year_begin) * 372 + (month_end * 31 + date_end) - (month_begin * 31 + date_begin)
    else:
        interval = (month_end * 31 + date_end) - (month_begin * 31 + date_begin)
    # print(interval)
    if interval < 0:
        print("cannot get the right date interval")
    else:
        date_list.append(day_begin)
        date = date_begin
        month = month_begin
        year = year_begin
        if interval != 0:
            for i in range(interval):
                if date < 31:
                    date += 1
                elif month < 12:
                    month += 1
                    date = 1
                else:
                    year += 1
                    month = 1
                    date = 1
                day = year * 10000 + month * 100 + date
                date_list.append(day)
    return date_list


def get_1day_data(day):  # day is int, e.g. 20160503
    data = {}
    sday = str(day)
    year = sday[:4]
    month = sday[4:6]
    day_path = path1 + '/' + str(year) + '/' + str(month) + '/' + str(sday) + '.csv'
    f = open(day_path, 'r')
    lines = f.readlines()
    for line in lines:
        sline = line.strip()
        if sline == '':
            continue
        items = re.split('\s+', sline)
        stock = str(items[0])
        oo = float(items[1])
        cc = float(items[4])
        isopen = int(items[-3])
        adj = float(items[-2])
        data[stock] = {}
        data[stock]['isopen'] = isopen
        data[stock]['adj'] = adj
        data[stock]['open'] = oo
        data[stock]['close'] = cc
    f.close()

    return data


def get_1day_ret(day):
    # lastday = IndexData().get_front_day(str(day))
    lastday = day_list[day_list.index(day) - 1]
    print('lastday', lastday)
    data1 = get_1day_data(lastday)
    data = get_1day_data(day)


# print ( data1['600276']['isopen'],data1['600276']['adj'],data1['600276']['open'],data1['600276']['close'] )
# print ( data['600276']['isopen'],data['600276']['adj'],data['600276']['open'],data['600276']['close'] )
# print ( data['600276']['close'] / data['600276']['adj'] / data1['600276']['close'] )

if __name__ == '__main__':
    # start_day = 20160101
    # end_day = 20190214
    # day_list = IndexData().get_deal_day_list_in_period(start_day,end_day)
    day_list = estab_day_list(20190601, 20190630)
    global_info_dict = {}
    for day in day_list:
        sday = str(day)
        year = sday[:4]
        month = sday[4:6]
        day_path = path1 + '/' + str(year) + '/' + str(month) + '/' + str(sday) + '.csv'
        if not os.path.exists(day_path):
            continue
        else:
            file_path = path2 + '/' + str(day) + '.csv'
            f = open(file_path, 'w')
            data = get_1day_data(day)
            for stock in data:
                # do not calculate return FirstTime
                FirstTime = False
                # default line
                line = stock + '\tnan\tnan\n'
                if (not stock in global_info_dict) and data[stock]['isopen']:
                    # if the code is not in global_info_dict and the code is open on 'day'
                    global_info_dict[stock] = {}
                    FirstTime = True
                if not FirstTime:
                    # first time denotes the first day the code occurs
                    if data[stock]['isopen']:
                        # print day,stock
                        ret_c_c = data[stock]['close'] / data[stock]['adj'] / global_info_dict[stock][
                            'last_real_close'] - 1.0
                        ret_o_o = data[stock]['open'] / data[stock]['adj'] / global_info_dict[stock][
                            'last_real_open'] - 1.0
                        line = stock + '\t' + str(ret_c_c) + '\t' + str(ret_o_o) + '\n'
                f.writelines(line)
                # refresh global info
                if data[stock]['isopen']:
                    global_info_dict[stock]['last_trade_day'] = day
                    global_info_dict[stock]['last_real_open'] = data[stock]['open']
                    global_info_dict[stock]['last_real_close'] = data[stock]['close']
            f.close()
