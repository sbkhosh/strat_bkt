#!/usr/bin/python3

import itertools
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd 
import pprint
import seaborn as sns
import warnings

from backtesting import Backtest
from dt_help import Helper
from dt_optmz import OptimizeStrat
from dt_pdr import HistData
from dt_strat import SmaCross
from pandas.plotting import register_matplotlib_converters

warnings.filterwarnings('ignore',category=FutureWarning)
pd.options.mode.chained_assignment = None 
register_matplotlib_converters()
matplotlib.use( 'tkagg' )

CASH = 100_000

def target(params):
    dct = {'margin': params[0], 'commission': params[1], 'data': params[2],
           'strat': params[3], 'cash': params[4]}
    bt_obj = Backtest(dct['data'], dct['strat'], cash=dct['cash'], commission=dct['commission'], margin=dct['margin'])

    res = bt_obj.run()
    case = {(params[0],params[1]): [res.get(k) for k in metrics]}
    return(case)

def get_filtered(df,lwt,upt):
    lower_threshold=lwt
    upper_threshold=upt
    booleangrid=(np.asarray(df.values) > lower_threshold) & (np.asarray(df.values) < upper_threshold)
    intgrid=booleangrid*1

    down=[];up=[];left=[];right=[]
    for i, eachline in enumerate(intgrid):
        for j, each in enumerate(eachline):
            if each==1:
                down.append([[j,j+1],[i,i]])
                up.append([[j,j+1],[i+1,i+1]])
                left.append([[j,j],[i,i+1]])
                right.append([[j+1,j+1],[i,i+1]])

    together=[]
    for each in down:
        together.append(each)
    for each in up:
        together.append(each)
    for each in left:
        together.append(each)
    for each in right:
        together.append(each)

    filtered=[]
    for each in together:
        c=0
        for EACH in together:
            if each==EACH:
                c+=1
        if c==1:
            filtered.append(each)
    return(filtered)

def plot_heatmap(df):
    ax = sns.heatmap(df)

    # format text labels
    fmt = '{:0.3f}'
    xticklabels = []
    for item in ax.get_xticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        xticklabels += [item]
    yticklabels = []
    for item in ax.get_yticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        yticklabels += [item]

    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    
    plt.xlabel('margin')
    plt.ylabel('commission')
    plt.title('Equity Final [$]')
    

if __name__ == '__main__':
    obj_helper = Helper('data_in','data_out','conf_help.yml')
    obj_helper.read_prm()
    
    metrics = obj_helper.metrics
    scan_leverage = obj_helper.scan_leverage
    
    data_obj = HistData('data_in','data_out','conf_pdr.yml')
    data_obj.read_prm()
    data_obj.process()

    ######################################################################
    #                      Principle of the strategy                     #
    ######################################################################
    # moving average cross-over with trailing stop based on ATR value
    ######################################################################

    ######################################################################
    #                          Logic of files                            #
    ######################################################################
    # (1) reading files and processing => dt_pdr.py
    # (2) strategy implementation => dt_strat.py
    # (3) stratrgy optimization => dt_optmz.py
    # (4) helper functions and parameters settings => dt_help.py
    ######################################################################

    ######################################################################
    #                  Output after executing exec.py)                   #
    ######################################################################
    # (1) an html file called SmaCross.html
    # (2) in data_out/summary_strat.xls giving the metrics
    ######################################################################

    print('##################################################################')
    print('  optimizing strategy parameters with 0 commission & no leverage  ')
    print('##################################################################')

    # optimize strategy by parameters given in data_in/conf_optmz.yml
    # commission is set to 0 and no leverage is included
    bt_obj = Backtest(data_obj.data_raw, SmaCross, cash = CASH, commission=0.0, margin=1.0)
    optmz_obj = OptimizeStrat('data_in','data_out','conf_optmz.yml')
    optmz_obj.read_prm()
    optmz_obj.optimize_SmaCross(bt_obj)

    print('##################################################################')
    print('                 optimized strategy paramters set                 ')
    print('     running optimal strategy with 0 commission & no leverage     ')
    print('##################################################################')    

    # once optimize with the commission set to 0 and no leverage
    # the strategy is simulated with the above optimal parameters
    bt_obj = Backtest(data_obj.data_raw, SmaCross, cash = CASH, commission=0.0, margin=1.0)
    val_parameters = pd.read_csv('/'.join([obj_helper.output_directory,'opt_params_strat.csv']))    
    all_parameters = [ el for el in dir(bt_obj._strategy) if not el.endswith('__') and el.endswith('_') ]

    # ensure integer values for the SmaCross window parameters
    for el in range(len(all_parameters)):
        if('wnd' in val_parameters['parameters'].iloc[el]):
            setattr(bt_obj._strategy,val_parameters['parameters'].iloc[el],int(val_parameters['values'].iloc[el]))
        else:
            setattr(bt_obj._strategy,val_parameters['parameters'].iloc[el],val_parameters['values'].iloc[el])
            
    res = bt_obj.run()
    case = {k: res.get(k) for k in metrics}
    bt_obj.plot()
    pprint.pprint(case)
    obj_helper.write_to_xls(case, '/'.join([obj_helper.output_directory,'strat_summary.xlsx']))

    print('##################################################################')
    print('  scanning over commission & leverage with the optimal strategy   ')
    print('##################################################################')    
    # after optimization and testing of specified parameters
    # then scan over commission and margins parameters

    margins = np.linspace(0.1,0.5,41)
    commissions = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]
    list_obj = [data_obj.data_raw]
    list_strat = [SmaCross]
    list_cash = [CASH]

    # launch scan with multiprocessing
    paramlist = list(itertools.product(margins,commissions,list_obj,list_strat,list_cash))
    pool = multiprocessing.Pool()
    case_list = pool.map(target,paramlist)
    pool.close()
    pool.join()

    # transforming the output of the scan 
    df_res = pd.DataFrame.from_dict(case_list).T 
    df_res.columns = [ 'values_' + str(el) for el in range(len(margins)*len(commissions)) ]
    df_res.index = pd.MultiIndex.from_tuples(df_res.index)
    midx = pd.MultiIndex.from_tuples(list(zip(df_res.index,df_res.columns)))
    df_res = pd.DataFrame(data=np.diag(df_res), index=midx).reset_index(level=1, drop=True)
    df_res.columns = ['values']
    
    df_new = pd.DataFrame(df_res)
    df_new[metrics] = pd.DataFrame(df_new['values'].tolist(), index=df_new.index)
    df_new.drop(columns=['values'],inplace=True)
    df_new.index = pd.MultiIndex.from_tuples(df_new.index, names=['margin', 'commission'])
    df_save = df_new.reset_index().pivot(columns='margin',index='commission',values='Equity Final [$]') 
  
    # plot heatmap of equity value vs margins & commissions
    plot_heatmap(df_save)
    
    # adding on previous heatmap specific equity value regions
    buy_and_hold = df_new[[ el for el in metrics if 'Hold' in el ]].iloc[0].values[0]
    filtered = get_filtered(df_save,buy_and_hold*1_000,500*1_000)
    [ plt.plot(filtered[x][0],filtered[x][1],c='black', linewidth=2) for x in range(len(filtered)) ] 
    
    plt.savefig('/'.join([obj_helper.output_directory,'opt_commarg_strat.pdf']))
    plt.tight_layout()
    plt.show()
