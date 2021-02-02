#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import yaml

from backtesting.lib import plot_heatmaps
from dt_help import Helper
from itertools import combinations
from typing import Callable, List, Union

class OptimizeStrat():
    def __init__(self,input_directory, output_directory, input_prm_file):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.output_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, output directory = {}, input parameter file  = {}'.\
               format(self.input_directory, self.output_directory, self.input_prm_file))
    
    @Helper.timing
    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)
        self.price_delta = self.conf.get('price_delta')        
        self.size_lot = self.conf.get('size_lot')        
        self.n_days = self.conf.get('n_days')
        self.var_opt = self.conf.get('var_opt')
        self.max_tries = self.conf.get('max_tries')
        self.random_state = self.conf.get('random_state')
        self.grouping = self.conf.get('grouping')
        self.divq = self.conf.get('divq')
        self.short_wnd_ = self.conf.get('short_wnd')
        self.long_wnd_ = self.conf.get('long_wnd')
        self.atr_factor_ = self.conf.get('atr_factor')
        self.avail_liq_ = self.conf.get('avail_liq')
        
    @Helper.timing
    def optimize_OpenClose(self,bkt):
        self.price_delta = range(self.price_delta['start'],self.price_delta['stop'],self.price_delta['step'])
        self.size_lot = range(self.size_lot['start'],self.size_lot['stop'],self.size_lot['step'])

        self.price_delta = [ float(el)/self.divq for el in self.price_delta ]
        self.size_lot = [ float(el)/self.divq for el in self.size_lot ]
        
        stats, heatmap = bkt.optimize(
            price_delta= self.price_delta,
            size_lot = self.size_lot,      
            maximize=self.var_opt,
            max_tries=self.max_tries,
            random_state=self.random_state,
            return_heatmap=True)

        hm = heatmap.groupby(self.grouping).mean().unstack()
        opt_params = heatmap.sort_values().iloc[-1:]
        opt_params = pd.DataFrame(opt_params).reset_index().drop(columns=[self.var_opt])
        self.best_params = opt_params.to_dict('records')
        if(len(self.best_params) == 1):
            self.best_params = self.best_params[0]
        print(self.best_params)
        sns.heatmap(hm[::-1], cmap='jet')
        plt.show()

    @Helper.timing
    def optimize_SmaCross(self,bkt):
        self.short_wnd_ = range(self.short_wnd_['start'],self.short_wnd_['stop'],self.short_wnd_['step'])
        self.long_wnd_ = range(self.long_wnd_['start'],self.long_wnd_['stop'],self.long_wnd_['step'])
        self.atr_factor_ = range(self.atr_factor_['start'],self.atr_factor_['stop'],self.atr_factor_['step'])
        self.avail_liq_ = range(self.avail_liq_['start'],self.avail_liq_['stop'],self.avail_liq_['step'])
        self.avail_liq_ = [ el/100 for el in self.avail_liq_ ]
        all_parameters = [ el for el in dir(bkt._strategy) if not el.endswith('__') and el.endswith('_') ]
        
        stats, heatmap = bkt.optimize(
            short_wnd_ = self.short_wnd_,
            long_wnd_ = self.long_wnd_,
            atr_factor_ = self.atr_factor_,
            avail_liq_ = self.avail_liq_,
            constraint=lambda p: p.short_wnd_ < p.long_wnd_,
            maximize=self.var_opt,
            max_tries=self.max_tries,
            random_state=self.random_state,
            return_heatmap=True)

        best_params_list = []
        hm_list = []
        
        for el in combinations(all_parameters,2):
            hm = heatmap.groupby(list(el)).mean().unstack()
            opt_params = heatmap.sort_values().iloc[-1:]
            opt_params = pd.DataFrame(opt_params).reset_index() # .drop(columns=[self.var_opt])
            self.best_params = opt_params.to_dict('records')
            if(len(self.best_params) == 1):
                self.best_params = self.best_params[0]
            best_params_list.append(self.best_params)
            hm_list.append(hm[::-1])

        fig = plt.figure(figsize=(32,20))
        idx = 0
        for el in hm_list:
            idx += 1 
            ax = fig.add_subplot(2,len(hm_list)//2, idx)
            sns.heatmap(el, cmap='jet')
        plt.savefig('/'.join([self.output_directory,'opt_params_strat.pdf']))
        plt.show()

        df_best_params = pd.DataFrame(best_params_list)
        optimal_params = df_best_params.iloc[df_best_params[self.var_opt].idxmax()].to_frame().reset_index()
        optimal_params.columns = ['parameters','values']
        optimal_params.to_csv('/'.join([self.output_directory,'opt_params_strat.csv']),index=False)
        print(df_best_params)

        
