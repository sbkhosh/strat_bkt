#!/usr/bin/python3

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import yaml

from datetime import datetime, timedelta
from dt_help import Helper
from sqlalchemy import create_engine

class HistData():
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
        self.new_db = self.conf.get('new_db')

    @Helper.timing
    def process(self):
        HistData.get_data(self)
        self.raw_cols = ['Open-Close_1','Open-Close','Open','High','Low','Close','Volume']
                
    @Helper.timing
    def get_data(self):
        if(self.new_db):
            df = pd.read_excel(self.input_directory + "/QQQ.xls")
            df.drop(columns=['Ticker'],inplace=True)
            df.columns = ['Date','Open-Close_1','Open-Close','Open','High','Low','Close','Volume']
            
            engine_data_raw = create_engine("sqlite:///" + self.output_directory + "/data_raw.db", echo=False)
            df.to_sql(
                'data_raw',
                engine_data_raw,
                if_exists='replace',
                index=True
            )
            self.data_raw = pd.read_sql_table(
                'data_raw',
                con=engine_data_raw
                ).drop(columns=['index']).set_index('Date')
            self.data_raw.index = pd.to_datetime(self.data_raw.index,format='%Y%m%d')
        else:
            engine_data_raw = create_engine("sqlite:///" + self.output_directory + "/data_raw.db", echo=False)
            self.data_raw = pd.read_sql_table(
                'data_raw',
                con=engine_data_raw
                ).drop(columns=['index']).set_index('Date')
            self.data_raw.index = pd.to_datetime(self.data_raw.index,format='%Y%m%d')
