#!/usr/bin/python3

import csv
import inspect
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import yaml

from functools import wraps

class Helper():
    def __init__(self,input_directory, output_directory, input_prm_file):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.output_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, output directory = {}, input parameter file  = {}'.\
               format(self.input_directory, self.output_directory, self.input_prm_file))
        
    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)
        self.font_size = self.conf.get('font_size')
        self.optimize_strat = self.conf.get('optimize_strat')
        self.metrics = self.conf.get('metrics')
        self.scan_leverage = self.conf.get('scan_leverage')
        
    @staticmethod
    def timing(f):
        """Decorator for timing functions
        Usage:
        @timing
        def function(a):
        pass
        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            end = time.time()
            print('function:%r took: %2.2f sec' % (f.__name__,  end - start))
            return(result)
        return wrapper

    @staticmethod
    def write_to_xls(dct,filename):
        df = pd.DataFrame(dct, index=[0])
        df.T.to_excel(filename)

        


        
