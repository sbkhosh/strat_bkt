#!/usr/bin/python3

import numpy as np
import pandas as pd

from backtesting.test import SMA
from backtesting import Strategy
from backtesting.lib import SignalStrategy, TrailingStrategy

class SmaCross(SignalStrategy,TrailingStrategy):
    short_wnd_ = 25
    long_wnd_ = 180
    atr_factor_ = 4
    avail_liq_ = 0.91
    
    def init(self):
        # super method to properly initialize the parent classes
        super().init()
        
        # Precompute the two moving averages
        sma1 = self.I(SMA, self.data['Close'], self.short_wnd_)
        sma2 = self.I(SMA, self.data['Close'], self.long_wnd_)
        
        # Where sma1 crosses sma2 upwards. Diff gives [-1,0,1]
        signal = (pd.Series(sma1) > sma2).astype(int).diff().fillna(0)
        signal = signal.replace(-1, 0)  # Upwards/long only
        
        # Use 95% of available liquidity (at the time) on each order.
        # (Leaving a value of 1. would instead buy a single share.)
        entry_size = signal * self.avail_liq_
                
        # Set order entry sizes using the method provided by SignalStrategy
        self.set_signal(entry_size=entry_size)
        
        # Set trailing stop-loss to 2x ATR using
        # the method provided by `TrailingStrategy`
        self.set_trailing_sl(self.atr_factor_)

        
