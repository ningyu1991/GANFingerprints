#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:42:24 2018

@author: mikolajbinkowski
"""
import time
   
class Timer(object):
    def __init__(self, start_time=time.time(), limit=100):
        self.start_time = start_time
        self.limit = limit
        
    def __call__(self, step, mess='', prints=True):
        if prints and (step % self.limit != 0) and (step > 10):
            return
        message = '[%8d][%s] %s' % (step, hms(self.start_time), mess)
        if prints:
            print(message)
        else:
            return message
        

def hms(start_time):
    t = int(time.time() - start_time)
    m, s = t//60, t % 60
    h, m = m//60, m % 60
    if h > 0:
        return '%2dh%02dm%02ds' % (h, m, s)
    elif m > 0:
        return '%5dm%02ds' % (m, s)
    else:
        return '%8ds' % s