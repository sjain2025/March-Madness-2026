'''
    utils.py

    Copyright (c) 2025, Reid Simmons, Carnegie Mellon University
      This software is distributed under the terms of the 
      Simplified BSD License (see ipc/LICENSE.TXT)
'''
import pandas as pd

def read_data(file_name, which='M'):
    return pd.read_csv('data/'+which+file_name+".csv")

def read_predictions(whom, which='M'):
    return pd.read_csv('submissions/%sNCAATourneyPredictions - %s.csv'
                       %(which, whom))

def read_tourney_predictions(which='M', folder='predictions'):
    """
    Load the global predictions file used by `Bracket.fill()`.
    Defaults to reading from `predictions/{which}NCAATourneyPredictions.csv`.
    """
    return pd.read_csv(f"{folder}/{which}NCAATourneyPredictions.csv")