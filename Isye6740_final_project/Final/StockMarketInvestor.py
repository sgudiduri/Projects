
from cvxpy import *
import numpy as np
import pandas as pd
from pathlib import Path  
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

'''
 Responsible to returning stocks based on parameters
'''
class StockMarketInvestor:
    
    def __init__(self, df, req_return=0.1, include=list(), exclude=list(), gamma=0):
        self.df = df
        self.req_return = req_return
        self.include = include
        self.exclude = exclude
        self.gamma = gamma
        
    def train(self):
        mp  = self.df
        mr = mp.resample('BMS').first()
        mr = mr.pct_change().dropna()
        symbols = mr.columns
        gamma = Parameter(nonneg=True)
        #r = expected return of stock i relative to market   
        r = mr.mean(axis=0).to_numpy()
        #C = covariance of returns of stocks i and j
        C = mr.cov().to_numpy()
        #n = number of stocks
        n = len(symbols)
        # The variables vector
        x = Variable(n)
        # Return on Investment
        ret = r.T@x        
        # The risk in xT.Q.x format
        risk = quad_form(x, C)
        constraints = []
        # The core problem definition with the Problem class from CVXPY
        constraints += [sum(x)==1, ret >= self.req_return, x >= 0]
        
        ## Include stocks that you have choosen.
        for i in self.include:
            ind = list(symbols).index(i)
            constraints += [x[ind] >= 0.01]
            constraints += [x[ind] <= 0.99]
            
        ## Excluse stocks that you have choosen.    
        for i in self.exclude:
            ind = list(symbols).index(i)
            constraints += [x[ind] <= 0.0]
        
        exp_ret = []
        exp_risk = []
        try:
            SAMPLES = 25
            prob = Problem(Maximize(ret-gamma*risk), constraints)
            gamma_vals = np.logspace(-1.5, 1, num=SAMPLES)
            for i in range(SAMPLES):        
                gamma.value = gamma_vals[i]
                prob.solve()
                rets = round(100*ret.value,2)
                rsks = round(100*risk.value**0.5,2)
                exp_ret.append(rets)
                exp_risk.append(rsks)                  
                if rsks - rets < 1 and self.gamma == 0:
                    self.gamma = gamma_vals[i]                  
        except:
            print ("Error")  
            
        plt.figure(figsize=(10,6)) 
        plt.plot(gamma_vals, exp_ret)
        plt.plot(gamma_vals, exp_risk)         
        plt.show()
        
        print(f"Optimal gamma @ {self.gamma}")
        
    def optimize(self):
        df_copy = self.df.copy()
        df_copy = df_copy.resample('BMS').first()
        df_copy = df_copy.pct_change().dropna()
        symbols = self.df.columns
        #r = expected return of stock i relative to market   
        r = df_copy.mean(axis=0).to_numpy()
        #C = covariance of returns of stocks i and j
        C =  df_copy.cov().to_numpy()
        #n = number of stocks
        n = len(symbols)
        # The variables vector
        x = Variable(n)
        # Return on Investment
        ret = r.T@x        
        # The risk in xT.Q.x format
        risk = quad_form(x, C)
        constraints = []
        # The core problem definition with the Problem class from CVXPY
        constraints += [sum(x)==1, ret >= self.req_return, x >= 0]
        
        ## Include stocks that you have choosen.
        for i in self.include:
            ind = list(symbols).index(i)
            constraints += [x[ind] >= 0.01]
            constraints += [x[ind] <= 0.99]
            
        ## Excluse stocks that you have choosen.     
        for i in self.exclude:
            ind = list(symbols).index(i)
            constraints += [x[ind] <= 0.0]
            
        prob = Problem(Maximize(ret-self.gamma*risk), constraints)
        df_ret = pd.DataFrame(list(), columns=["Symbol", "InvestmentPercentage"])
        try:
            prob.solve()
            print ("Optimal portfolio")
            print ("----------------------")
            for s in range(len(symbols)):
                if(round(100*x.value[s]) > 0.0001):
                    df_ret.loc[len(df_ret.index)] = [symbols[s], round(100*x.value[s])]
                    print (" Investment in {} : {}% of the portfolio".format(symbols[s],round(100*x.value[s])))
            print ("----------------------")
            print ("Exp ret = {}%".format(round(100*ret.value,2)))
            print ("Expected risk    = {}%".format(round(100*risk.value**0.5,2)))
            
            return df_ret
        except:
            print ("Error")    
            
    def evaluate(self, ret,sp_hist, investment):
        spNew = sp_hist[list(ret["Symbol"])].resample('BMS').first().pct_change().dropna()        
        amount = 0
        for i in list(ret["Symbol"]):   
            v1 = list(ret[ret["Symbol"] == i]["InvestmentPercentage"])[0]
            v2 = list(spNew[len(spNew)-1:][i])[0]            
            amount += (((v1/100) * investment)  + v2)   
        
        return f"For Investment amount: ${round(investment,2)}, Expected return: ${round((amount-investment), 2)}"
        