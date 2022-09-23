'''
Monte-Carlo Portfolio Backtesting Toolkit
'''

#Startup Code Begin
#import os
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import torch
from scipy.stats import t,lognorm,gmean,gstd
from sklearn.cluster import KMeans
from datetime import datetime,timedelta
#Startup Code End


class PortfolioManager(object):
    def __init__(self,asset_list,risk_free_rate):
        '''
        :type asset_list: list of strings
        :param asset_list: list of tickers *must be uppercase, eg. SPY vs. spy

        :type cost_per_share: list of floats
        :param cost_per_share: average cost per share for each asset
        '''
        self.assets = asset_list
        self.Rf = risk_free_rate

    @classmethod
    def create_portfolio(cls,tickers,cost_per_share,Rf):
        tickers = [x.upper() for x in tickers]
        cps = np.asarray(cost_per_share)
        pcts = cps/cps.sum()
        pm1 = cls(tickers,Rf)
        pm1.percents = pcts
        return(pm1)

    def sharpe_ratio(self,Rp):
        exp_value = (Rp-self.Rf).mean(axis=0)
        sigma = (Rp-self.Rf).std(axis=0)
        sr = exp_value/sigma
        return(sr)

    def get_data(self,ndays=1250):
        #Create a datetime list
        end = datetime.today()
        start = datetime.today()-timedelta(ndays) # roughly the past 5 years of data
        ind = pd.date_range(start=(start+timedelta(days=1)).strftime('%Y-%m-%d'),end=end.strftime('%Y-%m-%d'),freq='B') #Business day frequency

        dfs = []
        for asset in self.asset_list:
            temp = web.DataReader(asset,'yahoo',start=start.strftime('%Y-%m-%d'),end=end.strftime('%Y-%m-%d'))
            temp['log_returns'] = np.log(1+temp.Close.pct_change())
            df = temp.dropna()
            #Fill values using reindex
            dfs.append(df.reindex(ind))

        #Get weekly returns
        wkdays = np.array([x.weekday() for x in ind])
        mdays = np.where(wkdays==4)[0][:-1]    #mondays

        friday_close = np.zeros(shape=(len(ind),len(self.assets)))
        weekly_returns = np.zeros(shape=(len(ind2),int(len(self.asset_list)*2)))   #weekly return matrix -> used to score portfolios
        weekly_trends = np.zeros_like(weekly_returns)   #weekly trend matrix -> used in monte carlo
        xr = np.arange(5)
        for i in range(len(ind2)):
            r = ind2[i]
            for j in range(len(self.asset_list)):
                df = dfs[j]
                wr = df.iloc[r].Close.values
                r1,r2 = wr[0],wr[-1]
                friday_close[i][j] = r2
                weekly_returns[i][j] = ((r2-r1)/r1)

                slope = np.polyfit(xr,wr,0)[0]
                weekly_trends[i][j] = slope


        lr = np.zeros(shape=(len(ind),int(len(self.asset_list)*2))) #Log return matrix of daysxasset -> used for monte carlo
        for i in range(len(dfs)):
            df = dfs[i]
            lr[:,i]=df.log_returns.values

        data = dict(FridayClose=friday_close,WeeklyReturns=weekly_returns,WeeklyTrends=weekly_trends,LogReturns=lr)
        return(data)

    def prep_mc(self,volatility_dir,trend_matrix):
        '''
        :type volatility_dir: str
        :param volatility_dir: directory to volatility arrays; ***format = r'E:/Investing_Data/vol/{}.csv'*** where {}=TICKER

        :type trend_matrix: np array
        :param trend_matrix: Matrix of format ndays x assets. Contains slopes from OLS linear fit.
        '''
        dd = dict()
        for i in range(len(self.asset_list)):
            #Load volatility values
            ticker = self.asset_list[i] #Must be uppercase, eg. 'AMD'
            volatility = np.loadtxt(volatility_dir.format(ticker),delimiter=',')

            #Fit volatility and create random values
            vfit = lognorm.fit(volatility)
            vdist = lognorm.rvs(*vfit,size=1e5)

            #Get vol levels using kmeans clustering
            v_cluster_model = KMeans(n_clusters=3)
            v_cluster_model.fit(volatility.reshape(-1,1))
            v_labels = v_cluster_model.labels_

            #Get Transition Matrix for Volatility Values
            tm = np.zeros(shape=(3,3))
            for i in range(1,len(v_labels)):
                past,current = v_labels[i-1],v_labels[i]
                tm[past][current]+=1

            #Get probability matrix
            t_matrix = (tm/tm.sum(axis=1)).cumsum(axis=1)

            #Create dictionary of levels:values using vdist and v_labels
            vdist_labels = v_cluster_model.fit_predict(vdist.reshape(-1,1))
            vdict = dict(str(k):vdist[vdist_labels==k] for k in range(3))     #3 clusters; labels = 0,1,2

            #Fit the trends and use clustering to get downtrends, random walks, and uptrends
            trends = trend_matrix[:,i]
            tr_cluster_model = KMeans(n_clusters=3) #3 types of trend
            tr_cluster_model.fit(trends)
            tr_labels = tr_cluster_model.labels_

            #Dictionary of levels:trends
            tdict = {str(k):trends[tr_labels==k] for k in range(3)}

            dd[ticker] = (tdict,vdict,t_matrix)

        return(dd)

    def stochastic_mc(self,data_dict,degrees_freedom,tsteps=5,npaths=1e5):
        '''
        :type data_dict: dictionary
        :param data_dict: data_dict[TICKER] = (trend_dict,volatility_dict,vol_transition_matrix)
        
        :type degrees_freedom: float
        :param degrees_freedom: guess based on t fit of past log returns

        :type tsteps: int
        :param tsteps: period the mc is prediction; needs to match the period used for returns in portfolio

        :type npaths: int
        :param npaths: number of paths MC runs
        '''
        forecasts = {}
        for ticker in self.asset_list:
            td,vd,t_matrix = data_dict[ticker]
            
            #Set up initial volatility array, sigma
            k = [x for x in vd.keys()]
            wgt = np.array([len(vd[x]) for x in k])
            wgt = wgt/wgt.sum()
            k2 = [int(x) for x in k]
            vlevels = np.random.choice(k2,weights=wgt) #weighted choices based on previous volatility levels

            #Get trends
            trend_k = [x for x in td.keys()]
            trend_k2 = [int(x) for x in trend_k]
            wgt = np.array([len(td[x]) for x in trend_k])
            wgt = wgt/wgt.sum()
            trend_array = np.random.choice(trend_k2,weights=wgt,size=npaths)

            prd = trend_array+t.rvs(df=degrees_freedom[ticker],loc=0.0,scale=1.0,size=(tsteps,npaths))    #daily prediction container
            for i in range(tsteps):
                #get vdist values from sigma levels and update vlevels array
                new_vlevels = np.zeros_like(vlevels)
                sigma = np.zeros_like(vlevels)
                for j in range(3):
                    ind = np.where(vlevels==j)[0]
                    if len(ind)!=0:
                        sigma[ind] = np.random.choice(vd[str(j)],size=len(ind))
                        new_vlevels[ind] = np.random.choice(k2,weights=t_matrix[j],size=len(ind))
                    else:
                        pass

                #update vlevels
                vlevels=new_vlevels

                #Multiply prd matrix by volatility values
                prd[i]*=sigma

            forecast = np.vstack((np.ones(npaths),np.exp(prd)))
            forecasts[ticker] = forecast

        return(forecasts)       

    def backtest_random_portfolio(self,nweeks=52,nportfolios=1e10,npaths=1e5):
        '''
        '''

        #Load data
        data = self.get_data() #dictionary

        #Create random portfolios
        rdraw = torch.rand(nportfolios,int(len(self.asset_list)*2),dtype=torch.float32)  #Creates torch tensor uniformly distributed between 0 and 1 as a CPU object
        ports = rdraw/rdraw.sum(axis=1).reshape(-1,1)

        #Score portfolios on Monte Carlo Values
        mc_data = self.prep_mc(r'E:/Investing_Data/volatility/{}.csv',data['WeeklyTrends'])
        dof_dict = dict()
        for i in range(len(self.assets)):
            lr = data['LogReturns'][:,i]
            dof = t.fit(lr)[0]
            dof_dict[self.assets[i]] = dof

        forecasts = self.stochastic_mc(mc_data,dof_dict,npaths)

        fr_close = data['FridayClose'][-nweeks:]    #S0 for prediction
        wr = data['WeeklyReturns'][-nweeks:]        #observed returns
        stdev = data['WeeklyReturns'][:-nweeks].std(axis=0)   #long run standard deviation
        
        port_returns = np.zeros(shape=(nweeks,nportfolios))
        for i in range(nweeks):
            #Get MC prediction
            prd_returns = np.zeros(shape=(npaths,int(len(self.asset_list)*2)))
            for j in range(len(self.asset_list)):
                s0 = fr_close[i][j]
                prd = s0*forecast[self.asset_list[j]].cumprod(axis=0)[-1]
                pct_return_long = (prd-s0)/s0
                pct_return = np.concatenate((pct_return_long,1-pct_return_long))
                prd_returns[:,j] = pct_return

            #Get weighted average of MC returns
            if i<nkweeks-1:
                s0 = fr_close[i]
                s1 = fr_close[i+1]
                

            else:
                s0 = fr_close[i]
                s1 = np.array([gmean(prd_returns[:,i]) for i in range(prd_returns.shape[1])])

            obs_return = (s1-s0)/s0
            w = abs(prd_returns-obs_return)/obs_return
            wt = 1-(w/w.sum(axis=1).reshape(-1,1))  #weight array for weighted average

            avg_return = np.average(prd_returns,weights=wt)

            #Get portfolio returns
            port_returns[i] = np.einsum('ij,j->i',ports,avg_return)  #returns sum of portfolio returns

        #Score the portfolios
        pr_tensor = torch.tensor(port_returns,dtype=torch.float16,device=0)     #GPU tensor of half precision floats. Much faster sort time with lower memory usage.
        weekly_port_scores = torch.argsort(port_returns,dim=1,descending=True) #Largest to smallest absolute returns

        sr = self.sharpe_ratio(port_returns,self.Rf)
        sharpe_ratio_scores = np.argsort(sr)[::-1]  #Largest to smallest sharpe ratio indexes
            
        
        return(port_returns,weekly_port_scores,sharpe_ratio_scores)