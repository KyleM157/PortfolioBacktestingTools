'''
Synthetic Data Creation Script
'''

#Startup Code Begin
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import torch
#Startup Code End

class PlotSelection(object):
    def __init__(self,y):
        self.y=y
        self.degree = 1

    def on_key(self,event):
        self.degree = int(event.key)

    def get_degree(self):
        fig,axes = plt.subplots(nrows=4,ncols=2,sharex=True,sharey=True)
        cid = fig.canvas.mpl_connect('key_press_event',self.on_key)
        xr = np.arange(len(self.y))
        fx = lambda n: np.poly1d(np.polyfit(xr,self.y,n))(xr)
        for i in range(4):
            for j in range(2):
                n = (2*i)+j+1
                axes[i][j].plot(xr,self.y,'k')
                axes[i][j].plot(xr,fx(n),'r')
                axes.set_title('N = {}'.format(n))
        plt.show()


class Portfolio(object):
    def __init__(self,tickers):
        self.tickers = tickers

    def create_portfolio(self):
        print("Enter Percentage as an Integer for Each Ticker\n")
        port = np.zeros(len(tickers),dtype=np.float32)
        for i in range(len(self.tickers)):
            pct = int(input("{}: ".format(self.tickers[i])))/100.
            port[i] = pct

        self.positions = port

    def get_historic_data(load_data=True,ndays=370):
        '''
        :type ndays: int
        :param ndays: number of days to load; needs to be more than a year
        '''
        if load_data==True:
            tday = datetime.today()
            end=(tday-timedelta(days=1)).strftime('%Y-%m-%d')
            start = (tday-timedelta(days=ndays)).strftime('%Y-%m-%d')

            df_list = []
            for x in self.tickers:
                df_list.append(web.DataReader(x,'yahoo',start,end))

        else:
            fname = r'E:/Investing/historical_data/{}.csv'
            df_list = []
            for x in self.tickers:
                df = pd.read_csv(fname.format(x),index_col='Date')
                df_list.append(df)

        return(df_list)


    def get_synthetic_data(self,data,npaths=100,fit_data=True):
        '''
        :type data: pandas DataFrame
        :param data: Data to create synthetic data

        :type ndays: int
        :param ndays: time range to load

        :type npaths: int
        :param npaths: Number of paths of synthetic data to create


        Loads data from yahoo finance and returns a dictionary of ticker:price pairs
        '''
        sdata = {}
        for i in range(len(self.tickers)):
            df = data[self.tickers[i]][-241:]
            c = df.Close.values
            rr = np.diff(c)

            m = abs(np.fft.rfft(rr))
            phi = np.random.uniform(-np.pi,np.pi,size=(npaths,len(r)))

            #Get Trend Line using Polyfit
            if fit_data==True:
                ps = PlotSelection(c)
                ps.get_degree()

                if ps.degree==1:
                    phi[:,0]=0.
                    z = m*np.exp(phi*1j)
                    sr = np.fft.irfft(z)
                    
                else:
                    xr=np.arange(len(c))
                    fx = lambda n: np.poly1d(np.polyfit(xr,c,ps.degree))(xr)
                    yy = fx(ps.degree)
                    dy = np.diff(yy)

                    m[0] = 0.
                    z = m*np.exp(phi*1j)
                    sr = np.fft.irfft(z)+dy

            else:
                phi[:,0]=0.
                z = m*np.exp(phi*1j)
                sr = np.fft.irfft(z)


            rmatrix = np.hstack((np.full(shape=(npaths,1),c[0]),sr)).astype(np.float32)
            prices = rmatrix.cumsum(axis=1)

            sdata[self.tickers[i]] = prices

        return(data)


    def get_synthetic_returns(self,synthetic_data,group_size=20):
        '''
        :type synthetic_data: dict
        :param synthetic_data: keys = tickers, values = tuple(data,index,polynomial fit); data format is np array paths,days

        :type group_size: int
        :param group_size: number of days to get returns over
        '''
        returns = []
        for x in self.tickers:
            mdata = synthetic_data[x][:,::group_size]    #***len(synthetic_data)-1 must be evenly divisible by group_size***
            mreturns = np.diff(mdata,axis=1)/mdata[:,:-1] #pct_change
            returns.append(mreturns)

        sreturns = []
        for i in range(12):
            temp = []
            for j in range(len(returns)):
                temp.append(returns[j][i])
            sreturns.append(np.column_stack(temp).astype(np.float32))

        return(sreturns)      


    def get_portfolios(self,n_paths=100):
        '''
        :type npaths: int
        :param npaths: number of paths to create for synthetic data
        '''
        #Setup Data
        sdata = self.get_synthetic_data(n_paths)
        sreturns = self.get_synthetic_returns(sdata)
        sreturns = [torch.tensor(x) for x in sreturns]
        
        #Create random portfolios
        ports = torch.rand(int(2e8),len(self.tickers),device=0)
        ports/=ports.sum(axis=1).reshape(-1,1)
        ports = ports.cpu()

        mx,mn = ports.max(dim=1)[0],ports.min(dim=1)[0]
        ind = torch.where((mx<.6)&(mn>.05))[0][:int(1e8)]
        
        #Get Best Portfolios
        best_portfolios = []
        for i in range(len(sreturns)):
            best_inds = []
            p2 = ports.cuda()
            sr2 = sreturns[i].cuda()

            preturns = torch.empty(100,ports.shape[0],dtype=torch.float16)
            for j in range(n_paths):
                preturns[j] = torch.einsum('ij,j->i',ports,sr2[j]).cpu()   #portfolio returns

            #Clear GPU memory
            del p2,sr2

            #Sort preturns and take top 100 portfolios per path
            nchunks = int(np.ceil(n_paths/3.))
            preturns = torch.chunk(preturns,nchunks,dim=0)
            for j in range(nchunks):
                temp = torch.argsort(preturns[j].cuda(),dim=1).cpu()
                if j<nchunks-1:
                    best = temp[:,-100:].flatten()
                else:
                    best = temp[-100:]

                best_inds.append(best)
                del temp

            best_inds = torch.cat(best_inds).reshape(-1,1)
            best_ports = ports[best_inds].numpy()
            best_portfolios.append(best_ports)

        return(best_portfolios)





#----------------------------------------------------------------------------------------------------------------------------------------
#Static Methods
def test_script():
    tickers = ['SPY','IWM','EEM','GLD']
    p1 = Portfolio(tickers)
    best = p1.get_portfolios()
    return(best)




            
        










    





