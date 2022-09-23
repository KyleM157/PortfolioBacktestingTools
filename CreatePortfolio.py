'''
Portfolio Selection
'''

#Startup Code Begin
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import torch
import itertools
from scipy.interpolate import CubicSpline
from datetime import datetime,timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from timeit import default_timer as timer
#Startup Code End

class PlotSelection(object):
    '''
    Used for picking point at either side of market shock event
    '''
    def __init__(self):
        self.points = []
        
    def event_handler(self,event):
        pt = np.round(event.xdata).astype(np.int32)
        self.points.append(pt)
        
    def get_points(self,data):
        xr = np.arange(len(data))
        fig,ax = plt.subplots()
        cid = fig.canvas.mpl_connect('button_press_event',self.event_handler)
        ax.plot(xr,data)
        plt.show()
        
        
class Portfolio(object):
    def __init__(self,tickers):
        self.tickers = tickers

    def read_data(self,nweeks):
        #Get ndays
        if nweeks%2==0:
            ndays=int(nweeks*5+1)
        else:
            ndays=int(nweeks*5)
            
        end_date = datetime.today()-timedelta(days=1)
        start_date = (end_date-timedelta(days=nweeks*8)).strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')


        data = []
        for i in range(len(self.tickers)):
            df = web.DataReader(self.tickers[i],'yahoo',start_date,end_date)[-ndays:]
            data.append(df)

        return(data)

    def create_synthetic_data(self,nweeks,npaths,calc_returns=False,save_data=False):
        '''
        :type data: dict of ndarrays
        :param data: close prices for each ticker in class

        :type npaths: int
        :param npaths: number of paths to create

        Returns sdata: list of data by ticker of shape npaths,nweeks*5
        '''
        
        #Get ndays
        end_date = datetime.today()-timedelta(days=1)
        start_date = (end_date-timedelta(days=nweeks*8)).strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')

        if nweeks%2==0:
            ndays=int(nweeks*5+1)
        else:
            ndays=int(nweeks*5)
        obs = []
        sdata = []
        for i in range(len(self.tickers)):
            df = web.DataReader(self.tickers[i],'yahoo',start_date,end_date)
            c = df.Close.values[-ndays:]
            obs.append(c)
            lc = np.log(c)
            lr = np.diff(lc)
            
            #Pick xpoints either side of market crash
            ps = PlotSelection()
            ps.get_points(lc)
            if len(ps.points)!=0:
                xr = np.arange(ps.points[0],ps.points[1])
                y = lc[ps.points[0]:ps.points[1]]
                yy = CubicSpline(xr,y)(xr)
                dy1 = np.diff(yy)
                
                dy0 = np.full(ps.points[0],lr.mean())
                dy2 = np.full(len(lc)-ps.points[1],lr.mean())
                
                dy = np.concatenate((dy0,dy1,dy2))
                
            else:
                dy = np.full(len(lr),lr.mean())

            #Get FFT
            r = abs(np.fft.rfft(lr))
            r[0]=0.
            phi = np.random.uniform(-np.pi,np.pi,size=(npaths,len(r)))
            z = r*np.exp(phi*1j)
            slr = np.fft.irfft(z)+dy

            #Convert to Prices
            return_matrix = np.hstack((np.ones((npaths,1)),np.exp(slr)))
            prices = c[0]*return_matrix.cumprod(axis=1) #shape=npaths,ndays

            sdata.append(prices)

        if calc_returns==True:
            sreturns = []
            for i in range(npaths):
                temp = np.vstack([x[i] for x in sdata])
                pc = np.diff(temp,axis=1)/temp[:,:-1]
                sreturns.append(pc.T)
            return(obs,sdata,sreturns)

        else:
            return(obs,sdata)


    def get_port_performance(self,Rf,n_weeks,n_paths,save_data=False):
        '''
        :type Rf: float
        :param Rf: Risk-Free rate; normally 10 year treasury yield

        :type n_weeks: int
        :param n_weeks: number of weeks to calculate sharpe ratio over per path of synthetic data

        :type n_paths: int
        :param n_paths: number of paths of synthetic data to create

        Returns the portfolio with the best sharpe ratio for each path over the lookback period.
        '''
        obs,sdata,sreturns = self.create_synthetic_data(n_weeks,n_paths,calc_returns=True)
        Rf = Rf/252.

        #Create Permutations of Portfolios
        a = np.arange(10,52,2).tolist()
        ports = np.array(list(itertools.permutations(a,r=len(self.tickers)+1)))
        ports = ports[np.where(ports.sum(axis=1)==100)[0]]/100.
        ports = ports.astype(np.float32)
        ports = np.vstack((np.full(shape=(3,),fill_value=1./len(self.tickers)+1,dtype=np.float32),ports))

        #Get Portfolio Returns and Calculate Sharpe Ratios for Each Path
        sharpe_ratios = []
        for i in range(n_paths):
            rr = np.hstack((sreturns[i],np.zeros((sreturns[i].shape[0],1),dtype=np.float32))) #shape = ndays,ntickers
            preturns = np.empty(shape=(rr.shape[0],len(ports)),dtype=np.float32)
            for j in range(rr.shape[0]):
                preturns[j] = np.log(1+(np.einsum('ij,j->i',ports,rr[j])-Rf)) #log returns adjusted by risk-free rate

            preturns = np.split(preturns,n_weeks,axis=0)

            #Calculate Sharpe Ratio for Weekly Returns
            sr = np.empty(shape=(n_weeks,len(ports)))
            for j in range(n_weeks):
                rr = preturns[j]    
                mu = rr.mean(axis=0)
                sigma = rr.std(axis=0)
                sr[j] = mu/sigma

            sharpe_ratios.append(sr)

        sharpe_ratios = np.array(sharpe_ratios)
            
        if save_data==True:
            tday = datetime.today().strftime('%Y%m%d')
            fname = r'E:/Investing_Data/Portfolios/{}'
            np.save(fname.format('sratios_'+tday),sharpe_ratios)
            np.save(fname.format('obs_'+tday),np.asarray(obs))
            np.save(fname.format('sdata_'+tday),np.asarray(sdata))
            np.save(fname.format('sreturns_'+tday),np.asarray(sreturns))
            np.save(fname.format('ports_'+tday),np.asarray(ports))

            
        else:
            pass

        return(sharpe_ratios,obs,sreturns,sdata,ports)  
            

#-----------------------------------------------------------------------------
#Static Methods

def load_port(date):
    fname = r'E:/Investing_Data/Portfolios/{}.npy'
    sharpe_ratios = np.load(fname.format('sratios_'+date))
    obs = np.load(fname.format('obs_'+date))
    sdata = np.load(fname.format('sdata_'+date))
    sreturns = np.load(fname.format('sreturns_'+date))
    ports = np.load(fname.format('ports_'+date))
    
    return(sharpe_ratios,obs,sdata,sreturns)


def get_best(sharpe_ratios,ports):
    '''
    '''
    ports = ports[1:]
    benchmark = sharpe_ratios[:,:,0].flatten()
    sharpe_ratios = sharpe_ratios[:,:,1:]

    npaths = sharpe_ratios.shape[0]
    nweeks = sharpe_ratios.shape[1]
    nports = sharpe_ratios.shape[2]

    w = np.ones(shape=(nweeks,nports),dtype=np.float32)

    bs = 500    #batch size
    for i in range(n_iterations):
        idx = np.random.randint(0,npaths,size=bs)
        for j in range(nweeks):
            batch = sharpe_ratios[idx][:,j] #shape npaths,nports
            for k in range(batch.shape[0]):
                ind1 = np.where(batch>benchmark[j])[0]
                ind2 = np.where(batch<=benchmark[j])[0]

                if j==w.shape[1]-1:
                    w[j][ind1]+=1
                    w[j][ind2]*=.75
                else:
                    w[j][ind1]+=1
                    w[j+1][ind1]+=1
                    w[j][ind2]*=.75

    idx = np.argmax(w,axis=1)
    best_ports = ports[idx]

    return(best_ports)





def get_features(best_ports,sreturns,sdata,ma_window=4):
    '''
    :type best_ports: ndarray
    :param best_ports: ndarray of ndarrays of shape nweeks,ntickers with portfolio weights

    :type sreturns: ndarray
    :param sreturns: ndarray of price matrices for each ticker; shape = ndays,ntickers

    :type sdata: ndarray
    :param sdata: synthetic data; len=ntickers,shape=npaths,ndays

    :type ma_window: int
    :param ma_window: window to calculate moving average of portfolio weights; default = 4 weeks.
    '''
    ntickers = len(sdata)
    nweeks = int(sreturns[0].shape[0]/5.)
    npaths = len(sreturns)
    
    #Get Volatility and Moving Averages for Each Path
    nweeks = int(sreturns[0].shape[0]/5)
    features = []
    for i in range(ntickers):
        rr = pd.DataFrame(x[:,0] for x in sreturns)
        lr = np.log(1+rr)

        v = (2*lr.ewm(span=2).std()-lr.ewm(span=4).std()).ewm(span=2).std()
        ema = lr.ewm(span=4).mean()
        hma = (2*lr.ewm(span=2).mean()-lr.ewm(span=4).mean()).ewm(span=2).mean()
        eh_ratio = ema/hma

        #Get ratio of Close to EMA/HMA
        c = pd.DataFrame(sdata[i].T)
        if len(c)%5==0:
            pass
        else:
            c=c[1:]

        c_ema = c.ewm(span=5).mean()
        c_hma = (2*c.ewm(span=5).mean()-c.ewm(span=10).mean()).ewm(span=3).mean()
        ce_ratio = c/c_ema
        ch_ratio = c/c_hma

        flist = [v,ema,hma,eh_ratio,ce_ratio,ch_ratio]
        ff = np.empty(shape=(len(flist),nweeks,npaths))
        for j in range(len(flist)):
            temp = flist[j]
            for k in range(ma_window,nweeks):
                if j<4:
                    n1 = k*4
                    n2 = n1+4
                else:
                    n1 = k*5
                    n2 = n1+5
                ff[j][k] = temp[n1:n2].mean(axis=0).values

        features.append(ff.T)
    features = np.concatenate(features,axis=2) #shape=npaths,nweeks,nfeatures*ntickers                
    
    #Use Moving Average to get Labels
    labels = np.empty(shape=(ntickers,nweeks-ma_window,npaths))
    for i in range(ntickers):
        df = pd.DataFrame([x[:,i] for x in best_ports]) #shape nweeks,npaths
        ma = df.rolling(window=ma_window).mean(axis=0).dropna().values
        labels[i] = ma

    labels = labels.T

    return(features,labels)

def get_train_test(features,labels):
    '''
    '''
    n = int(features[0].shape[0]*.7)

    train_x = [x[:n] for x in features]
    train_y = [x[:n] for x in labels]
    test_x = [x[n:] for x in features]
    test_y = [x[n:] for x in labels]      

    train = (train_x,train_y)
    test = (test_x,test_y)

    return(train,test)

def fit_regressor(features,labels):
    '''
    '''
    rf = RandomForestRegressor(n_estimators=10,warm_start=True)
    for i in range(features.shape[0]):
        rf.fit(features[i],labels[i])
        rf.n_estimators+=10

    return(rf)



def get_interpolated_features(best_ports,sreturns,sdata):
    '''
    '''
    ntickers = len(sdata)
    nweeks = int(sreturns[0].shape[0]/5.)
    npaths = len(sreturns)
    
    #Interpolate best_ports
    xr = np.arange(len(best_ports))
    xr2 = np.arange(nweeks*4)
    
    for i in range(ntickers):
        temp = pd.DataFrame([x[:,i] for x in best_ports])
        daily_ports = temp.apply(lambda y: CubicSpline(xr,y)(xr2))
    
        
        
        
    

def plot_ports(best_ports,show_plot=True):
    '''
    '''
    nweeks = best_ports.shape[0]
    ndays = int(nweeks*5)

    xr = np.arange(nweeks)
    xrr = np.linspace(0,nweeks,ndays)

    df = pd.DataFrame(best_ports).apply(lambda y: CubicSpline(xr,y)(xrr))
    h = (2*df.ewm(span=10).mean()-df.ewm(span=20).mean()).ewm(span=4).mean()

    if show_plot==True:
        h.plot(marker='o')
        plt.show()
    else:
        pass

    return(df,h)


def test_script():
    start = timer()
    tickers = ['SPY','IWM','EEM']
    P = Portfolio(tickers)
    bp = P.get_port_performance(.009,52,1000,save_data=True)
    end = timer()
    print('Wall Time: {}'.format(str(timedelta(seconds=end-start))))
    return(bp)
    













            


