import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



#BS pricing formula
def bs_pricing(s,k,r,t,sigma,option_type='call'):
    '''parameter：
    -----------------
    s:underlying price
    k:strike price
    r:risk-free rate
    t:tenor
    sigma:volitility
    option_type:call or put
    
    return：Option price (float)
    '''
    
    
    T = t #(unit:year)
    d1 = (np.log(s / k) + (r + 1 / 2 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        Call_price =lambda s,d1,k,r,T,d2: s * stats.norm.cdf(d1, 0.0, 1.0) - k * np.exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0)
        Call_delta=lambda d1: stats.norm.cdf(d1)
        Gamma=lambda d1,s,sigma,T: norm.pdf(d1,0.0,1.0)/(s*sigma*np.sqrt(T))
        Vega=lambda d1,s,T: norm.pdf(d1)*s*np.sqrt(T)
        Call_theta=lambda s,d1,sigma,T,r,k,d2: -s*norm.pdf(d1)*sigma/(2*np.sqrt(T))-r*k*np.exp(-r*T)*norm.cdf(d2)
        Call_rho=lambda k,T,r,d2: k*T*np.exp(-r*T)*norm.cdf(d2)
        print("Call_price={}\nCall_delta={}\nGamma={}\nVega={}\nCall_theta={}\nCall_rho={}\n".format(Call_price(s,d1,k,r,T,d2),Call_delta(d1),Gamma(d1,s,sigma,T),Vega(d1,s,T),Call_theta(s,d1,sigma,T,r,k,d2),Call_rho(k,T,r,d2)))
    elif option_type == 'put':
        Put_price =lambda s,k,r,T,d1,d2: k * np.exp(-r * T) * stats.norm.cdf(-1 * d2) - s * stats.norm.cdf(-1 * d1)
        Put_delta=lambda d1: stats.norm.cdf(d1)-1
        Gamma=lambda d1,s,sigma,T: norm.pdf(d1,0.0,1.0)/(s*sigma*np.sqrt(T))
        Vega=lambda d1,s,T: norm.pdf(d1)*s*np.sqrt(T)
        Put_theta=lambda s,d1,sigma,T,r,k,d2:-s*norm.pdf(d1)*sigma/(2*np.sqrt(T))+r*k*np.exp(-r*T)*norm.cdf(-d2)
        Put_rho=lambda k,T,r,d2:-k*T*np.exp(-r*T)*norm.cdf(-d2)
        print("Put_price={}\nPut_delta={}\nGamma={}\nVega={}\nPut_theta={}\nPut_rho={}\n".format(Put_price(s,d1,k,r,T,d2),Put_delta(d1),Gamma(d1,s,sigma,T),Vega(d1,s,T),Put_theta(s,d1,sigma,T,r,k,d2),Put_rho(k,T,r,d2)))
    else:
        return None
bs_pricing(49,50,0.05,0.3846,0.2,'call')
