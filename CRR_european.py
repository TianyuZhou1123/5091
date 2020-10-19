import math
import numpy as np
import matplotlib.pyplot as plt
def CRR_european_option_value(S0, K, T, r, sigma, q, type, N): #calculate the V[0,0] for r and sigma


    ''' Cox-Ross-Rubinstein European option valuation.
    Parameters
    ==========
    S0 : float
        stock/index level at time 0
    K : float
        strike price
    T : float
        date of maturity
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    q : float
        dividend
    otype : string
        either 'call' or 'put'
    N : int
        number of time intervals

    '''

    # make a Binary tree
    dt = T / N  # define time interval
    disc = math.exp(-r * dt)  # discount per interval
    u = math.exp(sigma * math.sqrt(dt)) #up movement
    d = 1 / u  # down movement
    p = (math.exp((r-q) * dt) - d) / (u - d)  #  branch probability of up

    # initiate the power matrix
    mu = np.arange(N + 1)
    mu = np.resize(mu, (N + 1, N + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md

    #stock price of each node
    S = S0 * mu * md

    #option value of each node
    if type == 'call':
        V = np.maximum(S - K, 0)  # inner values for European call option
    else:
        V = np.maximum(K - S, 0) 

    for z in range(0, N):  
        # backwards iteration
        V[0:N - z, N - z - 1] = (p * V[0:N - z, N - z] +(1 - p) * V[1:N - z + 1, N - z]) * disc 

    return V[0,0]

def CRR_european_option_value2(S0, K, T, r, sigma, q, type, N):
    
    v = CRR_european_option_value(S0, K, T, r, sigma, q, type, N)
    
    v1 = CRR_european_option_value(S0*0.99, K, T, r, sigma, q, type, N)  #dt=0.01*S0
    v2 = CRR_european_option_value(S0*1.01, K, T, r, sigma, q, type, N)
    delta = (v2 - v1)/(0.02*S0)
    
    v1 = CRR_european_option_value(S0, K, T, r, sigma, q, type, N)
    v2 = CRR_european_option_value(S0*1.05, K, T, r, sigma, q, type, N)
    v3 = CRR_european_option_value(S0*1.1, K, T, r, sigma, q, type, N)
    delta1 = (v2 - v1)/(0.05*S0)
    delta2 = (v3 - v2)/(0.05*S0)
    gamma = (delta2 - delta1)/(0.5*0.1*S0)
    
    v1 = CRR_european_option_value(S0, K, T, r, sigma, q, type, N)
    v2 = CRR_european_option_value(S0, K, T*0.98, r, sigma, q, type, N)
    theta = (v2 - v1)/(0.02*T)
    
    v1 = CRR_european_option_value(S0, K, T, r, sigma*0.99, q, type, N)
    v2 = CRR_european_option_value(S0, K, T, r, sigma*1.01, q, type, N)
    vega = (v2 - v1)/(0.02*sigma)
    
    v1 = CRR_european_option_value(S0, K, T, r*0.99, sigma, q, type, N)
    v2 = CRR_european_option_value(S0, K, T, r*1.01, sigma, q, type, N)
    rho = (v2 - v1)/(0.02*r)
    
    return v,delta,gamma,theta,vega,rho
