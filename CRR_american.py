import math
import numpy as np
import matplotlib.pyplot as plt

# CRR_american_option_value
def CRR_american_option_value(S0, K, T, r,sigma, q, type, N):
    #make a Binary tree
    dt = T / N  #define time interval
    u = math.exp(sigma * math.sqrt(dt)) # up movement
    d = 1 / u  # down movement
    p = (math.exp((r-q) * dt) - d) / (u - d)  # branch probability for up
    disc = math.exp(-r * dt)  # discount per interval
    
    # initiate the power matrix
    mu = np.arange(N + 1)
    mu = np.resize(mu, (N + 1, N + 1))
    md = np.transpose(mu)
    mus = u ** (mu - md)
    mds = d ** md
    
    S = S0 * mus * mds  # stock price in each node
    exs = S0 * (1/disc) ** mu # the expected price of stock in each node

    # the value of option values
    if type == 'call':
        V = np.maximum(S - K, 0)     
        oreturn = exs - K  #if the option is carried out before the maturity
    else:
        V = np.maximum(K - S, 0)       
        oreturn = K - exs

    for z in range(0, N):  # backwards iteration
        ovalue = (p * V[0:N - z, N - z] +(1 - p) * V[1:N - z + 1, N - z]) * disc
        #option price takes the maximum value of late discount and early exercise
        V[0:N - z, N - z - 1] = np.maximum(ovalue, oreturn[0:N - z, N - z - 1])
        
    return V[0,0]

def CRR_american_option_value2(S0, K, T, r, sigma, q, type, N):
    
    v = CRR_american_option_value(S0, K, T, r, sigma, q, type, N)
    
    v1 = CRR_american_option_value(S0*0.99, K, T, r, sigma, q, type, N)
    v2 = CRR_american_option_value(S0*1.01, K, T, r, sigma, q, type, N)
    delta = (v2 - v1)/(0.02*S0)
    
    v1 = CRR_american_option_value(S0, K, T, r, sigma, q, type, N)
    v2 = CRR_american_option_value(S0*1.05, K, T, r, sigma, q, type, N)
    v3 = CRR_american_option_value(S0*1.1, K, T, r, sigma, q, type, N)
    delta1 = (v2 - v1)/(0.05*S0)
    delta2 = (v3 - v2)/(0.05*S0)
    gamma = (delta2 - delta1)/(0.5*0.1*S0)
    
    v1 = CRR_american_option_value(S0, K, T, r, sigma, q, type, N)
    v2 = CRR_american_option_value(S0, K, T*0.98, r, sigma, q, type, N)
    theta = (v2 - v1)/(0.02*T)
    
    v1 = CRR_american_option_value(S0, K, T, r, sigma*0.99, q, type, N)
    v2 = CRR_american_option_value(S0, K, T, r, sigma*1.01, q, type, N)
    vega = (v2 - v1)/(0.02*sigma)
    
    v1 = CRR_american_option_value(S0, K, T, r*0.99, sigma, q, type, N)
    v2 = CRR_american_option_value(S0, K, T, r*1.01, sigma, q, type, N)
    rho = (v2 - v1)/(0.02*r)
    
    return v,delta,gamma,theta,vega,rho
    
v,delta,gamma,theta,vega,rho = CRR_american_option_value2(100, 100, 1, 0.06, 0.2, 0.03, "call", 100)
print(v,delta,gamma,theta,vega,rho)
