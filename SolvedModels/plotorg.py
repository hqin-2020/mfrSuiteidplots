import mfr.modelSoln as m
import numpy as np
import argparse
import time
import os
import json
import pickle
import itertools

parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--nV",type=int,default=30)
parser.add_argument("--nVtilde",type=int,default=0)
parser.add_argument("--V_bar",type=float,default=1.0)
parser.add_argument("--Vtilde_bar",type=float,default=0.0)
parser.add_argument("--sigma_V_norm",type=float,default=0.132)
parser.add_argument("--sigma_Vtilde_norm",type=float,default=0.0)

parser.add_argument("--a_e",type=float,default=0.14)
parser.add_argument("--a_h",type=float,default=0.135)
parser.add_argument("--psi_e",type=float,default=1.0)
parser.add_argument("--psi_h",type=float,default=1.0)
parser.add_argument("--gamma_e",type=float,default=1.0)
parser.add_argument("--gamma_h",type=float,default=1.0)
parser.add_argument("--chiUnderline",type=float,default=1.0)
args = parser.parse_args()

params = m.paramsDefault.copy()

## Dimensionality params
params['nDims']             = 3
params['nShocks']           = 3

## Grid parameters 
params['numSds']            = 5
params['uselogW']           = 0

params['nWealth']           = 100
params['nZ']                = 30
params['nV']                = args.nV
params['nVtilde']           = args.nVtilde


## Economic params
params['nu_newborn']        = 0.1
params['lambda_d']          = 0.02
params['lambda_Z']          = 0.252
params['lambda_V']          = 0.156
params['lambda_Vtilde']     = 1.38
params['delta_e']           = 0.05
params['delta_h']           = 0.05
params['a_e']               = args.a_e
params['a_h']               = args.a_h
params['rho_e']             = args.psi_e
params['rho_h']             = args.psi_h
params['phi']               = 3.0
params['gamma_e']           = args.gamma_e
params['gamma_h']           = args.gamma_h
params['equityIss']         = 2
params['chiUnderline']      = args.chiUnderline
params['alpha_K']           = 0.05

## Alogirthm behavior and results savings params
params['method']            = 2
params['dt']                = 0.1
params['dtInner']           = 0.1

params['tol']               = 1e-5
params['innerTol']          = 1e-5

params['verbatim']          = -1
params['maxIters']          = 4000
params['maxItersInner']     = 2000000
params['iparm_2']           = 28
params['iparm_3']           = 0
params['iparm_28']          = 0
params['iparm_31']          = 0
params['overwrite']         = 'True'
params['exportFreq']        = 10000
params['CGscale']           = 1.0
params['hhCap']             = 1

# Domain params
params['Vtilde_bar']        = args.Vtilde_bar
params['Z_bar']             = 0.0
params['V_bar']             = args.V_bar
params['sigma_K_norm']      = 0.04
params['sigma_Z_norm']      = 0.0141
params['sigma_V_norm']      = args.sigma_V_norm
params['sigma_Vtilde_norm'] = args.sigma_Vtilde_norm
params['wMin']              = 0.01
params['wMax']              = 0.99

## Shock correlation params
params['cov11']             = 1.0
params['cov12']             = 0.0
params['cov13']             = 0.0
params['cov14']             = 0.0
params['cov21']             = 0.0
params['cov22']             = 1.0
params['cov23']             = 0.0
params['cov24']             = 0.0
params['cov31']             = 0.0
params['cov32']             = 0.0
params['cov33']             = 1.0
params['cov34']             = 0.0
params['cov41']             = 0.0
params['cov42']             = 0.0
params['cov43']             = 0.0
params['cov44']             = 0.0

psi_e = str("{:0.3f}".format(params['rho_e'])).replace('.', '', 1) 
psi_h = str("{:0.3f}".format(params['rho_h'])).replace('.', '', 1) 
gamma_e = str("{:0.3f}".format(params['gamma_e'])).replace('.', '', 1) 
gamma_h = str("{:0.3f}".format(params['gamma_h'])).replace('.', '', 1) 
a_e = str("{:0.3f}".format(params['a_e'])).replace('.', '', 1) 
a_h = str("{:0.3f}".format(params['a_h'])).replace('.', '', 1) 
chiUnderline = str("{:0.3f}".format(params['chiUnderline'])).replace('.', '', 1) 

folder_name = 'chiUnderline_' + chiUnderline + '_a_e_' + a_e + '_a_h_' + a_h  + '_gamma_e_' + gamma_e + '_gamma_h_' + gamma_h + '_psi_e_' + psi_e + '_psi_h_' + psi_h

params['folderName']        = folder_name
params['preLoad']           = folder_name

#### Now, create a Model
Model = m.Model(params)

ExpertsExpoConsumption = pickle.load(open(os.getcwd()+"/" + folder_name + "/ExpertsExpoConsumption.pkl", "rb"))
ExpertsPriceConsumption = pickle.load(open(os.getcwd()+"/" + folder_name + "/ExpertsPriceConsumption.pkl", "rb"))

HouseholdsExpoConsumption = pickle.load(open(os.getcwd()+"/" + folder_name + "/HouseholdsExpoConsumption.pkl", "rb"))
HouseholdsPriceConsumption = pickle.load(open(os.getcwd()+"/" + folder_name + "/HouseholdsPriceConsumption.pkl", "rb"))

os.makedirs(os.getcwd()+"/plots/" + folder_name, exist_ok = True)
plotdir = os.getcwd()+"/plots/" + folder_name

np.savetxt(plotdir + '/ExpertsExpoConsumption.txt', np.mean(ExpertsExpoConsumption.firstType[0,0,:]))
np.savetxt(plotdir + '/ExpertsPriceConsumption.txt', np.mean(ExpertsPriceConsumption.firstType[0,0,:]))
np.savetxt(plotdir + '/HouseholdsExpoConsumption.txt', np.mean(HouseholdsExpoConsumption.firstType[0,0,:]))
np.savetxt(plotdir + '/HouseholdsPriceConsumption.txt', np.mean(HouseholdsPriceConsumption.firstType[0,0,:]))


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.options.display.float_format = '{:.3g}'.format
sns.set(font_scale = 1.5)
T = 48
# Calculate the shock elasticity at 0.25, 0.5 and 0.75 quantile for W
quantile = [0.25, 0.5, 0.75]

## Plot the exposure elasticity for consumption growth
index = ['T','Aggregate Volatility 0.25 quantile','Aggregate Volatility 0.5 quantile','Aggregate Volatility 0.75 quantile']
fig, axes = plt.subplots(1,3, figsize = (25,8))
expo_elas_shock_0 = pd.DataFrame([np.arange(T),ExpertsExpoConsumption.firstType[0,0,:],ExpertsExpoConsumption.firstType[1,0,:],ExpertsExpoConsumption.firstType[2,0,:]], index = index).T
expo_elas_shock_1 = pd.DataFrame([np.arange(T),ExpertsExpoConsumption.firstType[0,1,:],ExpertsExpoConsumption.firstType[1,1,:],ExpertsExpoConsumption.firstType[2,1,:]], index = index).T
expo_elas_shock_2 = pd.DataFrame([np.arange(T),ExpertsExpoConsumption.firstType[0,2,:],ExpertsExpoConsumption.firstType[1,2,:],ExpertsExpoConsumption.firstType[2,2,:]], index = index).T

n_qt = len(quantile)
plot_elas = [expo_elas_shock_0, expo_elas_shock_1, expo_elas_shock_2]
shock_name = ['TFP shock', 'growth rate shock', 'aggregate stochastic volitility shock']
qt = ['Aggregate Volatility 0.25 quantile','Aggregate Volatility 0.5 quantile','Aggregate Volatility 0.75 quantile']
colors = ['green','red','blue']

for i in range(len(plot_elas)):
    for j in range(n_qt):
        sns.lineplot(data = plot_elas[i],  x = 'T', y = qt[j], ax=axes[i], color = colors[j], label = qt[j])
        axes[i].set_xlabel('Years')
        axes[i].set_ylabel('Exposure elasticity')
        axes[i].set_title('With respect to the ' + shock_name[i])
axes[0].set_ylim([-0.05,0.1])
axes[1].set_ylim([-0.05,0.1])
axes[2].set_ylim([-0.01,0.01])
fig.suptitle('Exposure elasticity for the Experts Consumption')
fig.tight_layout()
fig.savefig(plotdir + '/expoElasExperts_C_type1org.png')
plt.close()

## Plot the exposure elasticity for consumption growth
index = ['T','Aggregate Volatility 0.25 quantile','Aggregate Volatility 0.5 quantile','Aggregate Volatility 0.75 quantile']
fig, axes = plt.subplots(1,3, figsize = (25,8))
expo_elas_shock_0 = pd.DataFrame([np.arange(T),HouseholdsExpoConsumption.firstType[0,0,:],HouseholdsExpoConsumption.firstType[1,0,:],HouseholdsExpoConsumption.firstType[2,0,:]], index = index).T
expo_elas_shock_1 = pd.DataFrame([np.arange(T),HouseholdsExpoConsumption.firstType[0,1,:],HouseholdsExpoConsumption.firstType[1,1,:],HouseholdsExpoConsumption.firstType[2,1,:]], index = index).T
expo_elas_shock_2 = pd.DataFrame([np.arange(T),HouseholdsExpoConsumption.firstType[0,2,:],HouseholdsExpoConsumption.firstType[1,2,:],HouseholdsExpoConsumption.firstType[2,2,:]], index = index).T

n_qt = len(quantile)
plot_elas = [expo_elas_shock_0, expo_elas_shock_1, expo_elas_shock_2]
shock_name = ['TFP shock', 'growth rate shock', 'aggregate stochastic volitility shock']
qt = ['Aggregate Volatility 0.25 quantile','Aggregate Volatility 0.5 quantile','Aggregate Volatility 0.75 quantile']
colors = ['green','red','blue']

for i in range(len(plot_elas)):
    for j in range(n_qt):
        sns.lineplot(data = plot_elas[i],  x = 'T', y = qt[j], ax=axes[i], color = colors[j], label = qt[j])
        axes[i].set_xlabel('Years')
        axes[i].set_ylabel('Exposure elasticity')
        axes[i].set_title('With respect to the ' + shock_name[i])
axes[0].set_ylim([-0.001,0.1])
axes[1].set_ylim([-0.005,0.1])
axes[2].set_ylim([-0.01,0.01])
fig.suptitle('Exposure elasticity for the Households Consumption')
fig.tight_layout()
fig.savefig(plotdir + '/expoElasHouseholdss_C_type1org.png')
plt.close()

index = ['T','Aggregate Volatility 0.25 quantile','Aggregate Volatility 0.5 quantile','Aggregate Volatility 0.75 quantile']
fig, axes = plt.subplots(1,3, figsize = (25,8))
expo_elas_shock_0 = pd.DataFrame([np.arange(T),ExpertsPriceConsumption.firstType[0,0,:],ExpertsPriceConsumption.firstType[1,0,:],ExpertsPriceConsumption.firstType[2,0,:]], index = index).T
expo_elas_shock_1 = pd.DataFrame([np.arange(T),ExpertsPriceConsumption.firstType[0,1,:],ExpertsPriceConsumption.firstType[1,1,:],ExpertsPriceConsumption.firstType[2,1,:]], index = index).T
expo_elas_shock_2 = pd.DataFrame([np.arange(T),-ExpertsPriceConsumption.firstType[0,2,:],-ExpertsPriceConsumption.firstType[1,2,:],-ExpertsPriceConsumption.firstType[2,2,:]], index = index).T

n_qt = len(quantile)
plot_elas = [expo_elas_shock_0, expo_elas_shock_1, expo_elas_shock_2]
shock_name = ['TFP shock', 'growth rate shock', 'aggregate stochastic volitility shock']
qt = ['Aggregate Volatility 0.25 quantile','Aggregate Volatility 0.5 quantile','Aggregate Volatility 0.75 quantile']
colors = ['green','red','blue']

for i in range(len(plot_elas)):
    for j in range(n_qt):
        sns.lineplot(data = plot_elas[i],  x = 'T', y = qt[j], ax=axes[i], color = colors[j], label = qt[j])
        axes[i].set_xlabel('Years')
        axes[i].set_ylabel('Price elasticity')
        axes[i].set_title('With respect to the ' + shock_name[i])
axes[0].set_ylim([-0.05,0.4])
axes[1].set_ylim([-0.05,0.4])
axes[2].set_ylim([-0.05,0.05])
fig.suptitle('Price elasticity for the Experts Consumption')
fig.tight_layout()
fig.savefig(plotdir + '/priceElasExperts_C_type1org.png')
plt.close()

index = ['T','Aggregate Volatility 0.25 quantile','Aggregate Volatility 0.5 quantile','Aggregate Volatility 0.75 quantile']
fig, axes = plt.subplots(1,3, figsize = (25,8))
expo_elas_shock_0 = pd.DataFrame([np.arange(T),HouseholdsPriceConsumption.firstType[0,0,:],HouseholdsPriceConsumption.firstType[1,0,:],HouseholdsPriceConsumption.firstType[2,0,:]], index = index).T
expo_elas_shock_1 = pd.DataFrame([np.arange(T),HouseholdsPriceConsumption.firstType[0,1,:],HouseholdsPriceConsumption.firstType[1,1,:],HouseholdsPriceConsumption.firstType[2,1,:]], index = index).T
expo_elas_shock_2 = pd.DataFrame([np.arange(T),-HouseholdsPriceConsumption.firstType[0,2,:],-HouseholdsPriceConsumption.firstType[1,2,:],-HouseholdsPriceConsumption.firstType[2,2,:]], index = index).T

n_qt = len(quantile)
plot_elas = [expo_elas_shock_0, expo_elas_shock_1, expo_elas_shock_2]
shock_name = ['TFP shock', 'growth rate shock', 'aggregate stochastic volitility shock']
qt = ['Aggregate Volatility 0.25 quantile','Aggregate Volatility 0.5 quantile','Aggregate Volatility 0.75 quantile']
colors = ['green','red','blue']

for i in range(len(plot_elas)):
    for j in range(n_qt):
        sns.lineplot(data = plot_elas[i],  x = 'T', y = qt[j], ax=axes[i], color = colors[j], label = qt[j])
        axes[i].set_xlabel('Years')
        axes[i].set_ylabel('Price elasticity')
        axes[i].set_title('With respect to the ' + shock_name[i])
axes[0].set_ylim([-0.005,0.5])
axes[1].set_ylim([-0.005,0.5])
axes[2].set_ylim([-0.005,0.05])
fig.suptitle('Price elasticity for the Households Consumption')
fig.tight_layout()
fig.savefig(plotdir + '/priceElasHouseholds_C_type1org.png')
plt.close()
