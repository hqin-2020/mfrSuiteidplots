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
params['overwrite']         = 'Yes'
params['exportFreq']        = 10000
params['CGscale']           = 1.0
params['hhCap']             = 1
params['preLoad']           = 'None'

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

#### Now, create a Model
Model = m.Model(params)

# Step 2: Solve the model
#------------------------------------------#

#### This step is very simple: use the .solve() method.
start = time.time()
Model.solve()
Model.printInfo() ## This step is optional: it prints out information regarding time, number of iterations, etc.
Model.printParams() ## This step is optional: it prints out the parameteres used.
end = time.time()

Model.computeStatDent()

Model.computeShockElas(pcts = {'W':[.5], 'Z': [0.5], 'V': [0.25, 0.5, 0.75]}, T = 360, dt = 1, perturb = 'Ce')

with open(os.getcwd()+"/" + folder_name + "/ExpertsConsumption.json", 'wb') as file:   
    pickle.dump(Model.expoElas, file)

with open(os.getcwd()+"/" + folder_name + "/ExpertsConsumption.json", 'wb') as file:   
    pickle.dump(Model.priceElasExperts, file)

Model.computeShockElas(pcts = {'W':[.5], 'Z': [0.5], 'V': [0.25, 0.5, 0.75]}, T = 360, dt = 1, perturb = 'Ch')

with open(os.getcwd()+"/" + folder_name + "/HouseholdsConsumption.json", 'wb') as file:   
    pickle.dump(Model.expoElas, file)

with open(os.getcwd()+"/" + folder_name + "/HouseholdsConsumption.json", 'wb') as file:   
    pickle.dump(Model.priceElasHouseholds, file)

Model.dumpData()
solve_time = '{:.4f}'.format((end - start)/60)
MFR_time_info = {'solve_time': solve_time}
with open(os.getcwd()+"/" + folder_name + "/MFR_time_info.json", "w") as f:
    json.dump(MFR_time_info,f)

# %%


##### This method can only be called after the model is solved.
pcts = {'W':[.5],'Z':[.5],'V':[.25,.5,.75]}

# 30 year time periods
T = 360
dt = 1

# Natural boundatry conditions
bc = {'natural':True}

# Use defaults starting points
points = np.matrix([])

## Create input stateMat for shock elasticities, a tuple of ranges of the state space
Model.stateMatInput = []

for i in range(Model.params['nDims']):
    ## Note that i starts at zero but our variable names start at 1.
    Model.stateMatInput.append(np.linspace(np.min(Model.stateMat.iloc[:,i]),
                                np.max(Model.stateMat.iloc[:,i]),
                                np.unique(np.array(Model.stateMat.iloc[:,i])).shape[0]) )
## Create dictionary to store model
if Model.model is None:
    Model.model = {}

allPcts = []

## Find points
if points.shape[1] == 0:
    allPts = []
    for stateVar in Model.stateVarList:
        if Model.dent is None or np.max(Model.dent) < 0.0001:
            raise Exception("Stationary density not computed or degenerate.")
        allPts.append([Model.inverseCDFs[stateVar](pct) for pct in pcts[stateVar]])
        allPcts.append(pcts[stateVar])
    Model.x0 = np.matrix(list(itertools.product(*allPts)))
    allPcts = [list(x) for x in list(itertools.product(*allPcts))]
    Model.pcts = pcts
else:
    Model.x0 = points

modelsol = {
    'muCe' : Model.muCe(),
    'muCh' : Model.muCh(),
    'muSe' : Model.muSe(),
    'muSh' : Model.muSh(),
    'muX'  : Model.muX(),
    'muNh'  : -0.5*np.sum([((1-Model.params['gamma_h'])/(Model.params['rho_h']-Model.params['gamma_h'])*\
                                    (Model.sigmaSh()[:,s]+\
                                     Model.params['rho_h']*Model.sigmaCh()[:,s]))**2\
                                    for s in range(Model.params['nDims'])], axis = 0),
    'muNe'  : -0.5*np.sum([((1-Model.params['gamma_e'])/(Model.params['rho_e']-Model.params['gamma_e'])*\
                                    (Model.sigmaSe()[:,s]+\
                                     Model.params['rho_e']*Model.sigmaCe()[:,s]))**2\
                                    for s in range(Model.params['nDims'])], axis = 0),
    'sigmaSe' : Model.sigmaSe(),
    'sigmaSh' : Model.sigmaSh(),
    'sigmaCe' : Model.sigmaCe(),
    'sigmaCh' : Model.sigmaCh(),
    'sigmaX'  : Model.sigmaXList,
    'sigmaNh' : (1-Model.params['gamma_h'])/(Model.params['rho_h']-Model.params['gamma_h'])*\
                                    (Model.sigmaSh()+\
                                     Model.params['rho_h']*Model.sigmaCh()),
    'sigmaNe' : (1-Model.params['gamma_e'])/(Model.params['rho_e']-Model.params['gamma_e'])*\
                                    (Model.sigmaSe()+\
                                     Model.params['rho_e']*Model.sigmaCe()),
    'stateMatInput' : Model.stateMatInput,
    'gridSizeList' : Model.gridSizeList,
    'x0' : Model.x0,
    'nDims' : Model.params['nDims'],
    'nShocks' : Model.params['nShocks']
}

with open(os.getcwd()+"/" + folder_name + "/model_ela_sol.pkl", "wb") as f:
    pickle.dump(modelsol,f)