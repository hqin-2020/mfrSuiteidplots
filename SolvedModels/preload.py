import mfr.modelSoln as m
import numpy as np
import argparse
import time
import os
import json
import pickle
import itertools
from scipy.interpolate import RegularGridInterpolator
from shockElasm import computeElas

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

Model.solve()
Model.computeStatDent()
Model.dumpData()

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


pcts = {'W':[.5],'Z':[.5],'V':[.25,.5,.75]}

# 30 year time periods
T = 360
dt = 1/12

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

with open(os.getcwd()+'/' + folder_name + '/model_ela_sol.pkl', 'wb') as file:
    pickle.dump(modelsol,file)

muXs = [RegularGridInterpolator(modelsol['stateMatInput'], modelsol['muX'][:,n].reshape(modelsol['gridSizeList'], order = 'F')) for n in range(modelsol['nDims'])]
muCe = RegularGridInterpolator(modelsol['stateMatInput'], modelsol['muCe'].reshape(modelsol['gridSizeList'], order = 'F'))
muCh = RegularGridInterpolator(modelsol['stateMatInput'], modelsol['muCh'].reshape(modelsol['gridSizeList'], order = 'F'))
muSe = RegularGridInterpolator(modelsol['stateMatInput'], modelsol['muSe'].reshape(modelsol['gridSizeList'], order = 'F'))
muSh = RegularGridInterpolator(modelsol['stateMatInput'], modelsol['muSh'].reshape(modelsol['gridSizeList'], order = 'F'))
muNe = RegularGridInterpolator(modelsol['stateMatInput'], modelsol['muNe'].reshape(modelsol['gridSizeList'], order = 'F'))
muNh = RegularGridInterpolator(modelsol['stateMatInput'], modelsol['muNh'].reshape(modelsol['gridSizeList'], order = 'F'))
sigmaX = [[RegularGridInterpolator(modelsol['stateMatInput'], modelsol['sigmaX'][n][:,s].reshape(modelsol['gridSizeList'], order = 'F')) for s in range(modelsol['nShocks'])] for n in range(modelsol['nDims'])]
sigmaCe = [RegularGridInterpolator(modelsol['stateMatInput'], modelsol['sigmaCe'][:,s].reshape(modelsol['gridSizeList'], order = 'F')) for s in range(modelsol['nShocks'])]
sigmaCh = [RegularGridInterpolator(modelsol['stateMatInput'], modelsol['sigmaCh'][:,s].reshape(modelsol['gridSizeList'], order = 'F')) for s in range(modelsol['nShocks'])]
sigmaSe = [RegularGridInterpolator(modelsol['stateMatInput'], modelsol['sigmaSe'][:,s].reshape(modelsol['gridSizeList'], order = 'F')) for s in range(modelsol['nShocks'])]
sigmaSh = [RegularGridInterpolator(modelsol['stateMatInput'], modelsol['sigmaSh'][:,s].reshape(modelsol['gridSizeList'], order = 'F')) for s in range(modelsol['nShocks'])]
sigmaNe = [RegularGridInterpolator(modelsol['stateMatInput'], modelsol['sigmaNe'][:,s].reshape(modelsol['gridSizeList'], order = 'F')) for s in range(modelsol['nShocks'])]
sigmaNh = [RegularGridInterpolator(modelsol['stateMatInput'], modelsol['sigmaNh'][:,s].reshape(modelsol['gridSizeList'], order = 'F')) for s in range(modelsol['nShocks'])]

muXfn = lambda x: np.transpose([mu(x) for mu in muXs])
sigmaXfn = []
for n in range(modelsol['nDims']):
    def sigmaXs(n):
            return lambda x: np.transpose([vol(x) for vol in sigmaX[n] ])
    sigmaXfn.append(sigmaXs(n))
sigmaCefn = lambda x: np.transpose([vol(x) for vol in sigmaCe])
sigmaChfn = lambda x: np.transpose([vol(x) for vol in sigmaCh])
sigmaSefn = lambda x: np.transpose([vol(x) for vol in sigmaSe])
sigmaShfn = lambda x: np.transpose([vol(x) for vol in sigmaSh])
sigmaNefn = lambda x: np.transpose([vol(x) for vol in sigmaNe])
sigmaNhfn = lambda x: np.transpose([vol(x) for vol in sigmaNh])

bc = {'natural':True}
dt = 1/12
T = 360

modelInput = {'muX':muXfn, 'sigmaX':sigmaXfn, 'muG':muCe, 'sigmaG':sigmaCefn, 'muS':muSe, 'sigmaS':sigmaSefn, 'dt':dt, 'T' :T}
expoElasExpertsC, priceElasExpertsC, _, _, costElasExpertsC, phit1ExpertsC, phit2ExpertsC = computeElas(modelsol['stateMatInput'], modelInput, bc, modelsol['x0'])

modelInput = {'muX':muXfn, 'sigmaX':sigmaXfn, 'muG':muCe, 'sigmaG':sigmaCefn, 'muS':muNe, 'sigmaS':sigmaNefn, 'dt':dt, 'T':T}
expoElasExpertsN, priceElasExpertsN, _, _, costElasExpertsN, phit1ExpertsN, phit2ExpertsN = computeElas(modelsol['stateMatInput'], modelInput, bc, modelsol['x0'])

modelInput = {'muX':muXfn, 'sigmaX':sigmaXfn, 'muG':muCh, 'sigmaG':sigmaChfn, 'muS':muSh, 'sigmaS':sigmaShfn, 'dt':dt, 'T':T}
expoElasHouseholdsC, priceElasHouseholdsC, _, _, costElasHouseholdsC, phit1HouseholdsC, phit2HouseholdsC = computeElas(modelsol['stateMatInput'], modelInput, bc, modelsol['x0'])

modelInput = {'muX':muXfn, 'sigmaX':sigmaXfn, 'muG':muCh, 'sigmaG':sigmaChfn, 'muS':muNh, 'sigmaS':sigmaNhfn, 'dt':dt, 'T':T}
expoElasHouseholdsN, priceElasHouseholdsN, _, _, costElassHouseholdsN, phit1HouseholdsN, phit2HouseholdsN = computeElas(modelsol['stateMatInput'], modelInput, bc, modelsol['x0'])

elasol = {'expoElasExpertsC':expoElasExpertsC,
        'priceElasExpertsC':priceElasExpertsC,
        'costElasExpertsC':costElasExpertsC,
        'phit1ExpertsC':phit1ExpertsC,
        'phit2ExpertsC':phit2ExpertsC,
        'expoElasExpertsN':expoElasExpertsN,
        'priceElasExpertsN':priceElasExpertsN,
        'costElasExpertsN':costElasExpertsN,
        'phit1ExpertsN':phit1ExpertsN,
        'phit2ExpertsN':phit2ExpertsN,
        'expoElasHouseholdsC':expoElasHouseholdsC,
        'priceElasHouseholdsC':priceElasHouseholdsC,
        'costElasHouseholdsC':costElasHouseholdsC,
        'phit1HouseholdsC':phit1HouseholdsC,
        'phit2HouseholdsC':phit2HouseholdsC,
        'expoElasHouseholdsN':expoElasHouseholdsN,
        'priceElasHouseholdsN':priceElasHouseholdsN,
        'costElassHouseholdsN':costElassHouseholdsN,
        'phit1HouseholdsN':phit1HouseholdsN,
        'phit2HouseholdsN':phit2HouseholdsN,
        }


with open(os.getcwd()+"/" + folder_name + "/model_ela_data.pkl", "wb") as f:
    pickle.dump(elasol,f)