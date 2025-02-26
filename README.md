# PhoTorch

PhoTorch is a robust and generalized photosynthesis biochemical model fitting package based on PyTorch.
Read more about PhoTorch in our paper: https://arxiv.org/abs/2501.15484.

**Note: The latest version 1.3.0 includes changes to the file structure and function names within the package.** 

Currently, the package includes the Farquhar, von Caemmerer, and Berry (FvCB) model and stomatal conductance models including Buckley Mott Farquhar (BMF), Medlyn (MED), and Ball Woodrow Berry (BWB) versions. The Ball Berry Leuning (BBL) stomatal conductance model and the PROSPECT models are under development.
## Installation of dependencies
```bash
pip install torch
pip install numpy
pip install scipy
pip install pandas
```

**Next, download all the files and directories, and try running the examples in the testphotorch.py file!**

## 1. FvCB model usage
Create a python file in the PhoTorch directory and import necessary packages.

```bash
import fvcb
import pandas as pd
import torch
```
### Load data
Load the example CSV file. The loaded data frame should have columns with titles 'CurveID', 'FittingGroup', 'Ci', 'A', 'Qin', and 'Tleaf'. Each A/Ci curve should have a unique 'CurveID'.
If no 'Qin' and 'Tleaf' are available, it will be automatically set to 2000 and 25, respectively.

The data to be loaded should be:

| CurveID | FittingGroup | Ci  | A  | Qin  | Tleaf |
|---------|--------------|-----|----|------|-------|
| 1       | 1            | 200 | 20 | 2000 | 25    |
| 1       | 1            | 400 | 30 | 2000 | 25    |
| 1       | 1            | 600 | 40 | 2000 | 25    |
| 2       | 1            | 200 | 25 | 2000 | 30    |
| 2       | 1            | 400 | 35 | 2000 | 30    |
| 2       | 1            | 700 | 55 | 2000 | 30    |

```bash
dftest = pd.read_csv('exampledata/dfMAGIC043_lr.csv')
# remove the rows with negative A values
dftest = dftest[dftest['A'] > 0]
```
### Initialize the data
Then, specify the ID of the light response curve. If there is no light response curve in the dataset, ignore it (default is None).
```bash
# Specify the list of light response curve IDs, if no light response curve, input "lightresp_id = None" or ignore it.
lcd = fvcb.initLicordata(dftest, preprocess=True, lightresp_id = [118])
```
### Define the device
Default device is 'cpu'. If you have an NVIDIA GPU, set 'device_fit' to 'cuda' and execute the 'lcd.todevice(torch.device(device_fit))' line.
```bash
device_fit = 'cpu'
lcd.todevice(torch.device(device_fit)) # if device is cuda, then execute this line
```
### Initialize FvCB model
If 'onefit' is set to 'True', all curves in a fitting group will share the same set of Vcmax25, Jmax25, TPU25, and Rd25.
Otherwise, each curve will have its own set of these four main parameters but share the same light and temperature response parameters for the fitting group.

If no light response curve is specified, set 'LightResp_type' to 0.

LightResp_type 0: J is equal to Jmax.

LightResp_type 1: using equation $J = \frac{\alpha Q J_{max}}{\alpha Q + J_{max}}$ and fitting $\alpha$.

LightResp_type 2: using equation $J = \frac{\alpha Q + J_{max} - \sqrt{(\alpha Q + J_{max})^2 - 4 \theta \alpha Q J_{max}}}{2 \theta}$ and fitting $\alpha$ and $\theta$.

TempResp_type 0: Vcmax, Jmax, TPU, and Rd are equal to the Vcmax25, Jmax25, TPU25, and Rd25, respectively.

TempResp_type 1: using equation $k = k_{25} \exp{\left[\frac{\Delta{H_a}}{R}\left(\frac{1}{298}-\frac{1}{T_{leaf}}\right)\right]}$ and fitting $\Delta{H_a}$ for Vcmax, Jmax, and TPU.

TempResp_type 2: using equation $k = k_{25} \exp\left[\frac{\Delta H_a}{R} \left(\frac{1}{298}-\frac{1}{T_{leaf}}\right)\right]  \frac{f\left(298\right)}{f\left(T_{leaf}\right)}$, where $f(T) = 1+\exp \left[\frac{\Delta H_d}{R}\left(\frac{1}{T_{opt}}-\frac{1}{T} \right)-\ln \left(\frac{\Delta H_d}{\Delta H_a}-1 \right) \right]$, and fitting $\Delta{H_a}$ and $T_{opt}$ for Vcmax, Jmax, and TPU.

```bash
# initialize the model
fvcbm = fvcb.model(lcd, LightResp_type = 2, TempResp_type = 2, onefit = False)
```
### More fitting options
fitRd: option to fit $R_{d25}$, default is True. If set to False, $R_{d}$ will be fixed to 1% of $V_{cmax}$.

fitRdratio: option to fit $R_{d}$-to- $V_{cmax}$ ratio, default is False, the range is 0.01 to 0.02. 

fitag: option to fit $\alpha_g$, default is False, the range is 0 to 1.

fitKc: option to fit $k_{25}$, default is False.

fitKo: option to fit $k_{25}$, default is False.

fitgamma: option to fit $\Gamma^*_{25}$, default is False.

fitgm: option to fit $g_m$, default is False.
```bash
fvcbm = fvcb.model(lcd, LightResp_type = 0, TempResp_type = 1, onefit = False, fitRd = True, fitRdratio = False, fitag = False, fitgm= False, fitgamma=False, fitKo=False, fitKc=False, allparams=allparams)
```
### Specify default fixed or learnable parameters
```bash
allparams = fvcb.allparameters()
allparams.dHa_Vcmax = torch.tensor(40.0).to(device_fit) # If the device is cuda, then execute ".to(device_fit)"
fvcbm = fvcb.model(lcd, LightResp_type = 0, TempResp_type = 1, onefit = False, fitag = False, fitgm= False, fitgamma=False, fitKo=False, fitKc=False, allparams=allparams)
```
### Fit A/Ci curves
```bash
fitresult = fvcb.fit(fvcbm, learn_rate= 0.08, maxiteration = 20000, minloss= 1, recordweightsTF=False)
fvcbm = fitresult.model
```
### Get fitted parameters by ID
The main parameters are stored in the 'fvcbm'. The temperature response parameters are in 'fvcbm.TempResponse', just like the light response parameters.
```bash
id_index = 0
id = int(lcd.IDs[id_index]) # target curve ID
fg_index =  int(lcd.FGs_idx[id_index]) # index of the corresponding fitting group
if not fvcbm.onefit:
    Vcmax25_id = fvcbm.Vcmax25[id_index]
    Jmax25_id = fvcbm.Jmax25[id_index]
else:
    Vcmax25_id = fvcbm.Vcmax25[fg_index]
    Jmax25_id = fvcbm.Jmax25[fg_index]

dHa_Vcmax_id = fvcbm.TempResponse.dHa_Vcmax[fg_index]
alpha_id = fvcbm.LightResponse.alpha[fg_index]
```
### Get fitted A/Ci curves
```bash
A, Ac, Aj, Ap = fvcbm()
```

### Get fitted A/Ci curves by ID
```bash
id_index = 0
id = lcd.IDs[id_index]
indices_id = lcd.getIndicesbyID(id)
A_id = A[indices_id]
Ac_id = Ac[indices_id]
```
### Get the (preprocessed) photosynthesis data by ID
```bash
A_id_mea, Ci_id, Q_id, Tlf_id = lcd.getDatabyID(lcd.IDs[id_index])
```
***

## 2. Stomatal conductance model usage
Three stomatal conductance model is currently available: Buckley Mott Farquhar (BMF), Medlyn (MED), and Ball Woodrow Berry (BWB). The Ball Berry Leuning (BBL) model is under development.
More details about these four models can be found at: https://baileylab.ucdavis.edu/software/helios/_stomatal_doc.html.

Create a python file in the PhoTorch directory and import necessary packages.
```bash
import stomatal
import pandas as pd
import torch
```
### Initialize the stomatal conductance data
The data to be loaded should be:

| CurveID | gsw | VPDleaf_mmolmol-1 | A  | Qin   | Tleaf | RH   |
|---------|-----|-------------------|----|-------|-------|------|
| 0       | 0.34| 11.32             | 55 | 2000  | 21.81 | 21.0 |
| 0       | 0.34| 18.33             | 55 | 2000  | 22.71 | 30.02|
| 0       | 0.35| 29.57             | 51 | 2000  | 20.02 | 38.01|
| 0       | 0.38| 15.4              | 54 | 2000  | 22.6  | 26.99|
| 0       | 0.32| 15.44             | 54 | 1200  | 19.97 | 27.0 |
| 1       | 0.23| 29.03             | 49 | 2000  | 17.93 | 35.92|
| 1       | 0.29| 20.51             | 50 | 2000  | 20.51 | 29.96|
| 1       | 0.28| 11.77             | 49 | 2000  | 18.61 | 19.99|
'A' is not necessary for BMF model.

```bash
datasc = pd.read_csv('exampledata/steadystate_stomatalconductance.csv')
scd = stomatal.initscdata(datasc)
```
### Initialize the BMF model and fit the parameters Emerson effect (Em), quantum yield of electron transport (i0), curvature factor (k), and intercept (b).
```bash
scm = stomatal.BMF(scd)
#scm = stomatal.BWB(scd) 
#scm = stomatal.MED(scd)
fitresult = stomatal.fit(scm, learnrate = 0.5, maxiteration =20000)
scm = fitresult.model
```
### Get the fitted and measured stomatal conductance
```bash
gsw = scm()
gsw_mea = scd.gsw
```
### Get the fitted stomatal conductance by ID
```bash
id_index = 0
id = scd.IDs[id_index]
indices_id = scd.getIndicesbyID(id)
gsw_id = gsw[indices_id]
```
### Get the fitted parameters by ID
```bash
id_index = 0
id = int(scd.IDs[id_index])
Em_id = scm.Em[id_index]
i0_id = scm.i0[id_index]
k_id = scm.k[id_index]
b_id = scm.b[id_index]
```
  
## 3. PROSPECT-X model usage
The PROSPECT-X model is under development.
