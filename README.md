## Dichotomous Noise Models for Methane Production

This repository contains code for a numerical model used to analyze spikes in methane production from wetlands using a dichotomous noise model. 
Using this approach allows us to focus in on the effect that active water table dynamics have on methane capture and release. Rapid switching between oxic and anoxic
states across a threshold may create spikes in methane production.


**Included Files**
Codes:
sampledichotomousmodel.py - Basic code for implemeting a completely synthetic dichotomous noise model including some preliminary sensitivity tests on changing switching constants
data-cleaning.py - parameterization of sensor data from wetland Ameriflux sites with both methane and water table data. 

**Unpcoming work**
1. Continue to clean the data associated with the Ameriflux sites and run sample dichotomous models for each. Calibration will be needed to match methane data. 
2. Develop a pdf for methane rates based on where to position the critical threshold. To do this follow a similar switching derivation as found in the 'Crossing properties of soil moisture dynamics' chapter of Echoydrology of Water Controlled Ecosystems (Rodriguez-Iturbe, Porporato)

