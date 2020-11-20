# ClusterSinkingParticles
This repository contains the code that is used to produce the results of the paper.
Code from two other repositories were used: _OceanParcels/coherent_vortices_OPTICS_ and _OceanParcels/drifter_trajectories_network_ .

The **OPTICS** folder contains the code that was used for all figures with the OPTICS algorithm. The **hierclus** folder was used for all results that are related to hierarchical clustering.

## conda environments
Two conda environments were used (given by the two .yml files). Most of the script use the _Cartopy-py3_ environment. Those scripts that use any biostatistics, which either make use of the package _skbio_ or _ecopy_, use _skbio_env_ environment.

