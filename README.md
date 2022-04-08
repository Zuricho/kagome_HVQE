# Variational Quantum Eigensolver For Kagome Lattice

Author: Yuheng Guo, Bozitao Zhong, Xingyu Chen

**Our paper will be available soon in ArXiv. A preliminary version is in folder `./report`**

HVQE implementation of Kagome lattice. This project can calculate both ground state energy and excited energies.

## Install Environment on Windows Platform

```bash
conda create -n HVQE
conda activate HVQE
conda install cupy chainer scipy matplotlib numpy pandas
```

You can also install your environment by `conda env create -f HVQE_environment.yml`, this version is for Linux systems only.

If you want to install on Win: just install by the conda code above. Using Windows cannot use GPU because there is no GPU supported version of `cupy` (7.7.0~8.0.0), so I install the `cupy` 8.3.0
