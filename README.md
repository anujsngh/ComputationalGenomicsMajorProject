# 1. Introduction

COSMOS is a computational tool crafted to overcome the challenges associated 
with integrating spatially resolved multi-omics data. This software harnesses a 
graph neural network algorithm to deliver cutting-edge solutions for analyzing 
biological data that encompasses various omics types within a spatial framework. 
Key features of COSMOS include domain segmentation, effective visualization, and 
the creation of spatiotemporal maps. These capabilities empower researchers to 
gain a deeper understanding of the spatial and temporal dynamics within 
biological samples, distinguishing COSMOS from other tools that may only support 
single omics types or lack comprehensive spatial integration. The proven 
superior performance of COSMOS underscores its value as an essential resource in 
the realm of spatial omics.

![Fig](/Image/Figure_1.png) 

Paper: Cooperative Integration of Spatially Resolved Multi-Omics Data with 
COSMOS, Zhou Y., X. Xiao, L. Dong, C. Tang, G. Xiao*, and L Xu*, 2024. 

DOI for the Latest Released Version of the Github repository: 
[10.5281/zenodo.14114770](https://doi.org/10.5281/zenodo.14114770).
    
# 2. Environment setup and code compilation

__2.1. Download the package__

The package can be downloaded by running the following command in the terminal:
```
git clone https://github.com/Lin-Xu-lab/COSMOS.git
```
Then, use
```
cd COSMOS
```
to access the downloaded folder. 

If the "git clone" command does not work with your system, you can download the 
zip file from the website 
https://github.com/Lin-Xu-lab/COSMOS.git and decompress it. Then, the folder 
that you need to access is COSMOS-main. 

__2.2. Environment setup__

The package has been successuflly tested in a Linux environment of python 
version 3.8.8, pandas version 1.5.2, and so on. An option to set up 
the environment is to use Conda 
(https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

You can use the following command to create an environment for COSMOS:
```
conda create -n cosmos python pandas numpy scanpy matplotlib umap-learn scikit-learn seaborn torch networkx gudhi anndata cmcrameri pytorch-geometric
```

After the environment is created, you can use the following command to activate 
it:
```
conda activate cosmos
```

Please install Jupyter Notebook from https://jupyter.org/install. For example, 
you can run
```
pip install notebook
```
in the terminal to install the classic Jupyter Notebook.  

__2.3. Import COSMOS in different directories (optional)__

If you would like to import COSMOS in different directories, there is an option 
to make it work. Please run
```
python setup.py install --user &> log
```
in the terminal.

After doing these successfully, you are supposed to be able to import COSMOS 
when you are using Python or Jupyter Notebook in other folders:
```
import COSMOS
```

__2.4. Using "pip install" to install the COSMOS package__

Please run
```
pip install COSMOS-LinXuLab
```
in the terminal.

# 3. Tutorials

The step-by-step guides for closely replicating the COSMOS results on simulated mouse visual cortex multi-omics data, ATAC-RNA-Seq mouse brain multi-omics data, and DBiT-Seq mouse embryo multi-omics data are accessible at: [Tutorials](./Tutorials) and [COSMOS Tutorials on Read the Docs](https://cosmos-tutorials.readthedocs.io/en/latest/index.html). Furthermore, all the processed data required to reproduce the figures presented in the manuscript can be found at Zenodo under the DOI: [10.5281/zenodo.13932144](https://zenodo.org/records/13932144).

# 4. Contact information

Please contact our team if you have any questions:

Yuansheng Zhou (Yuansheng.Zhou@UTSouthwestern.edu)

Xue Xiao (Xiao.Xue@UTSouthwestern.edu)

Lei Dong (Lei.Dong@UTSouthwestern.edu)

Chen Tang (Chen.Tang@UTSouthwestern.edu)

Lin Xu (Lin.Xu@UTSouthwestern.edu)

Please contact Xue Xiao for programming questions about the *.py

# 5. Copyright information 

Please see the "LICENSE" file for the copyright information.
