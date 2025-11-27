# 1. Introduction

Spatial Multi-Omics Integration with Graph Transformers and Laplacian Positional Encoding

![Fig](/Image/architecture_enhanced_cosmos.png) 

This is an improvement upon the research Paper: Cooperative Integration of Spatially Resolved Multi-Omics Data with COSMOS, Zhou Y., X. Xiao, L. Dong, C. Tang, G. Xiao*, and L Xu*, 2024. 
    
# 2. Environment setup and code compilation

__2.1. Environment setup__

You can use the following command to install the dependencies with conda:
```
conda install pandas numpy scanpy matplotlib umap-learn scikit-learn seaborn torch networkx gudhi anndata cmcrameri torch-geometric louvain leidenalg
```

Or you can use pip:
```
pip install pandas numpy scanpy matplotlib umap-learn scikit-learn seaborn torch networkx gudhi anndata cmcrameri torch-geometric louvain leidenalg
```

__2.2. Code Repository and Dataset Availability__

Code Repository: Complete implementation  is available at https://github.com/anujsngh/ComputationalGenomicsMajorProject.git

Datasets: Datasets used are available at https://drive.google.com/file/d/1fY5CAmBUB7KdEmGxAKBZ0ypY_h_dldpF/view?usp=sharing


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


# 3. Copyright information 

Please see the "LICENSE" file for the copyright information.
