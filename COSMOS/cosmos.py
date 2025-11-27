'''
This open-source software is for implementing the COSMOS algorithm. 
Paper: Cooperative Integration of Spatially Resolved Multi-Omics Data with COSMOS

Please contact our team if you have any questions:
Yuansheng Zhou (Yuansheng.Zhou@UTSouthwestern.edu)
Xue Xiao (Xiao.Xue@UTSouthwestern.edu)
Chen Tang (Chen.Tang@UTSouthwestern.edu)
Lin Xu (Lin.Xu@UTSouthwestern.edu)

Please contact Xue Xiao for programming questions about the *.py files.

Version: 10/10/2024 (Updated with inference capabilities)

Please see the "LICENSE" file for the copyright information. 

NOTICE: This COSMOS software is adapted from the spaceflow code 
        (https://github.com/hongleir/SpaceFlow). 
        Please see the "LICENSE" file for copyright details of the spaceflow software.
        The implementation of the spaceflow software is described in 
        the publication "Identifying multicellular spatiotemporal organization of cells with SpaceFlow." 
        (https://www.nature.com/articles/s41467-022-31739-w).

        This COSMOS software includes functionality from pyWNN 
        (Weighted Nearest Neighbors Analysis implemented in Python), which is based on code 
        from the https://github.com/dylkot/pyWNN. 
        Please see the "LICENSE" file for copyright details of the pyWNN software.

        The DeepGraphInfomaxWNN function in the COSMOS software is adapted from the 
        torch_geometric.nn.models.deep_graph_infomax function in PyTorch Geometric (PyG),
        available at https://github.com/pyg-team/pytorch_geometric/tree/master. 
        Please see the "LICENSE" file for copyright details of the PyG software.
'''
import math
import os
import torch
import random
import gudhi
import anndata
import cmcrameri
import numpy as np
import scanpy as sc
import networkx as nx
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy.spatial import distance_matrix
from torch_geometric.nn import GCNConv
from sklearn.neighbors import kneighbors_graph
from .modulesWNN import DeepGraphInfomaxWNN
from .pyWNN import pyWNN


def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

    
def corruptionWNNit(x1, x2, edge_index,adata,w,w1,w2):
    return x1[torch.randperm(x1.size(0))], x2[torch.randperm(x2.size(0))], edge_index,adata,0,w1,w2

class Cosmos(object):
    """An object for analysis of spatial transcriptomics data.
    :param adata1 / adata2: `anndata.AnnData` object as input, see `https://anndata.readthedocs.io/en/latest/` for more info about`anndata`.
    :type adata: class:`anndata.AnnData`
    :param count_matrix1 / count_matrix2: count matrix of gene expression, 2D numpy array of size (n_cells, n_genes)
    :type count_matrix: class:`numpy.ndarray`
    :param spatial_locs: spatial locations of cells (or spots) match to rows of the count matrix, 1D numpy array of size (n_cells,)
    :type spatial_locs: class:`numpy.ndarray`
    :param sample_names: list of sample names in 1D numpy str array of size (n_cells,), optional
    :type sample_names: class:`numpy.ndarray` or `list` of `str`
    :param gene_names: list of gene names in 1D numpy str array of size (n_genes,), optional
    :type gene_names: class:`numpy.ndarray` or `list` of `str`
    :param save_inter_emb: if True, save intermediate embeddings before fusion
    :type save_inter_emb: bool
    :param save_fin_emb: if True, save final embeddings after fusion
    :type save_fin_emb: bool
    :param output_dir: directory to save embeddings
    :type output_dir: str
    """

    def __init__(self, adata1=None, adata2=None, count_matrix1=None, count_matrix2=None, 
                 spatial_locs=None, sample_names=None, gene_names=None,
                 save_inter_emb=False, save_fin_emb=False, output_dir='./embeddings'):
        """
        Inputs
        ------
        adata1 / adata2: the anndata.AnnData type object, optional (either input `adata` or both `count_matrix` and `spatial_locs`)
        count_matrix1 / count_matrix2 : count matrix of gene expression, 2D numpy array of size (n_cells, n_genes)
        spatial_locs : spatial locations of cells (or spots) match to rows of the count matrix, 1D numpy array of size (n_cells,)
        sample_names : list of sample names in 1D numpy str array of size (n_cells,), optional
        gene_names : list of gene names in 1D numpy str array of size (n_genes,), optional
        save_inter_emb : if True, save intermediate embeddings before fusion
        save_fin_emb : if True, save final embeddings after fusion
        output_dir : directory to save embeddings
        """
        if adata1 and isinstance(adata1, anndata.AnnData):
            self.adata1 = adata1
        if adata2 and isinstance(adata2, anndata.AnnData):
            self.adata2 = adata2
        elif count_matrix1 is not None and count_matrix2 is not None and spatial_locs is not None:
            self.adata1 = anndata.AnnData(count_matrix1.astype(float))
            self.adata1.obsm['spatial'] = spatial_locs.astype(float)
            self.adata2 = anndata.AnnData(count_matrix2.astype(float))
            self.adata2.obsm['spatial'] = spatial_locs.astype(float)
            if gene_names:
                self.adata1.var_names = np.array(gene_names).astype(str)
                self.adata2.var_names = np.array(gene_names).astype(str)
            if sample_names:
                self.adata1.obs_names = np.array(sample_names).astype(str)
                self.adata2.obs_names = np.array(sample_names).astype(str)
        else:
            print("Please input either an anndata.AnnData or both the count_matrix (count matrix of gene expression, 2D int numpy array of size (n_cells, n_genes)) and spatial_locs (spatial locations of cells (or spots) in 1D float numpy array of size (n_locations,)) to initiate COSMOS object.")
            exit(1)
        
        # NEW: Embedding saving configuration
        self.save_inter_emb = save_inter_emb
        self.save_fin_emb = save_fin_emb
        self.output_dir = output_dir
        
        # Create output directory if needed
        if save_inter_emb or save_fin_emb:
            os.makedirs(output_dir, exist_ok=True)
        
        # Storage for embeddings
        self.z1_intermediate = None  # Modality 1 before fusion
        self.z2_intermediate = None  # Modality 2 before fusion
        self.z_final = None          # After fusion
        
        # Model storage
        self.model = None
        self.device = None
        self.z_dim = None
        self.n_neighbors = None

    
    def preprocessing_data(self, do_norm = False, do_log = False, n_top_genes=None, do_pca = False, n_neighbors=10):
        """
        Preprocessing the spatial transcriptomics data
        Generates:  `self.adata_filtered`: (n_cells, n_locations) `numpy.ndarray`
                    `self.spatial_graph`: (n_cells, n_locations) `numpy.ndarray`
        :param adata: the annData object for spatial transcriptomics data with adata.obsm['spatial'] set to be the spatial locations.
        :type adata: class:`anndata.annData`
        :param do_norm: whether or not perfomr normalization on the data
        :type do_norm: bool, optional, default: False
        :param do_log: whether or not perfomrm log transformation on the data
        :type do_log: bool, optional, default: False
        :param n_top_genes: the number of top highly variable genes
        :type n_top_genes: int, optional,default: None
        :param do_pca: whether or not perfomrm pca  on the data
        :type do_pca: bool, optional, default: False
        :param n_neighbors: the number of nearest neighbors for building spatial neighbor graph
        :type n_neighbors: int, optional
        :return: a preprocessed annData object of the spatial transcriptomics data
        :rtype: class:`anndata.annData`
        :return: a geometry-aware spatial proximity graph of the spatial spots of cells
        :rtype: class:`scipy.sparse.csr_matrix`
        """
        adata1 = self.adata1
        adata2 = self.adata2
        if not adata1 or not adata2:
            print("Not enough annData objects")
            return
        if do_norm:
            sc.pp.normalize_total(adata1, target_sum=1e4)
            sc.pp.normalize_total(adata2, target_sum=1e4)
        if do_log:
            sc.pp.log1p(adata1)
            sc.pp.log1p(adata2)
        if n_top_genes:
            sc.pp.highly_variable_genes(adata1, n_top_genes=n_top_genes, flavor='cell_ranger', subset=True)
            sc.pp.highly_variable_genes(adata2, n_top_genes=n_top_genes, flavor='cell_ranger', subset=True)
        if do_pca:
            sc.pp.pca(adata1)
            sc.pp.pca(adata2)
        spatial_locs = adata1.obsm['spatial']
        spatial_graph = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors, mode='distance')
        self.adata1_preprocessed = adata1
        self.adata2_preprocessed = adata2
        self.spatial_graph = spatial_graph
        self.n_neighbors = n_neighbors  # Store for later use

    def train(self, embedding_save_filepath="./embedding.tsv", weights_save_filepath="./weights.tsv", spatial_regularization_strength=0.05, z_dim=50, lr=1e-3, wnn_epoch  = 100, total_epoch = 1000, max_patience_bef=10, max_patience_aft=30, min_stop=100, random_seed=42, gpu=0, regularization_acceleration=True, edge_subset_sz=1000000):
        adata1_preprocessed, adata2_preprocessed,spatial_graph = self.adata1_preprocessed, self.adata2_preprocessed, self.spatial_graph
        """
        Training the Deep GraphInfomax Model
        :param embedding_save_filepath: the default save path for the low-dimensional embeddings
        :type embedding_save_filepath: class:`str`
        :param spatial_regularization_strength: the strength for spatial regularization
        :type spatial_regularization_strength: float, optional, default: 0.05
        :param z_dim: the size of latent dimension
        :type z_dim: int, optional, default: 50
        :param lr: the learning rate for model optimization
        :type lr: float, optional, default: 1e-3
        :param wnn_epoch : the iteration number before performing WNN
        :type wnn_epoch : int, optional, default: 100
        :param total_epoch: the total iteration number for model optimization
        :type total_epoch: int, optional, default: 1000
        :param max_patience: the patience for early stopping training in contrastive learning phase
        :type max_patience: int, optional, default: 50
        :param max_patience_aft: the patience for early stopping training after WNN
        :type max_patience_aft: int, optional, default: 50
        :param min_stop: the minimum iteration number before trigger the early stopping 
        :type min_stop: int, optional, default: 100
        :param random_seed: the random seed for model optimization
        :type random_seed: int, optional, default: 42
        :param gpu: the gpu device index used for model optimization, or cpu by default
        :type gpu: int, optional, default: 0
        :param regularization_acceleration: whether or not using regularization acceleration for large datasets
        :type regularization_acceleration: bool, optional, default: True
        :param edge_subset_sz: number of spatial edges for a random subset to accelerate spatial regularization computation
        :type edge_subset_sz: int, optional, default: 1000000
        :return: a low-dimensional embedding representation of the spatial transcriptomics data
        :rtype: class:`numpy.ndarray`
        """
        if not adata1_preprocessed or not adata2_preprocessed:
            print("The data has not been preprocessed, please run preprocessing_data() method first!")
            return
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
        self.device = device  # Store device
        self.z_dim = z_dim    # Store z_dim
        
        model = DeepGraphInfomaxWNN(
        hidden_channels=z_dim, encoder=GraphEncoderWNNit(adata1_preprocessed.shape[0],adata1_preprocessed.shape[1],adata2_preprocessed.shape[1], z_dim),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruptionWNNit).to(device)
        
        self.model = model  # Store model
        
        expr1 = adata1_preprocessed.X.todense() if type(adata1_preprocessed.X).__module__ != np.__name__ else adata1_preprocessed.X
        expr1 = torch.tensor(expr1.copy()).float().to(device)
        
        expr2 = adata2_preprocessed.X.todense() if type(adata2_preprocessed.X).__module__ != np.__name__ else adata2_preprocessed.X
        expr2 = torch.tensor(expr2.copy()).float().to(device)
        
        edge_list = sparse_mx_to_torch_edge_list(spatial_graph).to(device)
        
        # Store for later inference use
        self.feature_X_1 = expr1
        self.feature_X_2 = expr2
        self.adj_1 = edge_list
        self.adj_2 = edge_list

        model.train()
        min_loss = np.inf
        patience_aft = 0
        patience_bef = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_params = model.state_dict()
        w1=0.5
        w2=0.5
        for epoch in range(1,total_epoch):
            train_loss = 0.0
            torch.set_grad_enabled(True)
            optimizer.zero_grad()
            if epoch == wnn_epoch  or patience_bef > max_patience_bef: 
                # setting early stop to run wnn
                # running wnn for only one time
                z, neg_z, summary,w1,w2 = model(expr1, expr2, edge_list, adata1_preprocessed,1,0,0)
                wnn_epoch  = 0 
                min_loss = np.inf
                max_patience_bef = total_epoch
            else:
                z, neg_z, summary,w1,w2 = model(expr1, expr2, edge_list, adata1_preprocessed,0,w1,w2)           
                
                
            loss = model.loss(z, neg_z, summary)
            coords = torch.tensor(adata1_preprocessed.obsm['spatial']).float().to(device)
            if spatial_regularization_strength > 0:
                if regularization_acceleration or adata1_preprocessed.shape[0] > 5000:
                    cell_random_subset_1, cell_random_subset_2 = torch.randint(0, z.shape[0], (edge_subset_sz,)).to(
                        device), torch.randint(0, z.shape[0], (edge_subset_sz,)).to(device)
                    z1, z2 = torch.index_select(z, 0, cell_random_subset_1), torch.index_select(z, 0, cell_random_subset_2)
                    c1, c2 = torch.index_select(coords, 0, cell_random_subset_1), torch.index_select(coords, 0,
                                                                                                     cell_random_subset_1)
                    pdist = torch.nn.PairwiseDistance(p=2)

                    z_dists = pdist(z1, z2)
                    z_dists = z_dists / torch.max(z_dists)

                    sp_dists = pdist(c1, c2)
                    sp_dists = sp_dists / torch.max(sp_dists)

                    n_items = z_dists.size(dim=0)
                else:
                    z_dists = torch.cdist(z, z, p=2)
                    z_dists = torch.div(z_dists, torch.max(z_dists)).to(device)
    
                    sp_dists = torch.cdist(coords, coords, p=2)
                    sp_dists = torch.div(sp_dists, torch.max(sp_dists)).to(device)
            
                    n_items = z.size(dim=0) * z.size(dim=0)
                penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items).to(device)
            else: penalty_1 = 0 
            
            loss = loss + spatial_regularization_strength * penalty_1
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if epoch > wnn_epoch:
                if train_loss > min_loss:
                    patience_aft += 1
                else:
                    patience_aft = 0
                    min_loss = train_loss
                    best_params = model.state_dict()
            else:
                if train_loss > min_loss:
                    patience_bef += 1
                else:
                    patience_bef = 0
                    min_loss = train_loss
                    best_params = model.state_dict()
            if epoch % 10 == 1:
                print(f"Epoch {epoch}/{total_epoch}, Loss: {str(train_loss)}")
            if patience_aft > max_patience_aft and epoch > min_stop:
                break

        model.load_state_dict(best_params)

        z, _, _,w1,w2 = model(expr1, expr2, edge_list, adata1_preprocessed,0,w1,w2)
        embedding = z.cpu().detach().numpy()
        w1 = w1.cpu().detach().numpy().reshape(-1,1)
        w2 = w2.cpu().detach().numpy().reshape(-1,1)
        ww = np.hstack((w1,w2))

        self.embedding = embedding
        self.weights = ww
        
        # NEW: Extract and save embeddings if requested
        print("\nExtracting final embeddings...")
        z1, z2, z_final = self._get_embeddings(return_intermediate=True)
        
        self.z1_intermediate = z1
        self.z2_intermediate = z2
        self.z_final = z_final
        
        if self.save_inter_emb:
            self._save_intermediate_embeddings()
        
        if self.save_fin_emb:
            self._save_final_embeddings()
        
        return embedding

    # NEW METHODS BELOW
    
    def _get_embeddings(self, return_intermediate=False):
        """
        Extract embeddings from the trained model.
        
        Parameters:
        -----------
        return_intermediate : bool
            If True, return embeddings before fusion as well
        
        Returns:
        --------
        If return_intermediate=False:
            z_final : numpy array of shape (n_cells, z_dim)
        If return_intermediate=True:
            z1, z2, z_final : tuple of numpy arrays
        """
        if self.model is None:
            raise ValueError("No trained model found. Please train the model first.")
        
        self.model.eval()
        
        with torch.no_grad():
            # Get modality 1 embedding
            x1 = self.model.encoder.conv1(
                self.feature_X_1.to(self.device),
                self.adj_1.to(self.device)
            )
            x1 = self.model.encoder.prelu(x1)
            z1 = self.model.encoder.conv3(x1, self.adj_1.to(self.device))
            z1 = self.model.encoder.prelu2(z1)
            z1 = nn.functional.normalize(z1, p=2.0, dim=1)
            
            # Get modality 2 embedding
            x2 = self.model.encoder.conv2(
                self.feature_X_2.to(self.device),
                self.adj_2.to(self.device)
            )
            x2 = self.model.encoder.prelu3(x2)
            z2 = self.model.encoder.conv4(x2, self.adj_2.to(self.device))
            z2 = self.model.encoder.prelu4(z2)
            z2 = nn.functional.normalize(z2, p=2.0, dim=1)
            
            # Get final embedding (weighted combination)
            w1 = self.weights[:, 0].reshape(-1, 1)
            w2 = self.weights[:, 1].reshape(-1, 1)
            w1_tensor = torch.from_numpy(w1).float().to(self.device)
            w2_tensor = torch.from_numpy(w2).float().to(self.device)
            
            z_final = z1 * w1_tensor + z2 * w2_tensor
        
        # Convert to numpy
        z1_np = z1.detach().cpu().numpy()
        z2_np = z2.detach().cpu().numpy()
        z_final_np = z_final.detach().cpu().numpy()
        
        if return_intermediate:
            return z1_np, z2_np, z_final_np
        else:
            return z_final_np
    
    def _save_intermediate_embeddings(self):
        """Save intermediate embeddings before fusion."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as numpy
        np.save(
            os.path.join(self.output_dir, f'z1_modality1_gcn_{timestamp}.npy'),
            self.z1_intermediate
        )
        np.save(
            os.path.join(self.output_dir, f'z2_modality2_gcn_{timestamp}.npy'),
            self.z2_intermediate
        )
        
        # Also save as h5ad for compatibility with scanpy
        adata_z1 = sc.AnnData(self.z1_intermediate)
        adata_z1.obs = self.adata1.obs.copy()
        if 'spatial' in self.adata1.obsm:
            adata_z1.obsm['spatial'] = self.adata1.obsm['spatial'].copy()
        adata_z1.write_h5ad(
            os.path.join(self.output_dir, f'z1_modality1_gcn_{timestamp}.h5ad')
        )
        
        adata_z2 = sc.AnnData(self.z2_intermediate)
        adata_z2.obs = self.adata2.obs.copy()
        if 'spatial' in self.adata2.obsm:
            adata_z2.obsm['spatial'] = self.adata2.obsm['spatial'].copy()
        adata_z2.write_h5ad(
            os.path.join(self.output_dir, f'z2_modality2_gcn_{timestamp}.h5ad')
        )
        
        print(f"✓ Saved intermediate embeddings to {self.output_dir}")
        print(f"  - Modality 1 shape: {self.z1_intermediate.shape}")
        print(f"  - Modality 2 shape: {self.z2_intermediate.shape}")

    def _save_final_embeddings(self):
        """Save final embeddings after fusion."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as numpy
        np.save(
            os.path.join(self.output_dir, f'z_final_fused_gcn_{timestamp}.npy'),
            self.z_final
        )
        
        # Save as h5ad
        adata_final = sc.AnnData(self.z_final)
        adata_final.obs = self.adata1.obs.copy()
        if 'spatial' in self.adata1.obsm:
            adata_final.obsm['spatial'] = self.adata1.obsm['spatial'].copy()
        adata_final.write_h5ad(
            os.path.join(self.output_dir, f'z_final_fused_gcn_{timestamp}.h5ad')
        )
        
        print(f"✓ Saved final embeddings to {self.output_dir}")
        print(f"  - Final embedding shape: {self.z_final.shape}")
    
    def save_model(self, save_path='./models/trained_gcn_model.pt'):
        """
        Save trained model weights and configuration.
        
        Parameters:
        -----------
        save_path : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save. Please train the model first.")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'weights': self.weights,
            'z_dim': self.z_dim,
            'feature_dims': {
                'modality1': self.adata1_preprocessed.shape[1],
                'modality2': self.adata2_preprocessed.shape[1]
            },
            'n_cells': self.adata1_preprocessed.shape[0],
            'spatial_coords': self.adata1_preprocessed.obsm['spatial'],
            'preprocessing_params': {
                'n_neighbors': self.n_neighbors if hasattr(self, 'n_neighbors') else None
            }
        }
        
        torch.save(save_dict, save_path)
        print(f"✓ Model saved to {save_path}")

    def load_model(self, load_path='./models/trained_gcn_model.pt', gpu=0):
        """
        Load trained model weights.
        
        Parameters:
        -----------
        load_path : str
            Path to the saved model
        gpu : int
            GPU device number
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Load checkpoint with weights_only=False to allow numpy arrays
        # This is safe if you trust the checkpoint source
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        
        # Reconstruct model with saved dimensions
        self.z_dim = checkpoint['z_dim']
        
        # Initialize model architecture
        self.model = DeepGraphInfomaxWNN(
            hidden_channels=self.z_dim,
            encoder=GraphEncoderWNNit(
                checkpoint['n_cells'],
                checkpoint['feature_dims']['modality1'],
                checkpoint['feature_dims']['modality2'],
                self.z_dim
            ),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruptionWNNit
        ).to(device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.weights = checkpoint['weights']
        self.n_neighbors = checkpoint['preprocessing_params']['n_neighbors']
        
        print(f"✓ Model loaded from {load_path}")
        print(f"  - z_dim: {self.z_dim}")
        print(f"  - WNN weights shape: {self.weights.shape}")
        
        return checkpoint

    def inference(self, adata1_new, adata2_new, save_embeddings=True, 
                  output_prefix='inference'):
        """
        Run inference on new data using trained model.
        
        Parameters:
        -----------
        adata1_new : AnnData
            New data for modality 1
        adata2_new : AnnData
            New data for modality 2
        save_embeddings : bool
            Whether to save the embeddings
        output_prefix : str
            Prefix for output files
        
        Returns:
        --------
        z1, z2, z_final : tuple of embeddings
        """
        if self.model is None:
            raise ValueError("No trained model found. Please train or load a model first.")
        
        print("\nRunning inference on new data...")
        
        # Store original data temporarily
        original_adata1 = self.adata1
        original_adata2 = self.adata2
        original_preprocessed1 = self.adata1_preprocessed if hasattr(self, 'adata1_preprocessed') else None
        original_preprocessed2 = self.adata2_preprocessed if hasattr(self, 'adata2_preprocessed') else None
        
        # Set new data
        self.adata1 = adata1_new
        self.adata2 = adata2_new
        
        # Preprocess new data (same as training)
        print("Preprocessing new data...")
        self.preprocessing_data(n_neighbors=self.n_neighbors)
        
        # Prepare data for model
        expr1 = self.adata1_preprocessed.X.todense() if type(self.adata1_preprocessed.X).__module__ != np.__name__ else self.adata1_preprocessed.X
        expr1 = torch.tensor(expr1.copy()).float().to(self.device)
        
        expr2 = self.adata2_preprocessed.X.todense() if type(self.adata2_preprocessed.X).__module__ != np.__name__ else self.adata2_preprocessed.X
        expr2 = torch.tensor(expr2.copy()).float().to(self.device)
        
        edge_list = sparse_mx_to_torch_edge_list(self.spatial_graph).to(self.device)
        
        # Store for embedding extraction
        self.feature_X_1 = expr1
        self.feature_X_2 = expr2
        self.adj_1 = edge_list
        self.adj_2 = edge_list
        
        # Get embeddings
        z1, z2, z_final = self._get_embeddings(return_intermediate=True)
        
        # Save if requested
        if save_embeddings:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save intermediate
            np.save(
                os.path.join(self.output_dir, f'{output_prefix}_z1_{timestamp}.npy'),
                z1
            )
            np.save(
                os.path.join(self.output_dir, f'{output_prefix}_z2_{timestamp}.npy'),
                z2
            )
            
            # Save final
            np.save(
                os.path.join(self.output_dir, f'{output_prefix}_z_final_{timestamp}.npy'),
                z_final
            )
            
            # Save as h5ad
            adata_z1 = sc.AnnData(z1)
            adata_z1.obs = adata1_new.obs.copy()
            if 'spatial' in adata1_new.obsm:
                adata_z1.obsm['spatial'] = adata1_new.obsm['spatial'].copy()
            adata_z1.write_h5ad(
                os.path.join(self.output_dir, f'{output_prefix}_z1_{timestamp}.h5ad')
            )
            
            adata_z2 = sc.AnnData(z2)
            adata_z2.obs = adata2_new.obs.copy()
            if 'spatial' in adata2_new.obsm:
                adata_z2.obsm['spatial'] = adata2_new.obsm['spatial'].copy()
            adata_z2.write_h5ad(
                os.path.join(self.output_dir, f'{output_prefix}_z2_{timestamp}.h5ad')
            )
            
            adata_final = sc.AnnData(z_final)
            adata_final.obs = adata1_new.obs.copy()
            if 'spatial' in adata1_new.obsm:
                adata_final.obsm['spatial'] = adata1_new.obsm['spatial'].copy()
            adata_final.write_h5ad(
                os.path.join(self.output_dir, f'{output_prefix}_embeddings_{timestamp}.h5ad')
            )
            
            print(f"✓ Inference embeddings saved to {self.output_dir}")
        
        # Restore original data
        self.adata1 = original_adata1
        self.adata2 = original_adata2
        if original_preprocessed1 is not None:
            self.adata1_preprocessed = original_preprocessed1
        if original_preprocessed2 is not None:
            self.adata2_preprocessed = original_preprocessed2
        
        return z1, z2, z_final

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphEncoderWNNit(nn.Module):
    def __init__(self,nsample, in_channels1, in_channels2, hidden_channels):
        super(GraphEncoderWNNit, self).__init__()
        self.conv1 = GCNConv(in_channels1, hidden_channels, cached=False)
        self.conv2 = GCNConv(in_channels2, hidden_channels, cached=False)
        self.prelu = nn.PReLU(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.conv4 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.prelu2 = nn.PReLU(hidden_channels)
        self.prelu3 = nn.PReLU(hidden_channels)
        self.prelu4 = nn.PReLU(hidden_channels)
        self.mddim = hidden_channels

    def forward(self, x1, x2, edge_index, adata,w,w1,w2):

        x1 = self.conv1(x1, edge_index)
        x1 = self.prelu(x1)
        x1 = self.conv3(x1, edge_index)
        x1 = self.prelu2(x1)
        x1 = nn.functional.normalize(x1, p=2.0, dim=1)

        x2 = self.conv2(x2, edge_index)
        x2 = self.prelu3(x2)
        x2 = self.conv4(x2, edge_index)
        x2 = self.prelu4(x2)
        x2 = nn.functional.normalize(x2, p=2.0, dim=1)
        
        
        if w==1:
            pc1 = x1.detach().cpu().numpy()
            pc2 = x2.detach().cpu().numpy()
            adata.obsm['Omics1_PCA'] = pc1
            adata.obsm['Omics2_PCA'] = pc2
            WNNobj = pyWNN(adata, reps=['Omics1_PCA', 'Omics2_PCA'], npcs=[self.mddim,self.mddim], n_neighbors=20, seed=14)
            adata = WNNobj.compute_wnn(adata)
            ww=adata.obsm['Weights']
            ww = ww.astype(np.float32)
            w1 = torch.reshape(torch.from_numpy(ww[:,0]),(-1,1)).to(device)
            w2 = torch.reshape(torch.from_numpy(ww[:,1]),(-1,1)).to(device)
            
        else:
            w1 = w1
            w2 = w2
            
        x1 = x1 * w1
        x2 = x2 * w2
        x = x1 + x2
        return x,w1,w2




