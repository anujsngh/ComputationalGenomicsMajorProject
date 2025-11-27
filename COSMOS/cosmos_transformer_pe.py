'''
Graph Transformer Encoder with Laplacian Positional Encoding for COSMOS
'''
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import get_laplacian, to_dense_adj
import numpy as np
import warnings


def compute_laplacian_pe(edge_index, num_nodes, pe_dim=8, normalization='sym', device=None):
    """
    Compute Laplacian Positional Encoding using eigenvectors.
    
    ROBUST VERSION: Handles edge cases, disconnected graphs, numerical issues
    
    Args:
        edge_index: Graph connectivity [2, num_edges]
        num_nodes: Number of nodes in graph
        pe_dim: Number of eigenvectors to use (default: 8)
        normalization: 'sym' for symmetric normalization (default)
        device: Target device for PE tensor
        
    Returns:
        pe: Positional encoding [num_nodes, pe_dim]
    """
    # Ensure we have enough eigenvectors
    if pe_dim >= num_nodes:
        warnings.warn(f"pe_dim ({pe_dim}) >= num_nodes ({num_nodes}). Reducing to {num_nodes-1}")
        pe_dim = max(1, num_nodes - 1)
    
    try:
        # Get normalized Laplacian
        edge_index_lap, edge_weight = get_laplacian(
            edge_index, 
            normalization=normalization, 
            num_nodes=num_nodes
        )
        
        # Convert to dense matrix for eigendecomposition
        L = to_dense_adj(
            edge_index_lap, 
            edge_attr=edge_weight, 
            max_num_nodes=num_nodes
        )[0]
        
        # Move to CPU for eigendecomposition (more stable)
        L_cpu = L.cpu()
        
        # Eigendecomposition: L = V @ Λ @ V^T
        eigenvalues, eigenvectors = torch.linalg.eigh(L_cpu)
        
        # Take k smallest non-trivial eigenvectors
        # Skip first eigenvector (constant vector for connected components)
        # Use eigenvectors 1 to pe_dim (corresponding to smallest eigenvalues)
        if num_nodes > pe_dim + 1:
            pe = eigenvectors[:, 1:pe_dim+1]
        else:
            # Handle small graphs
            pe = eigenvectors[:, 1:]
            # Pad if necessary
            if pe.shape[1] < pe_dim:
                padding = torch.zeros(num_nodes, pe_dim - pe.shape[1])
                pe = torch.cat([pe, padding], dim=1)
        
        # Handle sign ambiguity: make first element positive
        sign = torch.sign(pe[0, :])
        sign[sign == 0] = 1
        pe = pe * sign.unsqueeze(0)
        
        # Normalize each eigenvector
        pe = pe / (torch.norm(pe, dim=0, keepdim=True) + 1e-8)
        
        # Move to target device if specified
        if device is not None:
            pe = pe.to(device)
        
    except Exception as e:
        warnings.warn(f"Laplacian eigendecomposition failed: {e}. Using random PE.")
        # Fallback to random encoding
        pe = torch.randn(num_nodes, pe_dim)
        pe = pe / (torch.norm(pe, dim=0, keepdim=True) + 1e-8)
        if device is not None:
            pe = pe.to(device)
    
    return pe


class GraphTransformerEncoderWNNit_PE(nn.Module):
    """
    Graph Transformer encoder with Laplacian Positional Encoding.
    PRODUCTION VERSION - Fully tested and robust
    
    Key features:
    - Topology-aware: Encodes graph structure explicitly via Laplacian PE
    - Position-sensitive: Can distinguish different spatial arrangements
    - Global attention: Captures long-range dependencies
    - Proven WNN fusion: Unchanged from original COSMOS
    
    Args:
        nsample: Number of samples (cells)
        in_channels1: Input feature dimension for modality 1
        in_channels2: Input feature dimension for modality 2
        hidden_channels: Output embedding dimension
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
        pe_dim: Dimension of positional encoding (default: 8)
        use_pe: Whether to use positional encoding (default: True)
    """
    def __init__(self, nsample, in_channels1, in_channels2, hidden_channels, 
                 num_heads=8, dropout=0.1, pe_dim=8, use_pe=True):
        super(GraphTransformerEncoderWNNit_PE, self).__init__()
        
        self.nsample = nsample
        self.mddim = hidden_channels
        self.num_heads = num_heads
        self.pe_dim = pe_dim
        self.use_pe = use_pe
        self.dropout = dropout
        
        # Positional encoding projection layers
        if use_pe:
            # Project PE to a learnable space before concatenation
            self.pe_proj = nn.Linear(pe_dim, pe_dim)
            self.pe_norm = nn.LayerNorm(pe_dim)
        
        # Adjust input dimensions to account for PE
        effective_in_channels1 = in_channels1 + pe_dim if use_pe else in_channels1
        effective_in_channels2 = in_channels2 + pe_dim if use_pe else in_channels2
        
        # Modality 1 pathway (e.g., RNA)
        self.trans1 = TransformerConv(
            effective_in_channels1,
            hidden_channels, 
            heads=num_heads,
            concat=False,
            dropout=dropout,
            edge_dim=None,
            beta=True
        )
        self.prelu1 = nn.PReLU(hidden_channels)
        
        self.trans3 = TransformerConv(
            hidden_channels,
            hidden_channels,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            edge_dim=None,
            beta=True
        )
        self.prelu2 = nn.PReLU(hidden_channels)
        
        # Modality 2 pathway (e.g., Protein/ATAC)
        self.trans2 = TransformerConv(
            effective_in_channels2,
            hidden_channels,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            edge_dim=None,
            beta=True
        )
        self.prelu3 = nn.PReLU(hidden_channels)
        
        self.trans4 = TransformerConv(
            hidden_channels,
            hidden_channels,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            edge_dim=None,
            beta=True
        )
        self.prelu4 = nn.PReLU(hidden_channels)
        
        # Cache for PE (computed once per graph structure)
        self.pe_cache = None
        self.cached_edge_index = None
        self.cache_device = None

    def get_positional_encoding(self, edge_index, num_nodes):
        """
        Get or compute positional encoding with caching.
        
        FIXED v1.1: Caches raw eigenvectors, applies projection freshly each time.
        FIXED v1.2: Proper device handling
        
        This prevents "backward through graph a second time" errors.
        
        Args:
            edge_index: Graph connectivity [2, num_edges]
            num_nodes: Number of nodes
            
        Returns:
            pe: Positional encoding [num_nodes, pe_dim]
        """
        if not self.use_pe:
            return None
        
        device = edge_index.device
        
        # Check if we can use cached PE
        # We need to check device too, in case model was moved
        cache_valid = (
            self.pe_cache is not None and 
            self.cached_edge_index is not None and
            self.cache_device == device and
            edge_index.shape == self.cached_edge_index.shape and
            torch.equal(edge_index.cpu(), self.cached_edge_index.cpu())  # Compare on CPU to avoid device issues
        )
        
        if cache_valid:
            # Use cached raw PE
            raw_pe = self.pe_cache
            # Ensure it's on the right device
            if raw_pe.device != device:
                raw_pe = raw_pe.to(device)
        else:
            # Compute new PE
            raw_pe = compute_laplacian_pe(edge_index, num_nodes, self.pe_dim, device=device)
            # Cache the RAW eigenvectors (constants, no gradients)
            self.pe_cache = raw_pe.detach().clone()  # Detach and clone for safety
            self.cached_edge_index = edge_index.detach().clone().cpu()  # Store on CPU
            self.cache_device = device
        
        # ALWAYS APPLY PROJECTION (creates fresh computation graph for each forward pass)
        # This is CRITICAL to avoid "backward through graph" errors
        pe = self.pe_proj(raw_pe)
        pe = self.pe_norm(pe)
        
        return pe

    def forward(self, x1, x2, edge_index, adata, w, w1, w2):
        """
        Forward pass with positional encoding.
        
        Args:
            x1: Feature matrix for modality 1 [n_cells, n_features1]
            x2: Feature matrix for modality 2 [n_cells, n_features2]
            edge_index: Graph connectivity [2, n_edges]
            adata: AnnData object (for WNN computation)
            w: Flag to compute WNN weights (1=compute, 0=use existing)
            w1, w2: Existing weights for modality 1 and 2
            
        Returns:
            x: Fused embedding [n_cells, hidden_channels]
            w1: Updated weight for modality 1
            w2: Updated weight for modality 2
        """
        device = x1.device
        
        # Get positional encoding
        if self.use_pe:
            pe = self.get_positional_encoding(edge_index, x1.size(0))
            
            # Concatenate PE to features
            x1 = torch.cat([x1, pe], dim=1)
            x2 = torch.cat([x2, pe], dim=1)
        
        # Modality 1 encoding
        x1 = self.trans1(x1, edge_index)
        x1 = self.prelu1(x1)
        x1 = self.trans3(x1, edge_index)
        x1 = self.prelu2(x1)
        x1 = nn.functional.normalize(x1, p=2.0, dim=1)

        # Modality 2 encoding
        x2 = self.trans2(x2, edge_index)
        x2 = self.prelu3(x2)
        x2 = self.trans4(x2, edge_index)
        x2 = self.prelu4(x2)
        x2 = nn.functional.normalize(x2, p=2.0, dim=1)
        
        # WNN fusion (keeping COSMOS's proven mechanism)
        if w == 1:
            try:
                from .pyWNN import pyWNN
            except ImportError:
                from pyWNN import pyWNN
            
            # Move to CPU for WNN computation
            pc1 = x1.detach().cpu().numpy()
            pc2 = x2.detach().cpu().numpy()
            adata.obsm['Omics1_PCA'] = pc1
            adata.obsm['Omics2_PCA'] = pc2
            
            # Compute WNN weights
            WNNobj = pyWNN(adata, reps=['Omics1_PCA', 'Omics2_PCA'], 
                          npcs=[self.mddim, self.mddim], n_neighbors=20, seed=14)
            adata = WNNobj.compute_wnn(adata)
            ww = adata.obsm['Weights']
            ww = ww.astype(np.float32)
            
            # Convert to tensors on correct device
            w1 = torch.from_numpy(ww[:, 0]).reshape(-1, 1).to(device)
            w2 = torch.from_numpy(ww[:, 1]).reshape(-1, 1).to(device)
        
        # Weighted fusion
        x1 = x1 * w1
        x2 = x2 * w2
        x = x1 + x2
        
        return x, w1, w2


class GraphTransformerEncoderWNNit_PE_Deep(nn.Module):
    """
    Deeper 3-layer Graph Transformer with Positional Encoding.
    PRODUCTION VERSION - For advanced users who need more capacity
    
    Use this if:
    - You have complex, heterogeneous tissue
    - Standard version underfits (high training loss)
    - You have >20K cells
    
    Otherwise, use the standard 2-layer version above.
    """
    def __init__(self, nsample, in_channels1, in_channels2, hidden_channels, 
                 num_heads=8, dropout=0.1, pe_dim=8, use_pe=True):
        super(GraphTransformerEncoderWNNit_PE_Deep, self).__init__()
        
        self.nsample = nsample
        self.mddim = hidden_channels
        self.num_heads = num_heads
        self.pe_dim = pe_dim
        self.use_pe = use_pe
        self.dropout = dropout
        
        # Positional encoding projection
        if use_pe:
            self.pe_proj = nn.Linear(pe_dim, pe_dim)
            self.pe_norm = nn.LayerNorm(pe_dim)
        
        effective_in_channels1 = in_channels1 + pe_dim if use_pe else in_channels1
        effective_in_channels2 = in_channels2 + pe_dim if use_pe else in_channels2
        
        # Modality 1: 3-layer transformer
        self.trans1_layers = nn.ModuleList([
            TransformerConv(effective_in_channels1, hidden_channels, heads=num_heads, 
                          concat=False, dropout=dropout, beta=True),
            TransformerConv(hidden_channels, hidden_channels, heads=num_heads,
                          concat=False, dropout=dropout, beta=True),
            TransformerConv(hidden_channels, hidden_channels, heads=num_heads,
                          concat=False, dropout=dropout, beta=True)
        ])
        self.prelu1_layers = nn.ModuleList([
            nn.PReLU(hidden_channels) for _ in range(3)
        ])
        
        # Modality 2: 3-layer transformer
        self.trans2_layers = nn.ModuleList([
            TransformerConv(effective_in_channels2, hidden_channels, heads=num_heads,
                          concat=False, dropout=dropout, beta=True),
            TransformerConv(hidden_channels, hidden_channels, heads=num_heads,
                          concat=False, dropout=dropout, beta=True),
            TransformerConv(hidden_channels, hidden_channels, heads=num_heads,
                          concat=False, dropout=dropout, beta=True)
        ])
        self.prelu2_layers = nn.ModuleList([
            nn.PReLU(hidden_channels) for _ in range(3)
        ])
        
        self.pe_cache = None
        self.cached_edge_index = None
        self.cache_device = None

    def get_positional_encoding(self, edge_index, num_nodes):
        """Get or compute positional encoding with caching."""
        if not self.use_pe:
            return None
        
        device = edge_index.device
        
        cache_valid = (
            self.pe_cache is not None and 
            self.cached_edge_index is not None and
            self.cache_device == device and
            edge_index.shape == self.cached_edge_index.shape and
            torch.equal(edge_index.cpu(), self.cached_edge_index.cpu())
        )
        
        if cache_valid:
            raw_pe = self.pe_cache
            if raw_pe.device != device:
                raw_pe = raw_pe.to(device)
        else:
            raw_pe = compute_laplacian_pe(edge_index, num_nodes, self.pe_dim, device=device)
            self.pe_cache = raw_pe.detach().clone()
            self.cached_edge_index = edge_index.detach().clone().cpu()
            self.cache_device = device
        
        # ALWAYS apply projection (fresh computation graph)
        pe = self.pe_proj(raw_pe)
        pe = self.pe_norm(pe)
        
        return pe

    def forward(self, x1, x2, edge_index, adata, w, w1, w2):
        """Forward pass for deep version."""
        device = x1.device
        
        # Get positional encoding
        if self.use_pe:
            pe = self.get_positional_encoding(edge_index, x1.size(0))
            x1 = torch.cat([x1, pe], dim=1)
            x2 = torch.cat([x2, pe], dim=1)
        
        # Modality 1 - 3 layers
        for trans, prelu in zip(self.trans1_layers, self.prelu1_layers):
            x1 = trans(x1, edge_index)
            x1 = prelu(x1)
        x1 = nn.functional.normalize(x1, p=2.0, dim=1)
        
        # Modality 2 - 3 layers
        for trans, prelu in zip(self.trans2_layers, self.prelu2_layers):
            x2 = trans(x2, edge_index)
            x2 = prelu(x2)
        x2 = nn.functional.normalize(x2, p=2.0, dim=1)
        
        # WNN fusion
        if w == 1:
            try:
                from .pyWNN import pyWNN
            except ImportError:
                from pyWNN import pyWNN
            
            pc1 = x1.detach().cpu().numpy()
            pc2 = x2.detach().cpu().numpy()
            adata.obsm['Omics1_PCA'] = pc1
            adata.obsm['Omics2_PCA'] = pc2
            WNNobj = pyWNN(adata, reps=['Omics1_PCA', 'Omics2_PCA'],
                          npcs=[self.mddim, self.mddim], n_neighbors=20, seed=14)
            adata = WNNobj.compute_wnn(adata)
            ww = adata.obsm['Weights']
            ww = ww.astype(np.float32)
            w1 = torch.from_numpy(ww[:, 0]).reshape(-1, 1).to(device)
            w2 = torch.from_numpy(ww[:, 1]).reshape(-1, 1).to(device)
        
        x1 = x1 * w1
        x2 = x2 * w2
        x = x1 + x2
        
        return x, w1, w2
