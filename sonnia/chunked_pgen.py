"""
Drop-in replacement for parallel Pgen computation with chunking support.
"""

import multiprocessing as mp
import numpy as np
from typing import Sequence, List, Tuple, Optional, Callable
from olga.performance.fast_pgen import FastPgen


# Global variable for worker model (used with initializer)
_worker_model = None


def _init_worker(model: FastPgen):
    """Initialize worker process with the model (called once per worker)."""
    global _worker_model
    _worker_model = model


def _compute_pgen_chunk(chunk: List[Tuple[str, str, str]]) -> List[float]:
    """Compute Pgen for a chunk of sequences using the worker's model."""
    global _worker_model
    results = []
    for seq_tuple in chunk:
        if len(seq_tuple) == 3:
            # With V and J genes
            cdr3, v_gene, j_gene = seq_tuple
            pgen = _worker_model.compute_aa_CDR3_pgen(cdr3, v_gene, j_gene)
        elif len(seq_tuple) == 1:
            # Without V and J genes
            cdr3 = seq_tuple[0]
            pgen = _worker_model.compute_aa_CDR3_pgen(cdr3)
        else:
            raise ValueError(f"Invalid sequence tuple length: {len(seq_tuple)}")
        results.append(pgen)
    return results


def compute_all_pgens_chunked(
    sequences: Sequence[Sequence[str]],
    fast_pgen_model: FastPgen,
    chunk_size: int = 100,
    num_workers: Optional[int] = None,
    include_genes: bool = True
) -> np.ndarray:
    """Compute Pgen for sequences using optimized chunked parallel processing.
    
    This is a drop-in replacement for soNNia's compute_all_pgens that uses
    chunking to reduce pickling overhead. The model is pickled once per worker
    instead of once per sequence.
    
    Parameters
    ----------
    sequences : sequence of sequences
        List of sequences. Each sequence is either:
        - (CDR3_seq, V_gene, J_gene) if include_genes=True
        - (CDR3_seq,) if include_genes=False
    fast_pgen_model : FastPgen
        The FastPgen model instance to use
    chunk_size : int, optional
        Number of sequences per chunk (default: 100)
        Recommended values:
        - 50-100 for small sequences (< 20 AA)
        - 100-200 for medium sequences (20-30 AA)
        - 200-500 for large sequences (> 30 AA)
    num_workers : int, optional
        Number of worker processes (default: mp.cpu_count())
    include_genes : bool, optional
        Whether sequences include V and J genes (default: True)
        
    Returns
    -------
    np.ndarray
        Array of Pgen values, one per input sequence
        
    Examples
    --------
    >>> from olga.performance.fast_pgen import FastPgen
    >>> from olga.performance.chunked_pgen import compute_all_pgens_chunked
    >>> 
    >>> # Create your FastPgen model
    >>> fast_model = FastPgen(base_model)
    >>> 
    >>> # Prepare sequences
    >>> sequences = [
    ...     ("CAWSVAPDRGGYTF", "TRBV30", "TRBJ1-2"),
    ...     ("CASSQDRGQYF", "TRBV12", "TRBJ2-1"),
    ... ]
    >>> 
    >>> # Compute Pgens with chunking
    >>> pgens = compute_all_pgens_chunked(
    ...     sequences,
    ...     fast_model,
    ...     chunk_size=100,
    ...     num_workers=4
    ... )
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Convert sequences to list of tuples
    if include_genes:
        seq_tuples = [(seq[0], seq[1], seq[2]) for seq in sequences]
    else:
        seq_tuples = [(seq[0],) for seq in sequences]
    
    # Split into chunks
    chunks = []
    for i in range(0, len(seq_tuples), chunk_size):
        chunk = seq_tuples[i:i + chunk_size]
        chunks.append(chunk)
    
    # Process chunks in parallel with worker initializer
    # This ensures the model is pickled only once per worker
    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(fast_pgen_model,)
    ) as pool:
        chunk_results = pool.map(_compute_pgen_chunk, chunks)
    
    # Flatten results from chunks
    results = []
    for chunk_result in chunk_results:
        results.extend(chunk_result)
    
    return np.array(results)