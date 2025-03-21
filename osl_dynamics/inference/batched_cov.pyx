# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np
from scipy import fft
from scipy import linalg
cimport cython
from libc.math cimport sqrt

# Import LAPACK functions for direct Cholesky decomposition
from scipy.linalg.cython_lapack cimport dpotrf

# Define the C types for numpy arrays
ctypedef np.float64_t DTYPE_t
np.import_array()

# Helper function to get the index in the vectorized upper triangular matrix
cdef inline int get_vec_idx(int i, int j, int N) noexcept nogil:
    # If i > j, swap them to ensure we're in the upper triangle
    if i > j:
        i, j = j, i
    # Formula for the index in the vectorized upper triangular matrix
    return (N*(N+1))//2 - ((N-i)*(N-i+1))//2 + j - i

# Helper function to convert vectorized upper triangular matrix to full matrix
cdef void vec_to_full_matrix(DTYPE_t* vec_data, DTYPE_t* full_data, int N) noexcept nogil:
    cdef int i, j, vec_idx
    
    for i in range(N):
        for j in range(i, N):
            vec_idx = get_vec_idx(i, j, N)
            full_data[j*N + i] = vec_data[vec_idx]
            if i != j:
                full_data[i*N + j] = vec_data[vec_idx]  # Mirror for symmetry

# Compute the covariance for a batch using FFT-based convolution, directly vectorizing the result
cdef np.ndarray[DTYPE_t, ndim=2] compute_cov_fft_vectorized(np.ndarray[DTYPE_t, ndim=2] batch, int M):
    """
    Given a batch of data (shape (L, N)), compute the sliding-window covariance
    for windows of length M and directly store the result in vectorized upper triangular form.
    
    This handles proper covariance calculation by removing the product of means:
    cov(X,Y) = E[XY] - E[X]E[Y]
    
    Returns an array of shape (L - M + 1, N*(N+1)//2).
    """
    cdef int L = batch.shape[0]
    cdef int N = batch.shape[1]
    cdef int L_out = L - M + 1
    cdef int L_conv = L + M - 1  # length needed for full convolution
    cdef int vec_size = (N*(N+1))//2

    if L_out <= 0:
        # Return empty array with correct shape for 0 windows but proper column count
        return np.empty((0, (N*(N+1))//2), dtype=np.float64)
    
    # Allocate output for vectorized covariance matrices
    cdef np.ndarray[DTYPE_t, ndim=2] covs_vec = np.empty((L_out, vec_size), dtype=np.float64)
    
    # Precompute the FFT of the ones kernel for convolution
    cdef np.ndarray[DTYPE_t, ndim=1] kernel = np.ones(M, dtype=np.float64)
    cdef np.ndarray kernel_fft = fft.rfft(kernel, n=L_conv)
    
    # Arrays for products and means
    cdef np.ndarray[DTYPE_t, ndim=1] prod_array = np.empty(L, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] means = np.empty((L_out, N), dtype=np.float64)
    cdef np.ndarray prod_fft, conv_result
    cdef double mean_kernel_factor = 1.0 / M
    cdef int i, j, k, vec_idx

    cdef double unbiased_factor = 1.0 * M / (M-1)
    
    # First compute the sliding means for each channel
    for i in range(N):
        # FFT of the channel data
        prod_fft = fft.rfft(batch[:, i], n=L_conv)
        # Multiply by kernel_fft in frequency domain
        prod_fft *= kernel_fft
        # Inverse FFT to obtain sliding sum
        conv_result = fft.irfft(prod_fft, n=L_conv)
        # Extract valid part and normalize to get means
        means[:, i] = conv_result[M-1:M-1+L_out] * mean_kernel_factor
    
    # Compute E[XY] for each pair of channels
    for i in range(N):
        for j in range(i, N):  # Only compute upper triangle
            # Get the vector index for this pair
            vec_idx = get_vec_idx(i, j, N)
            
            # Compute the pointwise product of channels i and j
            for k in range(L):
                prod_array[k] = batch[k, i] * batch[k, j]
            
            # FFT of the product
            prod_fft = fft.rfft(prod_array, n=L_conv)
            
            # Multiply by kernel_fft in frequency domain
            prod_fft *= kernel_fft
            
            # Inverse FFT to obtain sliding sum of products (E[XY])
            conv_result = fft.irfft(prod_fft, n=L_conv)
            
            # Extract valid part, normalize, and subtract E[X]E[Y] to get covariance
            for k in range(L_out):
                covs_vec[k, vec_idx] = ((conv_result[M-1+k] * mean_kernel_factor) - (means[k, i] * means[k, j]))*unbiased_factor
    
    return covs_vec

# Perform in-place Cholesky decomposition using LAPACK's dpotrf
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int cholesky_decomp_inplace(DTYPE_t* A_data, int N) noexcept nogil:
    """
    Perform in-place Cholesky decomposition on matrix A (overwriting it)
    Returns 0 if successful, positive integer i if the i-th leading minor is not positive definite
    """
    cdef int info = 0
    cdef char uplo = b'L'  # Lower triangular (Fortran ordering)
    cdef int i, j
    
    # Call LAPACK's dpotrf
    dpotrf(&uplo, &N, A_data, &N, &info)
    
    # Zero out the upper triangular part (above diagonal)
    if info == 0:
        for i in range(N):
            for j in range(i+1, N):
                A_data[j*N + i] = 0.0
    
    return info

# Perform Cholesky decomposition on vectorized covariance matrices
@cython.boundscheck(False)
@cython.wraparound(False)
def compute_cholesky(np.ndarray[DTYPE_t, ndim=2] covs_vec, int N):
    """
    Compute Cholesky decomposition for a batch of vectorized covariance matrices.
    
    Parameters:
      covs_vec : 2D numpy array of shape (L, N*(N+1)//2) containing vectorized upper triangular
                 covariance matrices.
      N : int, dimension of each covariance matrix.
      
    Returns:
      chol_matrices : 3D numpy array of shape (L, N, N) containing the Cholesky factors.
                     If a matrix is not positive definite, its corresponding entry will be filled with NaNs.
    """
    cdef int L = covs_vec.shape[0]
    cdef int vec_size = covs_vec.shape[1]
    
    # Preallocate output for Cholesky factors (3D array)
    cdef np.ndarray[DTYPE_t, ndim=2] chol_matrices = np.empty((L, int(N*(N+1)/2)), dtype=np.float64)
    
    # Preallocate a single matrix for computations (reused for each covariance matrix)
    cdef np.ndarray[DTYPE_t, ndim=2] cov_matrix = np.empty((N, N), dtype=np.float64, order='F')
    
    cdef int i, j, k, info
    cdef DTYPE_t* cov_matrix_data = <DTYPE_t*>cov_matrix.data
    
    for i in range(L):
        # Convert vectorized representation to full matrix
        vec_to_full_matrix(<DTYPE_t*>&covs_vec[i, 0], cov_matrix_data, N)
        
        # Perform in-place Cholesky decomposition
        info = cholesky_decomp_inplace(cov_matrix_data, N)
        
        if info == 0:
            # Copy the result to output array
            for j in range(N):
                for k in range(j+1):
                    chol_matrices[i, j*(j+1)//2 + k] = cov_matrix[j, k]
        else:
            # If not positive definite, fill with NaNs
            for j in range(N):
                for k in range(N):
                    chol_matrices[i, j*(j+1)//2 + k] = np.nan
    
    return chol_matrices

def batched_covariance_noalloc(np.ndarray[DTYPE_t, ndim=2] x, int M, int B):
    """
    Compute sliding-window covariance directly in vectorized upper triangular form
    in batches, using FFT-based convolution. Batches are processed with an overlap
    of O = M-1 samples to avoid boundary effects.
    
    Parameters:
      x : 2D numpy array of shape (T, N)  -- the input data.
      M : int, the window length (cov[t] uses x[t:t+M]).
      B : int, batch size (must be >= M).
    
    Returns:
      out : 2D numpy array of shape ((T - M + 1), (N*(N+1)//2)) containing the vectorized
            upper triangular parts of the computed covariance matrices.
    """
    if B < M:
        raise ValueError("Batch size B must be >= window size M")
    
    cdef int T = x.shape[0]
    cdef int N = x.shape[1]
    cdef int L_total = T - M + 1
    cdef int vec_size = (N*(N+1))//2

    # Preallocate final output array
    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((L_total, vec_size), dtype=np.float64)

    # Overlap to ensure full windows are computed
    cdef int O = M - 1
    cdef int i = 0              # global start index for current batch
    cdef int batch_end          # end index for current batch (not inclusive)
    cdef int L_batch            # number of valid windows in current batch
    cdef int valid_start, valid_end, num_valid
    cdef int out_index = 0      # index into the final output array

    cdef np.ndarray[DTYPE_t, ndim=2] batch, batch_result
    cdef int vec_idx

    while i < T:
        batch_end = min(i + B, T)
        
        # Use a slice (view) of x; no new allocation is made
        batch = x[i:batch_end]

        # Skip this batch if it has fewer than M elements
        if batch_end - i < M:
            break
        
        # Compute covariances for this batch (directly vectorized)
        batch_result = compute_cov_fft_vectorized(batch, M)
        L_batch = batch_result.shape[0]
        
        # Determine the valid windows to avoid boundary effects
        if i == 0 and batch_end < T:
            valid_start = 0
            valid_end = L_batch - O
        elif batch_end == T and i > 0:
            valid_start = O
            valid_end = L_batch
        elif i == 0 and batch_end == T:
            valid_start = 0
            valid_end = L_batch
        else:
            valid_start = O
            valid_end = L_batch - O
        valid_end = max(valid_start,valid_end)
        num_valid = valid_end - valid_start
        if num_valid > 0:
            # Copy the valid windows into the preallocated output
            out[out_index:out_index+num_valid] = batch_result[valid_start:valid_end]
            out_index += num_valid
            
        # Advance the batch start by (B - O) to maintain the overlap
        i += (B - O)

    if out_index < L_total:
        return out[:out_index]
    return out

def batched_covariance_and_cholesky(np.ndarray[DTYPE_t, ndim=2] x, int M, int B):
    """
    Compute sliding-window covariance matrices and their Cholesky decompositions
    in batches, using FFT-based convolution.
    
    Parameters:
      x : 2D numpy array of shape (T, N)  -- the input data.
      M : int, the window length (cov[t] uses x[t:t+M]).
      B : int, batch size (must be >= M).
    
    Returns:
      covs : 2D numpy array of shape ((T - M + 1), (N*(N+1)//2)) containing the vectorized
             upper triangular parts of the computed covariance matrices.
      chols : 3D numpy array of shape ((T - M + 1), N, N) containing the Cholesky factors.
    """
    if B < M:
        raise ValueError("Batch size B must be >= window size M")
    
    cdef int T = x.shape[0]
    cdef int N = x.shape[1]
    cdef int L_total = T - M + 1
    cdef int vec_size = (N*(N+1))//2

    # Preallocate final output arrays
    cdef np.ndarray[DTYPE_t, ndim=2] covs_out = np.empty((L_total, vec_size), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] chols_out = np.empty((L_total, N*(N+1)//2), dtype=np.float64)


    cdef int batch_end
    cdef int L_batch
    cdef int valid_start
    cdef int valid_end 
    cdef int num_valid

    # Overlap between batches to ensure continuous coverage
    cdef int O = M - 1
    cdef int effective_batch_size = B - O  # The number of unique samples per batch after accounting for overlap
    
    # Process data in batches, but ensure we generate outputs for every valid window
    cdef int out_index = 0  # Position in the output arrays
    
    cdef int batch_start = 0  # Start index in the original array
    cdef np.ndarray[DTYPE_t, ndim=2] batch, batch_cov_result, batch_chol_result

    while batch_start < T - M + 1:  # Continue as long as we can form at least one valid window
        # Calculate the end of this batch (limited by array size)
        batch_end = min(batch_start + B, T)
        
        # Skip this batch if it doesn't have enough elements for a full window
        if batch_end - batch_start < M:
            break
            
        # Extract the current batch (view, not copy)
        batch = x[batch_start:batch_end]
        
        # Compute covariances for this batch
        batch_cov_result = compute_cov_fft_vectorized(batch, M)
        
        # Number of valid windows in this batch
        L_batch = batch_cov_result.shape[0]
        
        # Determine which outputs to keep from this batch
        valid_start = 0
        valid_end = L_batch
        
        # For all batches except the first, skip the first (M-1) windows
        # as they overlap with the previous batch
        #if batch_start > 0:
        #    valid_start = O
        
        # Ensure valid_start doesn't exceed valid_end
        valid_start = min(valid_start, valid_end)
        
        # Calculate number of valid windows
        num_valid = valid_end - valid_start
        
        if num_valid > 0:
            # Only process if there are valid windows
            batch_valid_cov = batch_cov_result[valid_start:valid_end]
            
            # Compute Cholesky decomposition
            batch_chol_result = compute_cholesky(batch_valid_cov, N)
            
            # Copy results to output arrays
            covs_out[out_index:out_index+num_valid] = batch_valid_cov
            chols_out[out_index:out_index+num_valid] = batch_chol_result
            
            # Update output index
            out_index += num_valid
        
        # Move to next batch, advancing by effective_batch_size
        batch_start += effective_batch_size
    
    # If we didn't fill the entire output arrays, trim them
    if out_index < L_total:
        return covs_out[:out_index], chols_out[:out_index]
    
    return covs_out, chols_out