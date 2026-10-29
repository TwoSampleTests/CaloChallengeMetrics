


import sys
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
from typing import Union
sys.path.append("../")  # Adjust this path as needed to point to your project root
import GMetrics
import NysMMD






class PrefetchedMixin:
    """Mixin to add tf.data pipeline capabilities to GMetrics tests with Ordered Logging."""
    
    def _build_pipeline(self, max_vectorize, metric_compute_fn, gpu_device = '/device:GPU:0', cpu_device = '/CPU:0', k = None):
        import tensorflow as tf
        import math
        import sys
        from timeit import default_timer as timer
        
        ndims = self.Inputs.ndims
        niter = int(self.get_niter_batch_size_tf()[0])
        batch_size = int(self.get_niter_batch_size_tf()[1])
        
        max_vectorize = int(tf.maximum(max_vectorize, ndims))  # you always want to get at least ndims for each parallel calculation to do at least 1 calculation of the whole batch
        chunk_size = int(max_vectorize // ndims)              
        if chunk_size > niter: chunk_size = niter              
        if chunk_size < 1: chunk_size = 1                     
        nchunks = math.ceil(niter / chunk_size)                
        
        # --- LOGGING SETUP ---
        if self.verbose:
            print(f"nchunks = {nchunks}")
            print(f"Batch size per iteration: {batch_size}")
            print("Starting prefetch pipeline...")
            
            self._last_log_time = timer()

            def python_log_step(start, end):
                current_time = timer()
                dt = current_time - self._last_log_time
                self._last_log_time = current_time
                print(f"Iterating from {start} to {end} out of {niter}. Time elapsed: {dt:.4f} s")
                sys.stdout.flush() 
                return []

        # 1. FAST PATH FOR STATIC NUMERICAL DATA (Equivalence Testing) (use index slicing of numerical dist array)
        if hasattr(self.Inputs.dist_1_input, "shape") and not hasattr(self.Inputs.dist_1_input, "sample"):
            total_samples = nchunks * chunk_size * batch_size
            
            d1_flat = self.Inputs.dist_1_input[:total_samples]
            d2_flat = self.Inputs.dist_2_input[:total_samples]
            
            @tf.function
            def data_chunk(chunk_idx):
                with tf.device(cpu_device):
                    start_idx = chunk_idx * chunk_size * batch_size
                    end_idx = start_idx + chunk_size * batch_size
                    d1 = tf.reshape(d1_flat[start_idx:end_idx], [chunk_size, batch_size, ndims])
                    d2 = tf.reshape(d2_flat[start_idx:end_idx], [chunk_size, batch_size, ndims])
                    return d1, d2, chunk_idx
            
            ds = tf.data.Dataset.range(nchunks).map(data_chunk, num_parallel_calls=1)
            
        # 2. FAST PATH FOR SYMBOLIC SAMPLING (Production) (use .sample(...) on symb dist to produce numerical array)
        else:
            @tf.function
            def data_chunk(chunk_idx):
                with tf.device(cpu_device):
                    seed1 = tf.cast(self.Inputs.seed_generator.make_seeds(2)[0], tf.int32)
                    seed2 = tf.cast(self.Inputs.seed_generator.make_seeds(2)[0], tf.int32)
                    d1 = self.Inputs.dist_1_input.sample(chunk_size * batch_size, seed=seed1)  #chunking equals to instead of doing niter all together you do chunk_size at a time
                    d2 = self.Inputs.dist_2_input.sample(chunk_size * batch_size, seed=seed2)
                    d1 = tf.reshape(d1, [chunk_size, batch_size, ndims])
                    d2 = tf.reshape(d2, [chunk_size, batch_size, ndims])
                    return d1, d2, chunk_idx
            
            ds = tf.data.Dataset.range(nchunks).map(data_chunk, num_parallel_calls=1)

        # 3. SEQUENTIAL LOGGING WRAPPER
        def log_pass_through(d1, d2, chunk_idx):
            if self.verbose:
                start_iter = chunk_idx * chunk_size
                end_iter = tf.minimum(start_iter + chunk_size, niter)
                tf.py_function(func=python_log_step, inp=[start_iter, end_iter], Tout=[])
            return d1, d2
        
        # We map with num_parallel_calls=1 to keep printing strictly ordered
        ds = ds.map(log_pass_through, num_parallel_calls=1)

        # 4. COMPUTATION (Standard Graph Compilation to prevent searchsorted OOM)
        @tf.function()
        def fast_compute(d1, d2):
            if k is not None:
                return metric_compute_fn(d1, d2, gpu_device, cpu_device,k)
            else: 
                return metric_compute_fn(d1, d2, gpu_device, cpu_device)

        ds = ds.map(fast_compute, num_parallel_calls=1)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds, niter, chunk_size



class PrefetchedMMDTest(GMetrics.MMDMetric,PrefetchedMixin):     
    def Test_tf(self, max_vectorize: int = 100, gpu_dev='/device:GPU:0',cpu_dev = '/CPU:0') -> None:
        
        import tensorflow as tf
        import numpy as np
        import datetime
        import GMetrics
        from NysMMD import test
        from timeit import default_timer as timer
        from scipy.stats import iqr
        
        dtype = tf.as_dtype(self.Inputs.dtype)
        ndims = self.Inputs.ndims
        niter = int(self.get_niter_batch_size_tf()[0])
        batch_size = int(self.get_niter_batch_size_tf()[1])
        max_vectorize_calc = int(tf.maximum(max_vectorize, ndims))
        chunk_size = int(max_vectorize_calc // ndims)
        if chunk_size > niter: chunk_size = niter
        if chunk_size < 1: chunk_size = 1
        
        #auto_k: int = int(np.sqrt(2*batch_size)*2)  #if bs=100 --> k=28, bs=10000 -->k=282
        auto_k: int = min(batch_size, int(2 * np.sqrt(2 * batch_size)))
        
        def compute_mmd_nys_native(d1_chunk, d2_chunk, gpu_dev, cpu_dev, k=auto_k):
            

            with tf.device(gpu_dev):

                Z = tf.concat([d1_chunk, d2_chunk], axis=1)           # [C, 2b_s, d]
                b_s = tf.shape(d1_chunk)[1]
                sigma = test.median_heuristic_sigma_tf(Z, max_points=1000)
                
                
                inds = tf.random.shuffle(tf.range(tf.shape(Z)[1]))[:k]  
                psi_Z = test.nystrom_features_tf(Z, inds, sigma, multi_sigma=True)       # [C, 2b_s, k]

                psi_X = psi_Z[:, :b_s, :]                           # [C, b_s, k]
                psi_Y = psi_Z[:, b_s:, :]                           # [C, b_s, k]

                bar_psi_X = tf.reduce_mean(psi_X, axis=1)         # [C, k]
                bar_psi_Y = tf.reduce_mean(psi_Y, axis=1)         # [C, k]

                T = bar_psi_X - bar_psi_Y                         # [C, k]
                return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(T), axis=1), 0.0))         # [C]

        
        
        self._start = timer()
        ds, _, _ = self._build_pipeline(max_vectorize, compute_mmd_nys_native, gpu_device=gpu_dev, cpu_device= cpu_dev, k=auto_k)
        results = np.concatenate(list(ds.as_numpy_iterator()), axis=0)  # a list of results for MMD on each point in the Nth chunk then concatenated
        self._end = timer()
        if self.verbose:
            print(f"MMD tests calculation completed in {self._end - self._start:.4f} seconds.")
        metric_median = np.median(results) 
        metric_error = iqr(results,rng=(16.275, 83.725))/2
        res_val = {
            "metric_list": results,  #metric_list.append(np.median(vals_point))
            #"metric_error": metric_error #metric_error_list.append(iqr(vals_point, rng=(16.275, 83.725)) / 2)
        }
        self.Results.append(GMetrics.base.TwoSampleTestResult(datetime.datetime.now().isoformat(), "MMD Test_tf_prefetched", {**self.param_dict, "backend": "tf_prefetched"}, res_val))




class PrefetchedKSTest(GMetrics.KSTest, PrefetchedMixin):
    def Test_tf(self, max_vectorize: int = 100, gpu_dev='/device:GPU:0',cpu_dev = '/CPU:0') -> None:
        import tensorflow as tf
        import numpy as np
        import datetime
        import GMetrics
        from timeit import default_timer as timer

        dtype = tf.as_dtype(self.Inputs.dtype)
        ndims = self.Inputs.ndims
        niter = int(self.get_niter_batch_size_tf()[0])
        batch_size = int(self.get_niter_batch_size_tf()[1])
        max_vectorize_calc = int(tf.maximum(max_vectorize, ndims))
        chunk_size = int(max_vectorize_calc // ndims)
        if chunk_size > niter: chunk_size = niter
        if chunk_size < 1: chunk_size = 1

        def compute_ks_native(d1_chunk, d2_chunk, gpu_dev, cpu_dev):
            with tf.device(gpu_dev):
                d1_T = tf.transpose(d1_chunk, perm=[0, 2, 1]) 
                d2_T = tf.transpose(d2_chunk, perm=[0, 2, 1]) 
                
                n_tests = chunk_size * ndims
                d1_flat = tf.reshape(d1_T, [n_tests, batch_size])
                d2_flat = tf.reshape(d2_T, [n_tests, batch_size])
                
                d1_sorted = tf.sort(d1_flat, axis=-1)
                d2_sorted = tf.sort(d2_flat, axis=-1)
                
                data_all = tf.concat([d1_sorted, d2_sorted], axis=-1) ###tf.sort(tf.concat([d1_sorted, d2_sorted], axis=-1), axis=-1) #second sort useless(?) changed
                
                n1 = tf.cast(batch_size, dtype)
                n2 = tf.cast(batch_size, dtype)
                
                cdf1 = tf.cast(tf.searchsorted(d1_sorted, data_all, side='right'), dtype) / n1
                cdf2 = tf.cast(tf.searchsorted(d2_sorted, data_all, side='right'), dtype) / n2
                
                cddiffs = cdf1 - cdf2
                max_abs_diff = tf.reduce_max(tf.abs(cddiffs), axis=-1)
                
                en = (n1 * n2) / (n1 + n2)
                z = tf.sqrt(en) * max_abs_diff
                z_expanded = tf.expand_dims(z, -1) 
                
                z_expanded_64 = tf.cast(z_expanded, tf.float64)
                k_values = tf.range(-10, 11, dtype=tf.float64)
                
                terms = tf.pow(tf.cast(-1.0, tf.float64), k_values) * tf.exp(-2.0 * (k_values**2) * (z_expanded_64**2))
                prob = tf.clip_by_value(1.0 - tf.reduce_sum(terms, axis=-1), 0.0, 1.0)
                prob = tf.cast(prob, dtype) 
                
                stats = tf.reshape(max_abs_diff, [chunk_size, ndims])
                pvals = tf.reshape(prob, [chunk_size, ndims])
                
                return (tf.reduce_mean(stats, 1), tf.math.reduce_std(stats, 1), stats,
                        tf.reduce_mean(pvals, 1), tf.math.reduce_std(pvals, 1), pvals)

        self._start = timer()
        
        ds, _, _ = self._build_pipeline(max_vectorize, compute_ks_native, gpu_device=gpu_dev, cpu_device= cpu_dev)
        results = [list(x) for x in ds]
        
        self._end = timer()
        
        if self.verbose:
            print(f"KS tests calculation completed in {self._end - self._start:.4f} seconds.")
        
        cols = list(zip(*results)) 
        res_val = {
            "statistic_means": np.concatenate([x.numpy() for x in cols[0]]),
            "statistic_stds":  np.concatenate([x.numpy() for x in cols[1]]),
            "statistic_lists": np.concatenate([x.numpy() for x in cols[2]]),
            "pvalue_means":    np.concatenate([x.numpy() for x in cols[3]]),
            "pvalue_stds":     np.concatenate([x.numpy() for x in cols[4]]),
            "pvalue_lists":    np.concatenate([x.numpy() for x in cols[5]]),
        }
        self.Results.append(GMetrics.base.TwoSampleTestResult(datetime.datetime.now().isoformat(), "KS Test_tf_prefetched", {**self.param_dict, "backend": "tf_prefetched"}, res_val))





class PrefetchedSKSTest(GMetrics.SKSTest, PrefetchedMixin):
    def Test_tf(self, max_vectorize: int = 100, gpu_dev='/device:GPU:0', cpu_dev = '/CPU:0') -> None:
        import tensorflow as tf
        import numpy as np
        import datetime
        import GMetrics
        from timeit import default_timer as timer

        dtype = tf.as_dtype(self.Inputs.dtype)
        ndims = self.Inputs.ndims
        dirs = tf.cast(self.directions, dtype)
        
        niter = int(self.get_niter_batch_size_tf()[0])
        batch_size = int(self.get_niter_batch_size_tf()[1])
        max_vectorize_calc = int(tf.maximum(max_vectorize, ndims))
        chunk_size = int(max_vectorize_calc // ndims)
        if chunk_size > niter: chunk_size = niter
        if chunk_size < 1: chunk_size = 1

        def compute_sks_native(d1_chunk, d2_chunk, gpu_dev, cpu_dev):
            with tf.device(gpu_dev):
                p1 = tf.einsum('cbd,sd->cbs', d1_chunk, dirs)
                p2 = tf.einsum('cbd,sd->cbs', d2_chunk, dirs)
                
                p1_T = tf.transpose(p1, perm=[0, 2, 1])
                p2_T = tf.transpose(p2, perm=[0, 2, 1])
                
                n_tests = chunk_size * tf.shape(dirs)[0] 
                
                p1_flat = tf.reshape(p1_T, [n_tests, batch_size])
                p2_flat = tf.reshape(p2_T, [n_tests, batch_size])
                
                d1_sorted = tf.sort(p1_flat, axis=-1)
                d2_sorted = tf.sort(p2_flat, axis=-1)
                
                data_all = tf.sort(tf.concat([d1_sorted, d2_sorted], axis=-1), axis=-1)
                
                B_f = tf.cast(batch_size, dtype)
                
                cdf1 = tf.cast(tf.searchsorted(d1_sorted, data_all, side='right'), dtype) / B_f
                cdf2 = tf.cast(tf.searchsorted(d2_sorted, data_all, side='right'), dtype) / B_f
                
                cddiffs = cdf1 - cdf2
                max_abs_diff = tf.reduce_max(tf.abs(cddiffs), axis=-1)
                
                ks_proj = tf.reshape(max_abs_diff, [chunk_size, tf.shape(dirs)[0]])
                
                sks_mean = tf.reduce_mean(ks_proj, axis=1) 
                sks_std = tf.math.reduce_std(ks_proj, axis=1) 
                
                return sks_mean, sks_std, ks_proj

        self._start = timer()
        
        ds, _, _ = self._build_pipeline(max_vectorize, compute_sks_native, gpu_device=gpu_dev, cpu_device= cpu_dev)
        results = [list(x) for x in ds]
        
        self._end = timer()
        
        if self.verbose:
            print(f"SKS tests calculation completed in {self._end - self._start:.4f} seconds.")
        
        cols = list(zip(*results))
        res_val = {
            "metric_means": np.concatenate([x.numpy() for x in cols[0]]),
            "metric_stds":  np.concatenate([x.numpy() for x in cols[1]]),
            "metric_lists": np.concatenate([x.numpy() for x in cols[2]]),
        }
        self.Results.append(GMetrics.base.TwoSampleTestResult(datetime.datetime.now().isoformat(), "SKS Test_tf_prefetched", {**self.param_dict, "backend": "tf_prefetched"}, res_val))



class PrefetchedSWDMetric(GMetrics.SWDMetric, PrefetchedMixin):
    def Test_tf(self, max_vectorize: int = 100, gpu_dev='/device:GPU:0', cpu_dev = '/CPU:0') -> None:
        import tensorflow as tf
        import numpy as np
        import datetime
        import GMetrics
        from timeit import default_timer as timer

        dtype = tf.as_dtype(self.Inputs.dtype)
        ndims = self.Inputs.ndims
        dirs = tf.cast(self.directions, dtype)
        
        niter = int(self.get_niter_batch_size_tf()[0])
        batch_size = int(self.get_niter_batch_size_tf()[1])
        max_vectorize_calc = int(tf.maximum(max_vectorize, ndims))
        
        chunk_size = int(max_vectorize_calc // ndims)
        if chunk_size > niter: chunk_size = niter
        if chunk_size < 1: chunk_size = 1

        def compute_swd_native(d1_chunk, d2_chunk, gpu_dev, cpu_dev):
            """
            Input: [Chunk, Batch, Dim]
            Output: SWD Stats [Chunk] (Mean, Std)
            """
            with tf.device('/device:GPU:0'):
                # 1. Project Data onto Slices simultaneously
                p1 = tf.einsum('cbd,sd->cbs', d1_chunk, dirs)
                p2 = tf.einsum('cbd,sd->cbs', d2_chunk, dirs)
                
                # 2. Sort along the Batch dimension (axis=1)
                p1_sorted = tf.sort(p1, axis=1)
                p2_sorted = tf.sort(p2, axis=1)
                
                # 3. Compute 1-Wasserstein distance per slice (Mean Absolute Error)
                abs_diff = tf.abs(p1_sorted - p2_sorted)
                
                # Reduce along Batch dimension -> [Chunk, Slices]
                wd_proj = tf.reduce_mean(abs_diff, axis=1)
                
                # 4. Aggregate across slices -> [Chunk]
                swd_mean = tf.reduce_mean(wd_proj, axis=1)
                swd_std = tf.math.reduce_std(wd_proj, axis=1)
                
                return swd_mean, swd_std, wd_proj

        self._start = timer()
        
        ds, _, _ = self._build_pipeline(max_vectorize, compute_swd_native, gpu_device=gpu_dev, cpu_device= cpu_dev)
        results = [list(x) for x in ds]
        
        self._end = timer()
        
        if getattr(self, "verbose", False):
            print(f"SWD tests calculation completed in {self._end - self._start:.4f} seconds.")
        
        cols = list(zip(*results))
        res_val = {
            "metric_means": np.concatenate([x.numpy() for x in cols[0]]),
            "metric_stds":  np.concatenate([x.numpy() for x in cols[1]]),
            "metric_lists": np.concatenate([x.numpy() for x in cols[2]]),
        }
        
        self.Results.append(GMetrics.base.TwoSampleTestResult(
            datetime.datetime.now().isoformat(), 
            "SWD Test_tf_prefetched", 
            {**self.param_dict, "backend": "tf_prefetched"}, 
            res_val
        ))



class BootstrapDistribution(tfd.Distribution):
    def __init__(self, data: Union[np.ndarray, tf.Tensor], 
                 prefetch: bool = False,   
                 device: str = None,           
                 dtype: tf.DType = tf.float32, 
                 validate_args: bool = False, 
                 allow_nan_stats: bool = True, 
                 name: str = "BootstrapDistribution"):
        """
        Args:
            data: The dataset (numpy or tensor).
            prefetch: If True, forces data to CPU (for prefetch pipelines).
                      If False, defaults to GPU (for standard XLA execution).
            device: Explicit device string (e.g., "/CPU:0" or "/GPU:0"). 
                    Overrides 'prefetch' logic if provided.
        """
        parameters = dict(locals())
        
        # --- Device Selection Logic ---
        if device is not None:
            self._device = device
        elif prefetch:
            # Prefetching requires CPU storage to allow async transfer to GPU
            self._device = "/CPU:0"
        else:
            # Standard/Legacy mode: Put data on GPU to allow XLA compilation
            # Check if GPU is available, otherwise fall back to CPU
            gpus = tf.config.list_physical_devices('GPU')
            self._device = "/GPU:0" if gpus else "/CPU:0"
            
        print(f"BootstrapDistribution initialized on: {self._device}")

        # --- Data Loading ---
        with tf.device(self._device):
            # FIX: Use tf.Variable. This keeps the data mutable/external and prevents 
            # the graph from exploding in size.
            data_tensor = tf.convert_to_tensor(data, dtype=dtype)
            self._data = tf.Variable(data_tensor, trainable=False, name="bootstrap_data")
        
        self._nsamples = tf.shape(self._data)[0]
        self._ndims = tf.shape(self._data)[1]
        self._dtype = dtype
        
        super(BootstrapDistribution, self).__init__(
            dtype=dtype,
            reparameterization_type=tfd.NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name
        )
    
    def _batch_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)
    
    def _batch_shape(self):
        return tf.TensorShape([])
    
    def _event_shape_tensor(self):
        return tf.constant([self._ndims], dtype=tf.int32)
    
    def _event_shape(self):
        return tf.TensorShape([self._data.shape[1]])
    
    def _sample_n(self, n, seed=42):
        if seed is None:
            seed = tf.random.uniform([2], maxval=2**31-1, dtype=tf.int32)
        elif isinstance(seed, int):
            seed = tf.constant([seed, 0], dtype=tf.int32)
        else:
            seed = tf.convert_to_tensor(seed, dtype=tf.int32)
            seed_flat = tf.reshape(seed, [-1])
            seed = tf.concat([seed_flat, tf.zeros([2], dtype=tf.int32)], axis=0)[:2]
        
        # Ensure sampling happens on the same device as the data
        E_bins = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]
        
        with tf.device(self._device):
            indices = tf.random.stateless_uniform(
                shape=[n],
                seed=seed,
                minval=0,
                maxval=self._nsamples,
                dtype=tf.int32
            )
            # tf.gather on a Variable is efficient
            
            samples = tf.gather(self._data, indices)
            D = self._data.shape[1] if self._data.shape[1] is not None else self._ndims
            samples.set_shape([None, D])
            return samples
    
    def _log_prob(self, value):
        return tf.fill(tf.shape(value)[:-1], -tf.math.log(tf.cast(self._nsamples, self._dtype)))






