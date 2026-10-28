# Script for the evaluation of caloirimeter showers using the GMetrics library.
















import argparse
import os
import h5py
import numpy as np
import sys 
import time
import pickle
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
sys.path.append('/teo_fs_fast/projects/GENOVA/CaloChallenge/CaloChallengeMetrics/') 
import GMetrics
sys.path.pop()
# Assuming HighLevelFeatures.py is in the same directory or Python path
sys.path.append('/teo_fs_fast/projects/GENOVA/CaloChallenge/CaloChallengeMetrics/CaloChallenge/code/')
import HighLevelFeatures as HLF

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

sys.path.pop()
from CaloChallenge.CaloChallengeMetrics.code.utils import BootstrapDistribution, PrefetchedKSTest, PrefetchedSKSTest, PrefetchedSWDMetric, PrefetchedMMDTest

def extract_shower_and_energy(file_handle):
    """Reads .hdf5 file and returns showers and incident energies."""
    showers = file_handle['showers'][:]
    energies = file_handle['incident_energies'][:]
    #do the shuffling for the events with random different seed for all submissions 
    #rng = np.random.default_rng(seed=None)
    #perm = rng.permutation(len(showers))
    #showers_shuffled = showers[perm]
    #energies_shuffled = energies[perm]
    return showers, energies

def dict_to_array(data_dict):
    if isinstance(data_dict, np.ndarray):
        print("Input is already a numpy array. Returning as is.")
        return data_dict
    elif isinstance(data_dict, dict):
        print("Input is a dictionary. Converting to numpy array.")
    elif isinstance(data_dict, list):
        print("Input is a list. Converting to numpy array.")
        return np.array(data_dict)
    
    # Assuming 'data_dict' is your dictionary
    # 1. Sort keys to ensure columns are in the correct order (0 to 44)
    sorted_keys = sorted(data_dict.keys())

    # 2. Stack the arrays as rows, then transpose to get shape (N, 45)
    # np.vstack creates (45, N), .T makes it (N, 45)
    matrix = np.vstack([data_dict[k] for k in sorted_keys]).T
    # shape (N,n_layer)
    return matrix

def compute_distributions(
    metrics_config: dict,
    unique_key: str,
    ref_bootstrap_obj,
    sub_bootstrap_obj,                 
    ndims: int,
    batch_size_test: int,
    niter: int = 1000,
    dtype=tf.float32,
    verbose: bool = False,
    cache_file: str | None = None,
    overwrite: bool = False,
    prefetched_classes: dict = None,
    metrics = None, 
    submission_file: bool = False,
):
    if prefetched_classes is None:
        prefetched_classes = {}

    if cache_file and (not overwrite) and os.path.isfile(cache_file):
        if verbose:
            print(f"[compute_alternative_distributions] Loaded cached results from: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f).get("results", {})

    AltInputs = GMetrics.TwoSampleTestInputs(
        dist_1_input    = ref_bootstrap_obj,
        dist_2_input    = sub_bootstrap_obj, 
        niter           = niter,
        batch_size_test = batch_size_test,
        small_sample_threshold = 1,
        dtype_input     = dtype,
        seed_input      = 42,
        use_tf          = True,
        mirror_strategy = False, 
        verbose         = verbose,
    )

    results = {}
    metric_items = metrics_config.get(unique_key, {}).items()

    for name, meta in metric_items:
        if name not in metrics:
            continue
        json_file = meta.get("null_file")
        
        # --- CACHE LOADING PHASE ---
        if json_file and os.path.isfile(json_file) and not overwrite:
            if verbose:
                print(f"[{name}] Found cached data at {json_file}. Loading...")
            with open(json_file, "r") as f:
                cached_data = json.load(f)
                results[name] = {
                    "alt_array"  if submission_file else "null_array": np.array(cached_data["alt_array"  if submission_file else "null_array"]),
                    "alt_mean"  if submission_file else "null_mean": cached_data["alt_mean"  if submission_file else "null_mean"],   
                    "alt_std"  if submission_file else "null_std": cached_data["alt_std"  if submission_file else "null_std"],
                    "time": cached_data["time"]
                }
            continue # Skip computation for this metric and move to the next

        # --- COMPUTATION PHASE ---
        
        lname = name.lower()
        MetricClass = prefetched_classes.get(lname)
        if MetricClass is None:
            try: MetricClass = eval(meta["class_name"])
            except (NameError, AttributeError): continue

        obj = MetricClass(
            AltInputs,
            progress_bar = verbose,
            **meta.get("kwargs", {})
        )

        test_fn = getattr(obj, "Test_tf", getattr(obj, "Test_np", None))
        
        call_kwargs = {}
        import inspect
        if "max_vectorize" in inspect.signature(test_fn).parameters:
            call_kwargs["max_vectorize"] = meta.get("max_vectorize", 100)
            
        if "gpu_dev" in inspect.signature(test_fn).parameters:
            call_kwargs["gpu_dev"] = meta.get("gpu_dev", "/device:GPU:0")
            
        if "cpu_dev" in inspect.signature(test_fn).parameters:
            call_kwargs["cpu_dev"] = meta.get("cpu_dev", "/CPU:0")

        t0 = time.time()
        test_fn(**call_kwargs)
        t1 = time.time()
        exec_time = t1 - t0

        result_key = meta["result_key"]
        
        raw_arr = np.array(obj.Results[0].result_value[result_key])
        for i in range(1, len(obj.Results)):
            raw_arr = np.concatenate((raw_arr, np.array(obj.Results[i].result_value[result_key])))

        ns_eff = batch_size_test / 2.0  
        scale_func = meta.get("scale_func")
        scale = scale_func(ns_eff, ndims) if callable(scale_func) else 1.0
        
        arr_scaled = raw_arr * scale

        total_time = sum(res.__dict__.get('computing_time', 0.0) for res in obj.Results)

        res_dict = {
            "alt_array"  if submission_file else "null_array": arr_scaled,
            "alt_mean"  if submission_file else "null_mean":  float(arr_scaled.mean()),
            "alt_std"  if submission_file else "null_std":   float(arr_scaled.std(ddof=1)) if arr_scaled.size > 1 else 0.0,
            "time":       total_time 
        }
        
        results[name] = res_dict
        
        # --- CACHE SAVING PHASE ---
        if json_file:
            # Create a copy for saving and convert the numpy array to a serializable list
            save_dict = res_dict.copy()
            save_dict["alt_array"  if submission_file else "null_array"] = save_dict["alt_array"  if submission_file else "null_array"].tolist()
            
            os.makedirs(os.path.dirname(json_file), exist_ok=True)
            with open(json_file, "w") as f:
                json.dump(save_dict, f, indent=4)
            
            if verbose:
                print(f"[{name}] Saved computed data to {json_file}")

        if verbose:
            print(f"[done] {name} in {t1 - t0:.2f}s")

    return results

def main():
    parser = argparse.ArgumentParser(description="Calorimeter Evaluation Skeleton Script")

    # Directories and Files
    parser.add_argument('--reference_file', type=str, required=True, help="Path to input .hdf5 file.")
    parser.add_argument('--submission_file', type=str, required=False, help="Path to submission .hdf5 file (if needed for full shower metrics).")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save output results.")
    
    # HLF specific arguments
    parser.add_argument('--particle_type', type=str, choices=['photon', 'pion', 'electron'], required=True, help="Particle type for HLF geometry.")
    parser.add_argument('--min_energy', type=float, default=0.01515, help="Minimum energy threshold for voxels.")
    parser.add_argument('--dataset', type=str, choices=['1', '2', '3'], required=True, help="Dataset name for HLF binning file selection.")

    # 1. Mode
    parser.add_argument('--mode', type=str, choices=['all', 'hlf', 'full_shower'], required=True)
    
    # 2. Metrics
    parser.add_argument('--metrics', type=str, nargs='+', choices=['ks', 'swd', 'sks', 'mmd', 'fgd'], required=True)
    
    # 3. Batchsizes
    parser.add_argument('--batchsizes', type=int, nargs='+', required=True)
    
    # 4. Small sample
    parser.add_argument('--small_sample', type=int, default=1)
    
    # 5. Verbose
    parser.add_argument('--verbose', action='store_true')
    
    # 6. Niter
    parser.add_argument('--niter', type=int, required=True)
    
    # 7. Max vectorize
    parser.add_argument('--max_vectorize', type=int, required=False)
    
    # 8. Nslices
    parser.add_argument('--nslices', type=int, required=True)

    # 9. Devices (CPU,GPU)
    parser.add_argument('--cpu', type=str, required=True, choices=["/CPU:0","/CPU:1","/CPU:2"], default="/CPU:0")
    parser.add_argument('--gpu', type=str, required=True, choices=["/device:GPU:0","/device:GPU:1","/device:GPU:2"], default="/device:GPU:0") 
    # 10. Overwrite option 
    parser.add_argument('--overwrite', action='store_true')
    
    
    
    args = parser.parse_args()

    
    



    # Assign parsed arguments to variables
    reference_file = args.reference_file
    submission_file = args.submission_file
    output_dir = args.output_dir
    particle_type = args.particle_type
    min_energy = args.min_energy
    
    mode = args.mode
    metrics = args.metrics
    batchsizes = args.batchsizes
    small_sample = args.small_sample
    verbose = args.verbose
    niter = args.niter
    nslices = args.nslices
    dataset = args.dataset
    cpu_dev = args.cpu
    gpu_dev = args.gpu
    max_vectorize = args.max_vectorize
    overwrite = args.overwrite

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if verbose:
        print(f"Loading reference data from {reference_file}...")

    if verbose:
        if submission_file:
            print(f"Loading submission data from {submission_file}...")

    # Load data
    with h5py.File(reference_file, 'r') as source_file:
        reference_showers, reference_incident_energies = extract_shower_and_energy(source_file)
    if submission_file:
        with h5py.File(submission_file, 'r') as submission_file_handle:
            submission_showers, submission_incident_energies = extract_shower_and_energy(submission_file_handle)
        # Apply noise thresholding (as done in evaluate.py)
        submission_showers = np.where(submission_showers < min_energy, np.zeros_like(submission_showers), submission_showers)
        
    reference_showers = np.where(reference_showers < min_energy, np.zeros_like(reference_showers), reference_showers)
    # --- High Level Features (HLF) Extraction ---
    if mode in ['all', 'hlf']:
        if verbose:
            print("Calculating High Level Features...")
            
        binning_path = '/teo_fs_fast/projects/GENOVA/CaloChallenge/CaloChallengeMetrics/CaloChallenge/code'
        filename = ''
        if particle_type == 'photon':
            if dataset != '1':
                raise ValueError(f"Dataset {dataset} not supported for photons. Only dataset 1 is available.")
            else:
                filename = binning_path + '/binning_dataset_1_photons.xml'
        elif particle_type == 'pion':
            if dataset != '1':
                raise ValueError(f"Dataset {dataset} not supported for photons. Only dataset 1 is available.")
            else:
                filename = binning_path + '/binning_dataset_1_pions.xml'
        elif particle_type == 'electron' and dataset == '2':
            filename = binning_path + '/binning_dataset_2.xml'
        elif particle_type == 'electron' and dataset == '3':
            filename = binning_path + '/binning_dataset_3.xml'
        else: 
            raise ValueError(f"Dataset {dataset} and particle {particle_type} not supported.")
        hlf = HLF.HighLevelFeatures(particle_type, filename=filename)
        
        if verbose:
            print("Calculating reference high level feature")

        # Calculate features (this populates internal dictionaries in the hlf object)
        hlf.CalculateFeatures(reference_showers)
        hlf.Einc = reference_incident_energies

        # Extract calculated quantities into Python variables
        # Note: Many of these are dictionaries keyed by layer ID.
        ref_e_inc      = hlf.Einc
        ref_e_tot      = hlf.GetEtot()
        ref_e_layers   = dict_to_array(hlf.GetElayers())
        ref_ec_r       = dict_to_array(hlf.GetECR())
        ref_ec_etas    = dict_to_array(hlf.GetECEtas())
        ref_ec_phis    = dict_to_array(hlf.GetECPhis())
        ref_width_r    = dict_to_array(hlf.GetWidthR())
        ref_width_etas = dict_to_array(hlf.GetWidthEtas())
        ref_width_phis = dict_to_array(hlf.GetWidthPhis())
        ref_sparsity   = dict_to_array(hlf.GetSparsity())
        
        
        
        if verbose:
            print("Calculating submission high level feature")
        if submission_file:
            hlf.CalculateFeatures(submission_showers)
            hlf.Einc = submission_incident_energies

            # Extract calculated quantities into Python variables
            # Note: Many of these are dictionaries keyed by layer ID.
            sub_e_inc      = hlf.Einc
            sub_e_tot      = hlf.GetEtot()
            sub_e_layers   = dict_to_array(hlf.GetElayers())
            sub_ec_r       = dict_to_array(hlf.GetECR())
            sub_ec_etas    = dict_to_array(hlf.GetECEtas())
            sub_ec_phis    = dict_to_array(hlf.GetECPhis())
            sub_width_r    = dict_to_array(hlf.GetWidthR())
            sub_width_etas = dict_to_array(hlf.GetWidthEtas())
            sub_width_phis = dict_to_array(hlf.GetWidthPhis())
            sub_sparsity   = dict_to_array(hlf.GetSparsity())

            if verbose:
                print("HLF Extraction Complete.")
                print(f"Total Showers Processed: {len(sub_e_tot)}")
            
        HLF_reusult_dir = os.path.join(output_dir, 'hlf_results')
        if not os.path.exists(HLF_reusult_dir):
            os.makedirs(HLF_reusult_dir)
            print(f"Created directory for HLF results: {HLF_reusult_dir}")

        features = ['e_inc', 'e_tot', 'e_layers', 'ec_r', 'ec_etas', 'ec_phis', 'width_r', 'width_etas', 'width_phis', 'sparsity']
        for feature in features:

            print(f"Processing feature: {feature}...")
            # Create a subdirectory for each feature to store results separately
            feature_output_dir = os.path.join(HLF_reusult_dir, f"feature_{feature}")
            if not os.path.exists(feature_output_dir):
                os.makedirs(feature_output_dir)
                print(f"Created directory for feature {feature}: {feature_output_dir}")

            ref_feature = 'ref_' + feature
            ref_feature_data = locals()[ref_feature]
            if submission_file:
                sub_feature = 'sub_' + feature
                sub_feature_data = locals()[sub_feature]

            # Reshape 1D arrays (N,) to 2D vectors (N, 1)
            # This makes e_tot look like a vector of size 1, compatible with the metrics library.
            if len(ref_feature_data.shape) == 1:
                ref_feature_data = ref_feature_data.reshape(-1, 1)
            if submission_file:
                if len(sub_feature_data.shape) == 1:
                    sub_feature_data = sub_feature_data.reshape(-1, 1)


            bootstrap_ref_feature = BootstrapDistribution(ref_feature_data, prefetch=True, dtype=tf.float32, device=cpu_dev)
            if submission_file:
                bootstrap_sub_feature = BootstrapDistribution(sub_feature_data, prefetch=True, dtype=tf.float32, device=cpu_dev)

            for batch in batchsizes:
                
                print(f"Processing batch size: {batch} for feature: {feature}...")

                batch_output_dir = os.path.join(feature_output_dir, f"batch_{batch}")
                if not os.path.exists(batch_output_dir):
                    os.makedirs(batch_output_dir)
                    print(f"Created directory for batch size {batch}: {batch_output_dir}")


                hypotheses_dir = os.path.join(batch_output_dir, f'alt_hypotheses_batch_{batch}' if submission_file else f'null_hypotheses_batch_{batch}' )
                if not os.path.exists(hypotheses_dir):
                    os.makedirs(hypotheses_dir)
                metrics_config_file = os.path.join(batch_output_dir, f"alt_metrics_config_batch_{batch}.json" if submission_file else f"metrics_config_batch_{batch}.json")
                #  --- Metric Evaluation ---
                prefetched_classes = {
                    "ks":  PrefetchedKSTest,
                    "sks": PrefetchedSKSTest,
                    "swd": PrefetchedSWDMetric,
                    "mmd": PrefetchedMMDTest
                }
                
                # Manually construct the configuration dictionary
                # This mimics the content that would be extracted from the TwoSampleTestInputs object
                test_config_alt = {
                    "niter": niter,
                    "batch_size_test": batch,
                    "batch_size_gen": 10000, # This is the standard pool size used in the original notebook 
                    "small_sample_threshold": small_sample,
                    "dtype_input": "float32",
                    "seed_input": 42,           # Using 0 as per the original notebook default
                    "use_tf": True,
                    "mirror_strategy": False,
                    "verbose": verbose
                }

                unique_key="HLF_metrics"
                metrics_config = {unique_key: {
                                                #"fgd": {"name": "fgd",
                                                #       "object_name": "FGDMetricNull",
                                                #       "class_name": "GMetrics.FGDMetric",
                                                #       "kwargs": {"num_batches": 20,
                                                #                  "num_points": 10,
                                                #                  "normalise": False,
                                                #                  "verbose": True}, 
                                                #       "result_key": "metric_list", 
                                                #       "scale_func": lambda ns, ndims: 1,
                                                #       "scale_func_string": "lambda ns, ndims: 1",
                                                #       "test_config": test_config_alt,
                                                #       "max_vectorize": 3200, # optimized for 5D
                                                #       "latex": "$t_{\mathrm{FGD}}$",
                                                #       "null_file": os.path.join(hypotheses_dir,"FGD.json")},
                                            "mmd": {"name": "mmd",
                                                    "object_name": "MMDMetricNull" ,
                                                    "class_name": "GMetrics.MMDMetric",
                                                    "kwargs": {"degree": 4, 
                                                                "block_size": 10_000, 
                                                                "normalise": False,
                                                                "verbose": True}, 
                                                    "result_key": "metric_list", 
                                                    "scale_func": lambda ns, ndims: 1,
                                                    "scale_func_string": "lambda ns, ndims: 1",
                                                    "test_config": test_config_alt,
                                                    "max_vectorize": 8, # optimized for 5D
                                                    "latex": "$t_{\mathrm{MMD}}$",
                                                    "null_file": os.path.join(hypotheses_dir,"MMD.json")},
                                            "ks":  {"name": "ks",
                                                    "gpu_dev": gpu_dev,
                                                    "cpu_dev": cpu_dev,
                                                    "object_name": "KSTestNull",
                                                    "class_name": "GMetrics.KSTest", 
                                                    "kwargs": {"verbose": verbose}, 
                                                    "result_key": "statistic_means", 
                                                    "scale_func": lambda ns, _ : np.sqrt(ns),
                                                    "scale_func_string": "lambda ns, _: np.sqrt(ns)",
                                                    "test_config": test_config_alt,
                                                    "max_vectorize": 1000,
                                                    "latex": "$t_{\overline{\mathrm{KS}}}$",
                                                    "null_file": os.path.join(hypotheses_dir,"KS.json")},
                                            "sks": {"name": "sks",
                                                    "gpu_dev": gpu_dev,
                                                    "cpu_dev": cpu_dev,
                                                    "object_name": "SKSTestNull",
                                                    "class_name": "GMetrics.SKSTest", 
                                                    "kwargs": {"nslices": nslices, 
                                                                "seed_slicing": 42,
                                                                "verbose": verbose}, 
                                                    "result_key": "metric_means", 
                                                    "scale_func": lambda ns, _ : np.sqrt(ns),
                                                    "scale_func_string": "lambda ns, _: np.sqrt(ns)",
                                                    "test_config": test_config_alt,
                                                    "max_vectorize": 10,
                                                    "latex": "$t_{\mathrm{SKS}}$",
                                                    "null_file": os.path.join(hypotheses_dir,"SKS.json")},
                                            "swd": {"name": "swd",
                                                    "gpu_dev": gpu_dev,
                                                    "cpu_dev": cpu_dev,
                                                    "object_name": "SWDMetricNull",
                                                    "class_name": "GMetrics.SWDMetric", 
                                                    "kwargs": {"nslices": nslices, 
                                                                "seed_slicing": 42,
                                                                "verbose": verbose}, 
                                                    "result_key": "metric_means", 
                                                    "scale_func": lambda ns, ndims: np.sqrt(ns/ndims),
                                                    "scale_func_string": "lambda ns, ndims: np.sqrt(ns/ndims)",
                                                    "test_config": test_config_alt,
                                                    "max_vectorize": 10,
                                                    "latex": "$t_{\mathrm{SW}}$",
                                                    "null_file": os.path.join(hypotheses_dir,"SWD.json")}
                                    }}
            

                #saving the configuration for reproducibility and later reference
                GMetrics.utils.save_update_metrics_config(metrics_config = metrics_config, metrics_config_file = metrics_config_file)


                # Here you would implement the logic to compute the specified metrics (ks, swd, sks) 
                # for each feature and batch size. This is a placeholder for where that logic would go.
                if submission_file:
                    Alt = True
                else:
                    Alt = False
                
                results = compute_distributions(
                    metrics_config=metrics_config, 
                    unique_key="HLF_metrics",
                    ref_bootstrap_obj=bootstrap_ref_feature,
                    sub_bootstrap_obj= bootstrap_sub_feature if submission_file else bootstrap_ref_feature, 
                    ndims=ref_feature_data.shape[1], 
                    batch_size_test=batch,
                    niter=niter,
                    dtype=tf.float32,
                    verbose=verbose,
                    overwrite=overwrite,
                    prefetched_classes=prefetched_classes,
                    metrics=metrics,
                    submission_file= Alt
                )

                # Define your Confidence Levels
                cl_list = [0.68, 0.95, 0.99]

                # Post-processing Loop
                print("Computing thresholds and updating config...")

                for name, data in results.items():
                    if name == "mmd":
                        continue
                    
                    # Retrieve the configuration for this specific metric
                    metric_conf = metrics_config[unique_key][name] 
                    
                    # Get the data we just computed
                    dist_t = data["alt_array"  if submission_file else "null_array"]
                    comp_time = data["time"]
                    
                    # --- Calculate Thresholds (Confidence Levels) ---
                    # We sort the array to find the value at the Xth percentile
                    sorted_dist = np.sort(dist_t)
                    n_points = len(sorted_dist)
                    
                    # Format: [CL, [Index_High, Index_Low], Value_at_CL]
                    metric_thresholds = []
                    for cl in cl_list:
                        idx = int(cl * n_points)
                        # Clamp index to avoid overflow
                        idx = min(idx, n_points - 1) 
                        
                        val = sorted_dist[idx]
                        metric_thresholds.append([cl, [idx, n_points - idx], float(val)])
                    
                    print(f"Metric {name}: Time={comp_time:.2f}s, Thresholds calculated.")

                    # --- Update the Config Object ---
                    metric_conf["thresholds"] = metric_thresholds
                    
                    # Note: Your supervisor's code summed time across results. 
                    # Since we run one big batch, the single 'comp_time' is the total time.
                    # If you want to save it to the config too (optional):
                    # metric_conf["computing_time"] = comp_time 

                # Save the updated config to file
                GMetrics.utils.save_update_metrics_config(
                    metrics_config=metrics_config, 
                    metrics_config_file=metrics_config_file # Make sure this path is defined in your args
                )
                print("Config file updated.")

        # Implement your HLF metric evaluation here using the extracted variables
        # Example: iterate through metrics list and pass relevant arrays to functions

    # --- Full Shower Processing ---
    if mode in ['all', 'full_shower']:
        
        print("Processing Full Shower metrics...")
        
        # Implement full voxel-level metrics here (e.g., mmd, fgd)
        # using the raw 'showers' array
        full_shower_result_dir = os.path.join(output_dir, 'full_shower_results')
        if not os.path.exists(full_shower_result_dir):
            os.makedirs(full_shower_result_dir)
            print(f"Created directory for Full Shower results: {full_shower_result_dir}")

        full_ref_shower = BootstrapDistribution(reference_showers, prefetch=True, dtype=tf.float32)
        if submission_file:
            full_sub_shower = BootstrapDistribution(submission_showers, prefetch=True, dtype=tf.float32)

        for batch in batchsizes:
                
            print(f"Processing batch size: {batch} for full shower metrics...")
            
            batch_output_dir = os.path.join(full_shower_result_dir, f"batch_{batch}")
            if not os.path.exists(batch_output_dir):
                os.makedirs(batch_output_dir)
                print(f"Created directory for batch size {batch}: {batch_output_dir}")


            hypotheses_dir = os.path.join(batch_output_dir,f'alt_hypotheses_batch_{batch}' if submission_file else f'null_hypotheses_batch_{batch}' )
            if not os.path.exists(hypotheses_dir):
                os.makedirs(hypotheses_dir)
            metrics_config_file = os.path.join(batch_output_dir, f"alt_metrics_config_batch_{batch}.json" if submission_file else f"metrics_config_batch_{batch}.json")
            #  --- Metric Evaluation ---
            prefetched_classes = {
                "ks":  PrefetchedKSTest,
                "sks": PrefetchedSKSTest,
                "swd": PrefetchedSWDMetric,
                "mmd": PrefetchedMMDTest
                }
            
            # Manually construct the configuration dictionary
            # This mimics the content that would be extracted from the TwoSampleTestInputs object
            test_config_alt = {
                "niter": niter,
                "batch_size_test": batch,
                "batch_size_gen": 100_000, # This is the standard pool size used in the original notebook 
                "small_sample_threshold": small_sample,
                "dtype_input": "float32",
                "seed_input": 42,           # Using 0 as per the original notebook default
                "use_tf": True,
                "mirror_strategy": False,
                "verbose": verbose
            }

            unique_key="HLF_metrics"
            metrics_config = {unique_key: {
                                            #"fgd": {"name": "fgd",
                                            #       "object_name": "FGDMetricNull",
                                            #       "class_name": "GMetrics.FGDMetric",
                                            #       "kwargs": {"num_batches": 20,
                                            #                  "num_points": 10,
                                            #                  "normalise": False,
                                            #                  "verbose": True}, 
                                            #       "result_key": "metric_list", 
                                            #       "scale_func": lambda ns, ndims: 1,
                                            #       "scale_func_string": "lambda ns, ndims: 1",
                                            #       "test_config": test_config_alt,
                                            #       "max_vectorize": 3200, # optimized for 5D
                                            #       "latex": "$t_{\mathrm{FGD}}$",
                                            #       "null_file": os.path.join(hypotheses_dir,"FGD.json")},
                                        "mmd": {"name": "mmd",
                                            "gpu_dev": gpu_dev,
                                            "cpu_dev": cpu_dev,
                                            "object_name": "MMDMetricNull" ,
                                            "class_name": "GMetrics.MMDMetric",
                                            "kwargs": {"degree": 4, 
                                                        "block_size": 10_000, 
                                                        "normalise": False,
                                                        "verbose": True},  
                                            "result_key": "metric_list", 
                                            "scale_func": lambda ns, ndims: 1,
                                            "scale_func_string": "lambda ns, ndims: 1",
                                            "test_config": test_config_alt,
                                            "max_vectorize": 8, # optimized for 5D
                                            "latex": "$t_{\mathrm{MMD}}$",
                                            "null_file": os.path.join(hypotheses_dir,"MMD.json")},
                                        "ks":  {"name": "ks",
                                                "gpu_dev": gpu_dev,
                                                "cpu_dev": cpu_dev,
                                                "object_name": "KSTestNull",
                                                "class_name": "GMetrics.KSTest", 
                                                "kwargs": {"verbose": verbose}, 
                                                "result_key": "statistic_means", 
                                                "scale_func": lambda ns, _ : np.sqrt(ns),
                                                "scale_func_string": "lambda ns, _: np.sqrt(ns)",
                                                "test_config": test_config_alt,
                                                "max_vectorize": 6480,                                         ## possibly add an argument to the arg parser for each case
                                                "latex": "$t_{\overline{\mathrm{KS}}}$",
                                                "null_file": os.path.join(hypotheses_dir,"KS.json")},
                                        "sks": {"name": "sks",
                                                "gpu_dev": gpu_dev,
                                                "cpu_dev": cpu_dev,
                                                "object_name": "SKSTestNull",
                                                "class_name": "GMetrics.SKSTest", 
                                                "kwargs": {"nslices": nslices, 
                                                            "seed_slicing": 42,
                                                            "verbose": verbose}, 
                                                "result_key": "metric_means", 
                                                "scale_func": lambda ns, _ : np.sqrt(ns),
                                                "scale_func_string": "lambda ns, _: np.sqrt(ns)",
                                                "test_config": test_config_alt,
                                                "max_vectorize": 10,
                                                "latex": "$t_{\mathrm{SKS}}$",
                                                "null_file": os.path.join(hypotheses_dir,"SKS.json")},
                                        "swd": {"name": "swd",
                                                "gpu_dev": gpu_dev,
                                                "cpu_dev": cpu_dev,
                                                "object_name": "SWDMetricNull",
                                                "class_name": "GMetrics.SWDMetric", 
                                                "kwargs": {"nslices": nslices, 
                                                            "seed_slicing": 42,
                                                            "verbose": verbose}, 
                                                "result_key": "metric_means", 
                                                "scale_func": lambda ns, ndims: np.sqrt(ns/ndims),
                                                "scale_func_string": "lambda ns, ndims: np.sqrt(ns/ndims)",
                                                "test_config": test_config_alt,
                                                "max_vectorize": 10,
                                                "latex": "$t_{\mathrm{SW}}$",
                                                "null_file": os.path.join(hypotheses_dir,"SWD.json")}
                                }}
        

            #saving the configuration for reproducibility and later reference
            GMetrics.utils.save_update_metrics_config(metrics_config = metrics_config, metrics_config_file = metrics_config_file)


            # Here you would implement the logic to compute the specified metrics (ks, swd, sks) 
            # for each feature and batch size. This is a placeholder for where that logic would go.
            if submission_file:
                Alt = True
            else:
                Alt = False
                
            
            results = compute_distributions(
                metrics_config=metrics_config, 
                unique_key="HLF_metrics",
                ref_bootstrap_obj=full_ref_shower, 
                sub_bootstrap_obj= full_sub_shower if submission_file else full_ref_shower,
                ndims=reference_showers.shape[1], 
                batch_size_test=batch,
                niter=niter,
                dtype=tf.float32,
                verbose=verbose,
                overwrite=overwrite,
                prefetched_classes=prefetched_classes,
                metrics=metrics,
                submission_file= Alt
            )

            # Define your Confidence Levels
            cl_list = [0.68, 0.95, 0.99]

            # Post-processing Loop
            print("Computing thresholds and updating config...")

            for name, data in results.items():
                # Retrieve the configuration for this specific metric
                metric_conf = metrics_config[unique_key][name] 
                
                # Get the data we just computed
                dist_t = data["alt_array"  if submission_file else "null_array"]
                comp_time = data["time"]
                
                # --- Calculate Thresholds (Confidence Levels) ---
                # We sort the array to find the value at the Xth percentile
                sorted_dist = np.sort(dist_t)
                n_points = len(sorted_dist)
                
                # Format: [CL, [Index_High, Index_Low], Value_at_CL]
                metric_thresholds = []
                for cl in cl_list:
                    idx = int(cl * n_points)
                    # Clamp index to avoid overflow
                    idx = min(idx, n_points - 1) 
                    
                    val = sorted_dist[idx]
                    metric_thresholds.append([cl, [idx, n_points - idx], float(val)])
                
                print(f"Metric {name}: Time={comp_time:.2f}s, Thresholds calculated.")

                # --- Update the Config Object ---
                metric_conf["thresholds"] = metric_thresholds
                
                # Note: Your supervisor's code summed time across results. 
                # Since we run one big batch, the single 'comp_time' is the total time.
                # If you want to save it to the config too (optional):
                # metric_conf["computing_time"] = comp_time 

            # Save the updated config to file
            GMetrics.utils.save_update_metrics_config(
                metrics_config=metrics_config, 
                metrics_config_file=metrics_config_file # Make sure this path is defined in your args
            )
            print("Config file updated.")

if __name__ == "__main__":
    main()