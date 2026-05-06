
import os
import gc
import time
import threading
import re

try:
    from contextlib import nullcontext
except Exception:
    class nullcontext:
        def __init__(self): pass

        def __enter__(self): return self

        def __exit__(self, *args): return False
import psutil
import humanize
import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp
import pyarrow as pa
from pyarrow import parquet as pq
from pathlib import Path
from functools import partial
from tqdm import tqdm
import shutil

# ========== HDF5 environment variables for high-memory servers ==========
os.environ['HDF5_DEFAULT_RDCC_NBYTES'] = str(4 * 1024 * 1024 * 1024)  # 4GB
os.environ['HDF5_DEFAULT_RDCC_NSLOTS'] = '100003'
os.environ['HDF5_METADATA_CACHE_SIZE'] = str(2 * 1024 * 1024 * 1024)  # 2GB
os.environ['HDF5_DEFAULT_RDCC_W0'] = '0.5'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['HDF5_NTHREADS'] = '16'
# ===========================================================

# ========== I/O reliability configuration ==========
ENABLE_LOCAL_CACHE = False
CACHE_STRATEGY = 'on_retry'
MAX_CACHE_FILE_SIZE_GB = 8

MONTH_CACHE_ENABLED = True
MONTH_CACHE_STRATEGY = 'on_retry'

# ========== Torch/GPU configuration ==========
ENABLE_TORCH = True  # Enable the GPU path; automatically fall back to CPU when torch is unavailable
TORCH_MAX_POINTS_PER_BLOCK = int(os.environ.get('TORCH_MAX_POINTS_PER_BLOCK', '16000'))  # Block size can be adjusted through an environment variable
TORCH_PREFERRED_DEVICES = os.environ.get('TORCH_PREFERRED_DEVICES', 'auto')  # 'auto' or 'cuda:0,cuda:1'
# Monitoring configuration; enabled automatically when CUDA is detected unless explicitly set
_RAW_TORCH_ENABLE_MONITOR = os.environ.get('TORCH_ENABLE_MONITOR')  # '1' enables, '0' disables, None uses automatic detection
TORCH_ENABLE_MONITOR = False  # Placeholder; finalized after torch import
TORCH_MONITOR_INTERVAL = float(os.environ.get('TORCH_MONITOR_INTERVAL', '0.5'))
TORCH_MONITOR_CSV_DIR = os.environ.get('TORCH_MONITOR_CSV_DIR', '')  # If empty, the default directory is assigned in main

TORCH_MONITOR_INTERVAL = float(os.environ.get('TORCH_MONITOR_INTERVAL', '0.5'))
TORCH_MONITOR_CSV_DIR = os.environ.get('TORCH_MONITOR_CSV_DIR', '')  # If non-empty, monthly CSV files are written

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# Optional NVML monitoring for GPU utilization and memory use
try:
    import pynvml  # type: ignore

    HAS_NVML = True
except Exception:
    HAS_NVML = False


def calculate_correction_factor_gpu(T_kelvin):
    """Compute the temperature correction factor for NumPy arrays or Torch tensors."""
    # Detect the input array type
    is_torch = HAS_TORCH and isinstance(T_kelvin, torch.Tensor)
    
    if is_torch:
        T_celsius = T_kelvin - 273.15
        T_REF = 25.0
        T_NEG_20 = -20.0
        T_NEG_40 = -40.0
        T_POS_80 = 80.0

        f_T = torch.zeros_like(T_celsius)
        mask_high = T_celsius > T_POS_80
        f_T[mask_high] = -0.22
        mask_mid = (T_celsius > T_NEG_20) & (T_celsius <= T_POS_80)
        f_T[mask_mid] = -0.004 * (T_celsius[mask_mid] - T_REF)
        mask_low = (T_celsius >= T_NEG_40) & (T_celsius <= T_NEG_20)
        f_T[mask_low] = -0.003 * (T_celsius[mask_low] - T_NEG_20) + 0.18
        mask_very_low = T_celsius < T_NEG_40
        f_T[mask_very_low] = 0.24
    else:
        # NumPy path
        T_celsius = T_kelvin - 273.15
        T_REF = 25.0
        T_NEG_20 = -20.0
        T_NEG_40 = -40.0
        T_POS_80 = 80.0

        f_T = np.zeros_like(T_celsius)
        mask_high = T_celsius > T_POS_80
        f_T[mask_high] = -0.22
        mask_mid = (T_celsius > T_NEG_20) & (T_celsius <= T_POS_80)
        f_T[mask_mid] = -0.004 * (T_celsius[mask_mid] - T_REF)
        mask_low = (T_celsius >= T_NEG_40) & (T_celsius <= T_NEG_20)
        f_T[mask_low] = -0.003 * (T_celsius[mask_low] - T_NEG_20) + 0.18
        mask_very_low = T_celsius < T_NEG_40
        f_T[mask_very_low] = 0.24
    
    return f_T



class GPUMonitor:
    """Sample GPU utilization and memory use in the background, then report peak and mean values.

    NVML monitoring is disabled automatically when pynvml or CUDA is unavailable.
    Sampling records can optionally be written to CSV by setting csv_path.
    """

    def __init__(self, device_ids, interval=0.5, csv_path=None):
        self.device_ids = [int(d) for d in device_ids]
        self.interval = interval
        self.csv_path = csv_path
        self._stop = threading.Event()
        self._thread = None
        self.enabled = HAS_NVML and len(self.device_ids) > 0
        self.samples = []  # (ts, dev, util_gpu, util_mem, mem_used_MB, mem_total_MB)

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            pynvml.nvmlInit()
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.device_ids]
        except Exception:
            self.enabled = False
            return self

        def _loop():
            while not self._stop.is_set():
                ts = time.time()
                for i, h in zip(self.device_ids, self.handles):
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(h)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                        self.samples.append(
                            (ts, i, util.gpu, util.memory, mem.used / (1024 * 1024), mem.total / (1024 * 1024)))
                    except Exception:
                        pass
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        # Summary output
        if not self.samples:
            return
        by_dev = {}
        for s in self.samples:
            _, dev, util_gpu, util_mem, mem_used_mb, mem_total_mb = s
            by_dev.setdefault(dev, {'gpu': [], 'mem': [], 'used': [], 'total': mem_total_mb})
            by_dev[dev]['gpu'].append(util_gpu)
            by_dev[dev]['mem'].append(util_mem)
            by_dev[dev]['used'].append(mem_used_mb)
        print("\n===== GPU Monitoring (NVML) Summary =====")
        for dev, d in sorted(by_dev.items()):
            avg_gpu = sum(d['gpu']) / max(len(d['gpu']), 1)
            peak_gpu = max(d['gpu']) if d['gpu'] else 0
            peak_used = max(d['used']) if d['used'] else 0
            total = d['total']
            print(
                f"cuda:{dev} | mean utilization: {avg_gpu:.1f}% | peak utilization: {peak_gpu:.1f}% | peak memory: {peak_used:.1f} / {total:.1f} MB")
        if self.csv_path:
            try:
                import csv
                with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(["timestamp", "device", "util_gpu_%", "util_mem_%", "mem_used_MB", "mem_total_MB"])
                    for s in self.samples:
                        w.writerow(s)
                print(f"GPU monitoring details written to: {self.csv_path}")
            except Exception as e:
                print(f"Warning: failed to write GPU monitoring CSV: {e}")


# Automatically determine monitoring status from CUDA availability unless explicitly configured
try:
    if _RAW_TORCH_ENABLE_MONITOR is None:
        TORCH_ENABLE_MONITOR = bool(HAS_TORCH and torch.cuda.is_available())
    else:
        TORCH_ENABLE_MONITOR = (_RAW_TORCH_ENABLE_MONITOR == '1')
except Exception:
    TORCH_ENABLE_MONITOR = False

# ========== Small-scale trial-run configuration through environment variables ==========
GPU_TEST_YEARS = os.environ.get('GPU_TEST_YEARS', '2024')  # Example: '2020' or '2020,2021'
GPU_TEST_MONTHS = os.environ.get('GPU_TEST_MONTHS', '')  # Example: '1' or '1,7,12'
GPU_TEST_MAX_POINTS = int(os.environ.get('GPU_TEST_MAX_POINTS', '0'))  # 0 means no limit


def ensure_local_copy(src_path: str, cache_root: Path, year: int, log_file_path: Path) -> str:
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        src = Path(src_path)
        dst = cache_root / src.name
        try:
            if dst.exists() and src.exists() and dst.stat().st_size == src.stat().st_size:
                return str(dst)
        except Exception:
            pass
        try:
            if src.exists() and src.stat().st_size > MAX_CACHE_FILE_SIZE_GB * (1024 ** 3):
                safe_log_write(log_file_path, f"[{year}] Skipped local cache because the file is larger than {MAX_CACHE_FILE_SIZE_GB} GB: {src}")
                return str(src)
        except Exception:
            pass
        tmp_dst = dst.with_suffix(dst.suffix + ".part")
        if tmp_dst.exists():
            try:
                tmp_dst.unlink()
            except Exception:
                pass
        copy_ok = False
        try:
            shutil.copy2(str(src), str(tmp_dst))
            tmp_dst.replace(dst)
            copy_ok = True
        except Exception as e:
            print(f"[cache] Warning: failed to copy to local cache: {e}")
            safe_log_write(log_file_path, f"[{year}] Local cache copy failed: {e}")
        if copy_ok:
            print(f"[cache] Cached locally: {dst}")
            safe_log_write(log_file_path, f"[{year}] Cached data locally: {dst}")
            return str(dst)
        else:
            return str(src)
    except Exception as e:
        print(f"[cache] Warning: local cache workflow failed: {e}")
        return src_path


def get_native_time_chunk(nc_path: str, engine: str = "h5netcdf", default_chunk: int = 168) -> int:
    """
    Read the native chunk size along the valid_time dimension from a NetCDF file.
    Return the default value if the chunk metadata cannot be obtained.
    """
    try:
        with xr.open_dataset(nc_path, engine=engine, chunks=None, decode_times=False) as ds:
            # Try to obtain chunk sizes from the primary variables
            for var_name in ['ssrd', 'aluvp', 'aluvd', 't2m', 'u10', 'v10']:
                if var_name in ds.data_vars:
                    cs = ds[var_name].encoding.get('chunksizes', None)
                    if cs is not None and isinstance(cs, (tuple, list)) and len(cs) >= 1:
                        return int(cs[0])
            # If none are found, inspect the first data variable
            for var_name in ds.data_vars:
                cs = ds[var_name].encoding.get('chunksizes', None)
                if cs is not None and isinstance(cs, (tuple, list)) and len(cs) >= 1:
                    return int(cs[0])
    except Exception:
        pass
    return default_chunk


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return f"RSS: {humanize.naturalsize(process.memory_info().rss)}"


def safe_log_write(log_file_path, message, max_retries=3, retry_delay=0.5):
    for attempt in range(max_retries):
        try:
            with open(log_file_path, 'a', encoding='utf-8', buffering=1) as log_f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                log_f.write(f"[{timestamp}] {message}\n")
                log_f.flush()
                os.fsync(log_f.fileno())
            return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                print(f"Warning: log write failed because the file is locked: {e}")
                return False
        except Exception as e:
            print(f"Warning: log write failed: {e}")
            return False
    return False


def check_missing_months(final_output_path, year):
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(final_output_path)
        expected_points = 313013
        actual_points = len(parquet_file.schema.names) - 1
        if actual_points != expected_points:
            print(f"Error: file {final_output_path.name} has an unexpected number of site columns!")
            print(f"   Actual: {actual_points} columns / expected: {expected_points} columns")
            return None
        num_records = parquet_file.metadata.num_rows
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        expected_min = 8760 if not is_leap else 8784
        if num_records < expected_min - 100:
            print(f"Warning: file {final_output_path.name} has too few time records: {num_records}/{expected_min}")
            return None
        df_existing = pd.read_parquet(final_output_path)
        if not isinstance(df_existing.index, pd.DatetimeIndex):
            try:
                df_existing.index = pd.to_datetime(df_existing.index)
            except:
                print(f"Warning: file {final_output_path.name} has an unparseable time index")
                return None
        existing_months = set(df_existing.index.month.unique())
        expected_months = set(range(1, 13))
        missing_months = sorted(list(expected_months - existing_months))
        if missing_months:
            print(f"Summary: file {final_output_path.name} is missing months: {missing_months}")
        return missing_months
    except Exception as e:
        print(f"Warning: error while checking file {final_output_path}: {e}")
        return None


def update_log_mark_fixed(log_file_path, year, fixed_months):
    try:
        fixed_months_str = [f"{year}-{m:02d}" for m in sorted(fixed_months)]
        fixed_msg = f"Success: [{year}] missing months completed: {', '.join(fixed_months_str)}"
        safe_log_write(log_file_path, fixed_msg)
    except Exception as e:
        print(f"Warning: failed to update the log marker: {e}")


def merge_months_to_existing(final_output_path, new_data_dict, year, log_file_path):
    try:
        df_existing = pd.read_parquet(final_output_path)
        new_dfs = [new_data_dict[m] for m in sorted(new_data_dict.keys())]
        if new_dfs:
            df_new = pd.concat(new_dfs, axis=0)
            df_combined = pd.concat([df_existing, df_new], axis=0).sort_index()
            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
            df_combined.to_parquet(final_output_path, engine='pyarrow')
            merged_months = sorted(new_data_dict.keys())
            msg = f"Success: merged months {merged_months} into {final_output_path.name}"
            print(msg)
            safe_log_write(log_file_path, f"[{year}] {msg}")
            update_log_mark_fixed(log_file_path, year, merged_months)
            return True
    except Exception as e:
        msg = f"Warning: failed to merge monthly data: {e}"
        print(msg)
        safe_log_write(log_file_path, f"[{year}] {msg}")
        return False


def calculate_cf_for_chunk(ds_chunk, year, month, content_land_site):
    if ds_chunk is None or len(ds_chunk.valid_time) == 0:
        raise ValueError(f"[{year}-{month:02d}] The data block is empty or has zero time steps")
    if 'ssrd' not in ds_chunk.data_vars:
        raise ValueError(f"[{year}-{month:02d}] Missing required variable 'ssrd'")
    if 'aluvp' not in ds_chunk.data_vars or 'aluvd' not in ds_chunk.data_vars:
        raise ValueError(f"[{year}-{month:02d}] Missing required variable 'aluvp' or 'aluvd'")
    ssrd_valid = ds_chunk.ssrd.dropna(dim='valid_time', how='all')
    if len(ssrd_valid) == 0:
        raise ValueError(f"[{year}-{month:02d}] The ssrd array contains only NaN values and cannot be processed")

    time_coords = ds_chunk.valid_time
    n_expanded = time_coords.dt.dayofyear
    hours_expanded = time_coords.dt.hour
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        n_expanded = n_expanded - (n_expanded >= 60).astype(int)
    theta_d = 23.45 * np.pi / 180 * np.sin(2 * np.pi * (284 + n_expanded) / 365)
    E_qt = xr.zeros_like(n_expanded, dtype=float)

    # E_qt = E_qt.where(n_expanded >= 107, -14.2 * np.sin(np.pi * (n_expanded + 7) / 111))
    # E_qt = E_qt.where(n_expanded <= 246, 16.4 * np.sin(np.pi * (n_expanded - 247) / 113))
    # mask1 = (n_expanded > 106) & (n_expanded < 167)
    # E_qt = E_qt.where(~mask1, 4.0 * np.sin(np.pi * (n_expanded - 106) / 59))
    # mask2 = (n_expanded > 166) & (n_expanded < 247)
    # E_qt = E_qt.where(~mask2, -6.5 * np.sin(np.pi * (n_expanded - 166) / 80))

    E_qt = (
    (n_t < 107).to(torch_dtype) * (-14.2 * torch.sin(pi * (n_t + 7) / 111))
    + (n_t > 246).to(torch_dtype) * (16.4 * torch.sin(pi * (n_t - 247) / 113))
    + ((n_t > 106) & (n_t < 167)).to(torch_dtype) * (4.0 * torch.sin(pi * (n_t - 106) / 59))
    + ((n_t > 166) & (n_t < 247)).to(torch_dtype) * (-6.5 * torch.sin(pi * (n_t - 166) / 80))
)
    T_solar = hours_expanded + 0.5 + E_qt / 60 + ds_chunk.longitude / 15
    theta_hr = np.pi * (T_solar - 12) / 12
    lat_rad = np.deg2rad(ds_chunk.latitude)
    sin_lat_b = np.sin(lat_rad).broadcast_like(ds_chunk.ssrd)
    cos_lat_b = np.cos(lat_rad).broadcast_like(ds_chunk.ssrd)
    sin_delta_b = np.sin(theta_d).broadcast_like(ds_chunk.ssrd)
    cos_delta_b = np.cos(theta_d).broadcast_like(ds_chunk.ssrd)
    cos_theta_hr_b = np.cos(theta_hr).broadcast_like(ds_chunk.ssrd)
    cos_zenith = (sin_lat_b * sin_delta_b + cos_lat_b * cos_delta_b * cos_theta_hr_b).clip(-1, 1)
    zenith = np.rad2deg(np.arccos(cos_zenith))
    sin_theta_hr_b = np.sin(theta_hr).broadcast_like(ds_chunk.ssrd)
    sin_azimuth = (-sin_theta_hr_b * cos_delta_b / np.sin(np.deg2rad(zenith))).clip(-1, 1)
    cos_azimuth = (
                (sin_delta_b - sin_lat_b * np.cos(np.deg2rad(zenith))) / (cos_lat_b * np.sin(np.deg2rad(zenith)))).clip(
        -1, 1)
    azimuth = xr.where((sin_azimuth >= 0) & (cos_azimuth >= 0), np.rad2deg(np.arcsin(sin_azimuth)),
                       xr.where(cos_azimuth < 0, 180 - np.rad2deg(np.arcsin(sin_azimuth)),
                                xr.where((sin_azimuth < 0) & (cos_azimuth >= 0),
                                         360 + np.rad2deg(np.arcsin(sin_azimuth)), 0)))
    T_solar_before = hours_expanded + E_qt / 60 + ds_chunk.longitude / 15
    theta_hr_before = np.pi * (T_solar_before - 12) / 12
    cos_zenith_before = (sin_lat_b * sin_delta_b + cos_lat_b * cos_delta_b * np.cos(theta_hr_before).broadcast_like(
        ds_chunk.ssrd)).clip(-1, 1)
    zenith_before = np.rad2deg(np.arccos(cos_zenith_before))
    T_solar_after = hours_expanded + 1 + E_qt / 60 + ds_chunk.longitude / 15
    theta_hr_after = np.pi * (T_solar_after - 12) / 12
    cos_zenith_after = (sin_lat_b * sin_delta_b + cos_lat_b * cos_delta_b * np.cos(theta_hr_after).broadcast_like(
        ds_chunk.ssrd)).clip(-1, 1)
    zenith_after = np.rad2deg(np.arccos(cos_zenith_after))
    zenith_logic = ((zenith_after < 90) & (zenith_before < 90))
    zenith_bound = xr.where(zenith_before < zenith_after, zenith_before, zenith_after).clip(max=89)
    E_sc = 1360.8
    E_a = (1 + 0.033 * np.cos(2 * np.pi * n_expanded / 365)) * E_sc
    cos_zenith_safe = xr.where(cos_zenith < 1e-9, 1e-9, cos_zenith)
    k_t = (ds_chunk.ssrd / 3600) / (E_a * cos_zenith_safe)
    k_t = k_t.fillna(0).clip(0, 1.5)
    k_t = xr.where(np.isinf(k_t), 0, k_t)
    k_d = xr.zeros_like(k_t)
    k_d = xr.where((k_t > 0) & (k_t < 0.22), 1 - 0.09 * k_t, k_d)
    k_d = xr.where(k_t > 0.8, 0.165, k_d)
    mask_kd = (k_t >= 0.22) & (k_t <= 0.8)
    k_d = xr.where(mask_kd, 0.9511 - 0.1604 * k_t + 4.388 * k_t ** 2 - 16.638 * k_t ** 3 + 12.336 * k_t ** 4, k_d)
    array_tilt = xr.where(zenith < 90, zenith, 90)
    array_azimuth = azimuth
    cos_AOI = (np.cos(np.deg2rad(zenith)) * np.cos(np.deg2rad(array_tilt)) +
               np.sin(np.deg2rad(zenith)) * np.sin(np.deg2rad(array_tilt)) *
               np.cos(np.deg2rad(azimuth - array_azimuth))).clip(-1, 1)
    AOI = xr.where(zenith < 90, np.arccos(cos_AOI), 0)
    albedo = ((1 - k_d) * ds_chunk.aluvp) + (k_d * ds_chunk.aluvd)
    albedo = albedo.fillna(0.2).clip(0, 1)
    albedo = xr.where(np.isinf(albedo), 0.2, albedo)
    cos_zenith_bound_safe = xr.where(np.cos(np.deg2rad(zenith_bound)) < 1e-9, 1e-9, np.cos(np.deg2rad(zenith_bound)))
    POA_dl = (
            zenith_logic * (AOI < np.pi / 2) * (ds_chunk.ssrd / 3600) * (1 - k_d) /
            cos_zenith_bound_safe * np.cos(AOI) +
            (ds_chunk.ssrd / 3600) * k_d * (1 + np.cos(np.deg2rad(array_tilt))) / 2 +
            (ds_chunk.ssrd / 3600) * albedo * (1 - np.cos(np.deg2rad(array_tilt))) / 2
    )
    cf_da = POA_dl.transpose("valid_time", "points")
    cf_df = cf_da.to_pandas()
    # Convert longitude from 0-360 degrees to -180-180 degrees
    cf_df.columns = [f"({lon - 360 if lon > 180 else lon:.2f},{lat:.2f})" for lon, lat in content_land_site]
    return cf_df


def _select_torch_devices():
    if not HAS_TORCH:
        return []
    if TORCH_PREFERRED_DEVICES != 'auto':
        return [d for d in TORCH_PREFERRED_DEVICES if
                (d == 'cpu') or (torch.cuda.is_available() and d.startswith('cuda'))]
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ['cpu']


from typing import Optional


def calculate_cf_for_chunk_torch(ds_chunk, year, month, content_land_site,
                                 devices=None, max_points_per_block=None, dtype='float32',
                                 monitor=False, monitor_csv: Optional[str] = None, monitor_interval: float = 0.5):
    if devices is None:
        devices = _select_torch_devices()
    if not devices:
        raise RuntimeError("Torch is unavailable or no device was detected")
    torch_dtype = getattr(torch, dtype)

    time_coords = ds_chunk.valid_time
    T = len(time_coords)
    P = ds_chunk.sizes['points']

    n_np = time_coords.dt.dayofyear.values.astype(np.int32)
    hours_np = time_coords.dt.hour.values.astype(np.int32)
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        n_np = n_np - (n_np >= 60).astype(np.int32)
    n_t_cpu = torch.from_numpy(n_np).to(dtype=torch_dtype).view(T, 1)
    hours_t_cpu = torch.from_numpy(hours_np).to(dtype=torch_dtype).view(T, 1)

    lat_all = ds_chunk.latitude.values.astype(np.float32)
    lon_all = ds_chunk.longitude.values.astype(np.float32)
    ssrd_all = ds_chunk.ssrd.values.astype(np.float32)
    aluvp_all = ds_chunk.aluvp.values.astype(np.float32)
    aluvd_all = ds_chunk.aluvd.values.astype(np.float32)

    # Parse CUDA device IDs and reset peak-memory counters
    cuda_ids = []
    if HAS_TORCH and torch.cuda.is_available():
        for d in devices:
            if isinstance(d, str) and d.startswith('cuda:'):
                try:
                    idx = int(d.split(':', 1)[1])
                    cuda_ids.append(idx)
                except Exception:
                    pass
        for idx in cuda_ids:
            try:
                torch.cuda.reset_peak_memory_stats(idx)
            except Exception:
                pass

    block_results = []
    block_colnames = []
    if max_points_per_block is None:
        max_points_per_block = TORCH_MAX_POINTS_PER_BLOCK
    start = 0
    device_idx = 0

    def _compute_block_on_device(dev, lat_blk, lon_blk, ssrd_blk, aluvp_blk, aluvd_blk):
        with torch.no_grad():
            n_t = n_t_cpu.to(dev)
            hours_t = hours_t_cpu.to(dev)
            pi = torch.tensor(np.pi, device=dev, dtype=torch_dtype)
            E_qt = torch.zeros_like(n_t)

            # E_qt = torch.where(n_t >= 107, E_qt, -14.2 * torch.sin(pi * (n_t + 7) / 111))
            # E_qt = torch.where(n_t <= 246, E_qt, 16.4 * torch.sin(pi * (n_t - 247) / 113))
            # mask1 = (n_t > 106) & (n_t < 167)
            # E_qt = torch.where(mask1, 4.0 * torch.sin(pi * (n_t - 106) / 59), E_qt)
            # mask2 = (n_t > 166) & (n_t < 247)
            # E_qt = torch.where(mask2, -6.5 * torch.sin(pi * (n_t - 166) / 80), E_qt)

            E_qt = (
            (n_t < 107).to(torch_dtype) * (-14.2 * torch.sin(pi * (n_t + 7) / 111))
            + (n_t > 246).to(torch_dtype) * (16.4 * torch.sin(pi * (n_t - 247) / 113))
            + ((n_t > 106) & (n_t < 167)).to(torch_dtype) * (4.0 * torch.sin(pi * (n_t - 106) / 59))
            + ((n_t > 166) & (n_t < 247)).to(torch_dtype) * (-6.5 * torch.sin(pi * (n_t - 166) / 80))
            )

            lat = torch.from_numpy(lat_blk).to(dev, dtype=torch_dtype).view(1, -1)
            lon = torch.from_numpy(lon_blk).to(dev, dtype=torch_dtype).view(1, -1)
            T_solar = hours_t + 0.5 + E_qt / 60 + lon / 15
            theta_hr = pi * (T_solar - 12) / 12
            lat_rad = lat * (pi / 180)
            sin_lat = torch.sin(lat_rad)
            cos_lat = torch.cos(lat_rad)
            theta_d = 23.45 * (pi / 180) * torch.sin(2 * pi * (284 + n_t) / 365)
            sin_delta = torch.sin(theta_d)
            cos_delta = torch.cos(theta_d)
            cos_theta_hr = torch.cos(theta_hr)
            cos_zenith = (sin_lat * sin_delta + cos_lat * cos_delta * cos_theta_hr).clamp(-1, 1)
            zenith = torch.acos(cos_zenith) * (180 / pi)
            sin_theta_hr = torch.sin(theta_hr)
            denom = torch.sin(zenith * (pi / 180)).clamp_min(1e-12)
            sin_azimuth = (-sin_theta_hr * cos_delta) / denom
            sin_azimuth = sin_azimuth.clamp(-1, 1)
            cos_azimuth = (sin_delta - sin_lat * torch.cos(zenith * (pi / 180))) / (cos_lat * denom)
            cos_azimuth = cos_azimuth.clamp(-1, 1)
            asin_sa = torch.asin(sin_azimuth) * (180 / pi)
            azimuth = torch.where((sin_azimuth >= 0) & (cos_azimuth >= 0), asin_sa,
                                  torch.where(cos_azimuth < 0, 180 - asin_sa,
                                              torch.where((sin_azimuth < 0) & (cos_azimuth >= 0),
                                                          360 + asin_sa, torch.zeros_like(asin_sa))))
            T_solar_before = hours_t + E_qt / 60 + lon / 15
            theta_hr_before = pi * (T_solar_before - 12) / 12
            cos_zenith_before = (sin_lat * sin_delta + cos_lat * cos_delta * torch.cos(theta_hr_before)).clamp(-1, 1)
            zenith_before = torch.acos(cos_zenith_before) * (180 / pi)
            T_solar_after = hours_t + 1 + E_qt / 60 + lon / 15
            theta_hr_after = pi * (T_solar_after - 12) / 12
            cos_zenith_after = (sin_lat * sin_delta + cos_lat * cos_delta * torch.cos(theta_hr_after)).clamp(-1, 1)
            zenith_after = torch.acos(cos_zenith_after) * (180 / pi)
            zenith_logic = (zenith_after < 90) & (zenith_before < 90)
            zenith_bound = torch.where(zenith_before < zenith_after, zenith_before, zenith_after)
            zenith_bound = torch.clamp(zenith_bound, max=89)
            E_sc = 1360.8
            E_a = (1 + 0.033 * torch.cos(2 * pi * n_t / 365)) * E_sc
            cos_zenith_safe = torch.where(cos_zenith < 1e-9, torch.tensor(1e-9, device=dev, dtype=torch_dtype),
                                          cos_zenith)
            ssrd = torch.from_numpy(ssrd_blk).to(dev, dtype=torch_dtype)
            aluvp = torch.from_numpy(aluvp_blk).to(dev, dtype=torch_dtype)
            aluvd = torch.from_numpy(aluvd_blk).to(dev, dtype=torch_dtype)
            k_t = (ssrd / 3600.0) / (E_a * cos_zenith_safe)
            k_t = torch.clamp(torch.nan_to_num(k_t, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.5)
            k_d = torch.zeros_like(k_t)
            k_d = torch.where((k_t > 0) & (k_t < 0.22), 1 - 0.09 * k_t, k_d)
            k_d = torch.where(k_t > 0.8, torch.tensor(0.165, device=dev, dtype=torch_dtype), k_d)
            mask_kd = (k_t >= 0.22) & (k_t <= 0.8)
            k_d = torch.where(mask_kd, 0.9511 - 0.1604 * k_t + 4.388 * k_t ** 2 - 16.638 * k_t ** 3 + 12.336 * k_t ** 4,
                              k_d)
            array_tilt = torch.where(zenith < 90, zenith, torch.tensor(90.0, device=dev, dtype=torch_dtype))
            array_azimuth = azimuth
            cos_AOI = (torch.cos(zenith * (np.pi / 180)) * torch.cos(array_tilt * (np.pi / 180)) +
                       torch.sin(zenith * (np.pi / 180)) * torch.sin(array_tilt * (np.pi / 180)) *
                       torch.cos((azimuth - array_azimuth) * (np.pi / 180)))
            cos_AOI = cos_AOI.clamp(-1, 1)
            AOI = torch.where(zenith < 90, torch.acos(cos_AOI), torch.zeros_like(cos_AOI))
            albedo = ((1 - k_d) * aluvp) + (k_d * aluvd)
            albedo = torch.clamp(torch.nan_to_num(albedo, nan=0.2, posinf=0.2, neginf=0.2), 0.0, 1.0)
            cos_zenith_bound = torch.cos(zenith_bound * (np.pi / 180)).clamp_min(1e-9)
            POA_dl = (
                    (zenith_logic & (AOI < (np.pi / 2))).to(torch_dtype) * (ssrd / 3600.0) * (1 - k_d) /
                    cos_zenith_bound * torch.cos(AOI)
                    + (ssrd / 3600.0) * k_d * (1 + torch.cos(array_tilt * (np.pi / 180))) / 2
                    + (ssrd / 3600.0) * albedo * (1 - torch.cos(array_tilt * (np.pi / 180))) / 2
            )
            return POA_dl.detach().to('cpu')

    # Optional GPU monitoring context
    monitor_ctx = GPUMonitor(cuda_ids, interval=monitor_interval, csv_path=monitor_csv) if monitor else nullcontext()

    with monitor_ctx:
        while start < P:
            end = min(P, start + max_points_per_block)
            lat_blk = lat_all[start:end]
            lon_blk = lon_all[start:end]
            ssrd_blk = ssrd_all[:, start:end]
            aluvp_blk = aluvp_all[:, start:end]
            aluvd_blk = aluvd_all[:, start:end]
            dev = devices[device_idx % len(devices)]
            cur_block = end - start
            t_block0 = time.perf_counter()
            while True:
                try:
                    cf_t = _compute_block_on_device(dev, lat_blk, lon_blk, ssrd_blk, aluvp_blk, aluvd_blk)
                    # Convert longitude from 0-360 degrees to -180-180 degrees
                    cols = [
                        f"({content_land_site[i][0] - 360 if content_land_site[i][0] > 180 else content_land_site[i][0]:.2f},{content_land_site[i][1]:.2f})"
                        for i in range(start, end)]
                    block_results.append(cf_t)
                    block_colnames.extend(cols)
                    break
                except RuntimeError as oom:
                    msg = str(oom)
                    if ('CUDA' in msg or 'out of memory' in msg.lower()) and cur_block > 1000:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        cur_block //= 2
                        end = start + cur_block
                        lat_blk = lat_all[start:end]
                        lon_blk = lon_all[start:end]
                        ssrd_blk = ssrd_all[:, start:end]
                        aluvp_blk = aluvp_all[:, start:end]
                        aluvd_blk = aluvd_all[:, start:end]
                        continue
                    raise
            t_block1 = time.perf_counter()
            elems = (T * (end - start))
            print(
                f"[GPU] month {month:02d} block {start}-{end} | device {dev} | elapsed {t_block1 - t_block0:.3f}s | throughput {(elems / max(t_block1 - t_block0, 1e-9)) / 1e6:.2f} M elems/s")
            device_idx += 1
            start = end

    # Print peak memory use when available
    if cuda_ids:
        for idx in cuda_ids:
            try:
                peak = torch.cuda.max_memory_allocated(idx) / (1024 * 1024)
                print(f"[GPU] cuda:{idx} peak allocated memory: {peak:.1f} MB")
            except Exception:
                pass

    cf_full = torch.cat(block_results, dim=1).numpy()
    index = pd.to_datetime(time_coords.values)
    cf_df = pd.DataFrame(cf_full, index=index, columns=block_colnames)
    return cf_df


def process_year_chunked(year, file_lists, site_indices, content_land_site, save_folder, log_file_path):
    process_name = mp.current_process().name
    t1 = time.time()
    final_output_path = save_folder / f"cf_{year}.parquet"

    missing_months = None
    if final_output_path.exists():
        missing_months = check_missing_months(final_output_path, year)
        if missing_months is None:
            print(f"[{process_name}] Warning: output file is corrupted; deleting and reprocessing: {year}")
            try:
                final_output_path.unlink()
                safe_log_write(log_file_path, f"[{year}] Corrupted output detected; file deleted for reprocessing")
            except Exception as e:
                print(f"[{process_name}] Error: failed to delete corrupted file: {e}")
                return {'year': year, 'status': 'error', 'message': f'Output file is corrupted and cannot be deleted: {e}', 'skipped_chunks': []}
        elif len(missing_months) == 0:
            return {'year': year, 'status': 'skipped', 'message': 'Final output already exists and is complete', 'skipped_chunks': []}
        else:
            print(f"[{process_name}] Output exists but is incomplete; missing months: {missing_months}; incremental merge will be used")

    is_incremental_mode = (missing_months is not None)
    if not is_incremental_mode:
        print(f"[{process_name}] Processing year: {year}")

    temp_output_path = save_folder / f"cf_{year}.parquet.tmp"
    if temp_output_path.exists():
        temp_output_path.unlink()

    writer = None
    skipped_chunks = []
    is_success = False
    new_months_data = {}

    try:
        # Get data-file paths for the current year
        ssrd_file = file_lists['ssrd_files'].get(year)
        albedo_file = file_lists['albedo_files'].get(year)
        wind_temp_file = file_lists['wind_temp_files'].get(year)

        if not ssrd_file or not os.path.exists(ssrd_file):
            return {'year': year, 'status': 'error', 'message': f'SSRD file does not exist: {year}', 'skipped_chunks': []}

        if not albedo_file or not os.path.exists(albedo_file):
            return {'year': year, 'status': 'error', 'message': f'Albedo file does not exist: {year}', 'skipped_chunks': []}

        # Temperature and wind-speed files are optional and used only after POA calculation
        if not wind_temp_file or not os.path.exists(wind_temp_file):
            print(f"[{process_name}] Warning: temperature/wind-speed file does not exist: {year}; only POA will be computed")
            wind_temp_file = None

        HOURS_PER_MONTH_CHUNK = 168  # Default value used only when native chunks cannot be obtained
        cache_root = save_folder / "_nc_cache"
        month_cache_root = save_folder / "_month_cache"

        used_ssrd_file = ssrd_file
        used_albedo_file = albedo_file
        used_wind_temp_file = wind_temp_file

        if ENABLE_LOCAL_CACHE and CACHE_STRATEGY == 'always':
            used_ssrd_file = ensure_local_copy(ssrd_file, cache_root, year, log_file_path)
            used_albedo_file = ensure_local_copy(albedo_file, cache_root, year, log_file_path)
            if wind_temp_file:
                used_wind_temp_file = ensure_local_copy(wind_temp_file, cache_root, year, log_file_path)

        # Dynamically obtain native chunk sizes to avoid xarray warnings
        ssrd_chunk = get_native_time_chunk(used_ssrd_file, engine="h5netcdf", default_chunk=HOURS_PER_MONTH_CHUNK)
        albedo_chunk = get_native_time_chunk(used_albedo_file, engine="h5netcdf", default_chunk=HOURS_PER_MONTH_CHUNK)
        wind_chunk = get_native_time_chunk(used_wind_temp_file, engine="h5netcdf", default_chunk=HOURS_PER_MONTH_CHUNK) if used_wind_temp_file else HOURS_PER_MONTH_CHUNK

        try:
            engine = "h5netcdf"
            ds_ssrd = xr.open_dataset(used_ssrd_file, engine=engine, chunks={'valid_time': ssrd_chunk},
                                      decode_times=True)
            ds_albedo = xr.open_dataset(used_albedo_file, engine=engine, chunks={'valid_time': albedo_chunk},
                                        decode_times=True)
            # Open temperature and wind-speed files when needed
            # ds_wind_temp = xr.open_dataset(used_wind_temp_file, engine=engine, chunks={'valid_time': HOURS_PER_MONTH_CHUNK}, decode_times=True) if used_wind_temp_file else None
        except (ValueError, OSError) as e:
            print(f"[{year}] Warning: h5netcdf failed; falling back to netcdf4. Error: {e}")
            engine = "netcdf4"
            # Recompute native chunk size with the netcdf4 engine
            ssrd_chunk = get_native_time_chunk(used_ssrd_file, engine="netcdf4", default_chunk=HOURS_PER_MONTH_CHUNK)
            albedo_chunk = get_native_time_chunk(used_albedo_file, engine="netcdf4", default_chunk=HOURS_PER_MONTH_CHUNK)
            wind_chunk = get_native_time_chunk(used_wind_temp_file, engine="netcdf4", default_chunk=HOURS_PER_MONTH_CHUNK) if used_wind_temp_file else HOURS_PER_MONTH_CHUNK
            ds_ssrd = xr.open_dataset(used_ssrd_file, engine=engine, chunks={'valid_time': ssrd_chunk})
            ds_albedo = xr.open_dataset(used_albedo_file, engine=engine, chunks={'valid_time': albedo_chunk})
            # ds_wind_temp = xr.open_dataset(used_wind_temp_file, engine=engine, chunks={'valid_time': HOURS_PER_MONTH_CHUNK}) if used_wind_temp_file else None

        with ds_ssrd, ds_albedo:
            # Extract SSRD data for selected sites
            ds_ssrd_points = ds_ssrd.isel(
                latitude=xr.DataArray(site_indices['lat'], dims="points"),
                longitude=xr.DataArray(site_indices['lon'], dims="points")
            )

            # Extract albedo data for selected sites (aluvp and aluvd)
            ds_albedo_points = ds_albedo.isel(
                latitude=xr.DataArray(site_indices['lat'], dims="points"),
                longitude=xr.DataArray(site_indices['lon'], dims="points")
            )

            # Merge SSRD and albedo data
            ds_points_full = xr.merge([ds_ssrd_points, ds_albedo_points], compat='override')

            # ========== Temperature and wind-speed data preparation ==========
            with xr.open_dataset(used_wind_temp_file, engine=engine,
                                 chunks={'valid_time': wind_chunk}) as ds_wind_temp:
                ds_wind_temp_points = ds_wind_temp.isel(
                    latitude=xr.DataArray(site_indices['lat'], dims="points"),
                    longitude=xr.DataArray(site_indices['lon'], dims="points")
                )
                ds_points_full = xr.merge([ds_points_full, ds_wind_temp_points[['t2m', 'u10', 'v10']]],
                                          compat='override')

            time_groups = ds_points_full.groupby('valid_time.month')

            # Select months from missing-month records in incremental mode and intersect with GPU_TEST_MONTHS when set
            if is_incremental_mode:
                months_to_process = set(missing_months)
            else:
                months_to_process = None
            if GPU_TEST_MONTHS:
                test_months = set(int(x) for x in GPU_TEST_MONTHS.split(',') if x.strip())
                months_to_process = test_months if months_to_process is None else (months_to_process & test_months)
                print(f"[{process_name}] Trial-run months: {sorted(months_to_process)}")

            first_chunk = True
            ds_ssrd_retry = None
            ds_albedo_retry = None
            ds_local_month = None
            local_month_file = None
            for month, ds_chunk in tqdm(time_groups, total=len(time_groups), desc=f"Processing monthly blocks for {year}"):
                if months_to_process is not None and month not in months_to_process:
                    continue

                max_attempts = 5
                chunk_success = False
                ds_chunk_to_use = ds_chunk

                for attempt in range(1, max_attempts + 1):
                    try:
                        if attempt > 1:
                            retry_wait = 5 if attempt == 2 else 10
                            print(f"[{year}-{month:02d}] Waiting before network retry: {retry_wait} seconds...")
                            time.sleep(retry_wait)
                            try:
                                if ds_ssrd_retry is not None:
                                    ds_ssrd_retry.close()
                                if ds_albedo_retry is not None:
                                    ds_albedo_retry.close()
                            except Exception:
                                pass
                            ds_ssrd_retry = None
                            ds_albedo_retry = None
                            try_used_ssrd = used_ssrd_file
                            try_used_albedo = used_albedo_file
                            if ENABLE_LOCAL_CACHE and CACHE_STRATEGY in ("on_retry",
                                                                         "always") and attempt == max_attempts:
                                try_used_ssrd = ensure_local_copy(used_ssrd_file, cache_root, year, log_file_path)
                                try_used_albedo = ensure_local_copy(used_albedo_file, cache_root, year, log_file_path)
                            retry_hours_chunk = HOURS_PER_MONTH_CHUNK
                            if attempt == 2:
                                retry_hours_chunk = 48
                            elif attempt >= 3:
                                retry_hours_chunk = 24
                            try:
                                if engine == "h5netcdf":
                                    ds_ssrd_retry = xr.open_dataset(try_used_ssrd, engine=engine,
                                                                    chunks={'valid_time': retry_hours_chunk},
                                                                    decode_times=True)
                                    ds_albedo_retry = xr.open_dataset(try_used_albedo, engine=engine,
                                                                      chunks={'valid_time': retry_hours_chunk},
                                                                      decode_times=True)
                                else:
                                    ds_ssrd_retry = xr.open_dataset(try_used_ssrd, engine=engine,
                                                                    chunks={'valid_time': retry_hours_chunk})
                                    ds_albedo_retry = xr.open_dataset(try_used_albedo, engine=engine,
                                                                      chunks={'valid_time': retry_hours_chunk})
                                ds_ssrd_points_r = ds_ssrd_retry.isel(
                                    latitude=xr.DataArray(site_indices['lat'], dims="points"),
                                    longitude=xr.DataArray(site_indices['lon'], dims="points"))
                                ds_albedo_points_r = ds_albedo_retry.isel(
                                    latitude=xr.DataArray(site_indices['lat'], dims="points"),
                                    longitude=xr.DataArray(site_indices['lon'], dims="points"))
                                ds_points_full_r = xr.merge([ds_ssrd_points_r, ds_albedo_points_r], compat='override')
                                ds_chunk_to_use = ds_points_full_r.sel(
                                    valid_time=ds_points_full_r.valid_time.dt.month == month)
                            except Exception as reopen_error:
                                print(f"[{year}-{month:02d}] Warning: failed to reopen dataset: {reopen_error}")
                                ds_chunk_to_use = ds_points_full.sel(
                                    valid_time=ds_points_full.valid_time.dt.month == month)

                        use_month_cache_now = MONTH_CACHE_ENABLED and (
                                    MONTH_CACHE_STRATEGY == 'always' or attempt == max_attempts)
                        if use_month_cache_now:
                            try:
                                month_cache_root.mkdir(parents=True, exist_ok=True)
                                local_month_file = month_cache_root / f"month_{year}_{month:02d}_{os.getpid()}_{attempt}.nc"
                                if local_month_file.exists():
                                    try:
                                        local_month_file.unlink()
                                    except Exception:
                                        pass
                                try:
                                    ds_chunk_loaded = ds_chunk_to_use.load()
                                    vars_to_keep = ['ssrd', 'aluvp', 'aluvd']  # Updated variable names
                                    ds_vars = ds_chunk_loaded[vars_to_keep]
                                    if 'latitude' not in ds_vars.coords or 'longitude' not in ds_vars.coords:
                                        ds_vars = ds_chunk_loaded
                                except Exception:
                                    try:
                                        ds_vars = ds_chunk_to_use.drop_vars(['expver'], errors='ignore')
                                    except Exception:
                                        ds_vars = ds_chunk_to_use
                                ds_vars.to_netcdf(str(local_month_file), engine='h5netcdf',
                                                  encoding={v: {'zlib': True, 'complevel': 1} for v in
                                                            ['ssrd', 'aluvp', 'aluvd'] if v in ds_vars})
                                try:
                                    if ds_ssrd_retry is not None:
                                        ds_ssrd_retry.close()
                                    if ds_albedo_retry is not None:
                                        ds_albedo_retry.close()
                                except Exception:
                                    pass
                                ds_ssrd_retry = None
                                ds_albedo_retry = None
                                try:
                                    ds_local_month = xr.open_dataset(str(local_month_file), engine="h5netcdf")
                                except Exception:
                                    ds_local_month = xr.open_dataset(str(local_month_file), engine="netcdf4")
                                ds_chunk_to_use = ds_local_month
                            except Exception as month_cache_err:
                                print(f"[{year}-{month:02d}] Warning: failed to create local monthly temporary file: {month_cache_err}")

                        if not (use_month_cache_now and ds_local_month is not None):
                            ds_chunk_to_use.load()

                        # Prefer Torch and fall back to CPU on exceptions
                        if ENABLE_TORCH and HAS_TORCH:
                            try:
                                monitor_csv_path = None
                                if TORCH_ENABLE_MONITOR and TORCH_MONITOR_CSV_DIR:
                                    # One CSV file per month: gpu_monitor_YEAR_MM.csv
                                    try:
                                        Path(TORCH_MONITOR_CSV_DIR).mkdir(parents=True, exist_ok=True)
                                        monitor_csv_path = str(
                                            Path(TORCH_MONITOR_CSV_DIR) / f"gpu_monitor_{year}_{month:02d}.csv")
                                    except Exception as mkd_err:
                                        print(f"[GPU] Warning: failed to create monitoring directory: {mkd_err}")
                                cf_df_chunk = calculate_cf_for_chunk_torch(
                                    ds_chunk_to_use, year, month, content_land_site,
                                    max_points_per_block=TORCH_MAX_POINTS_PER_BLOCK,
                                    monitor=TORCH_ENABLE_MONITOR,
                                    monitor_csv=monitor_csv_path,
                                    monitor_interval=TORCH_MONITOR_INTERVAL
                                )

                                # ========== Temperature and wind-speed data processing ==========
                                # POA data are stored in cf_df_chunk as POA multiplied by the loss factor
                                if 't2m' in ds_chunk_to_use and 'u10' in ds_chunk_to_use and 'v10' in ds_chunk_to_use:
                                    # Extract temperature and wind-speed data
                                    t2m = ds_chunk_to_use['t2m'].values  # 2 m temperature (K)
                                    u10 = ds_chunk_to_use['u10'].values  # 10 m zonal wind speed (m/s)
                                    v10 = ds_chunk_to_use['v10'].values  # 10 m meridional wind speed (m/s)

                                    # 1. Compute 10 m wind speed
                                    v10_combined = np.sqrt(u10 ** 2 + v10 ** 2)

                                    # 2. Convert wind speed from 10 m to 2 m using a wind-shear exponent of 0.143
                                    v2 = v10_combined * (2.0 / 10.0) ** 0.143

                                    # 3. Convert temperature from kelvin to degrees Celsius
                                    t_celsius = t2m - 273.15

                                    # 4. Recover POA from cf_df_chunk
                                    POA = cf_df_chunk.values

                                    # 5. Estimate surface temperature as T = 4.3 + 0.943*t + 0.028*POA - 1.528*v
                                    T_surface = 4.3 + 0.943 * t_celsius + 0.028 * POA - 1.528 * v2

                                    # 6. Compute corrected CF = POA * (1 - 0.005 * (T - 298.15))
                                    # Note: T_surface is in degrees Celsius and must be converted to kelvin before comparison with 298.15 K
                                    T_kelvin = T_surface + 273.15
                                    f_T_dual = calculate_correction_factor_gpu(T_kelvin)
                                    cf_corrected = POA * (1 + f_T_dual) / 1000.0

                                    # 7. Update cf_df_chunk
                                    cf_df_chunk = pd.DataFrame(cf_corrected, index=cf_df_chunk.index,
                                                               columns=cf_df_chunk.columns)
                                    cf_max = cf_df_chunk.values.max()
                                    print(f"[{year}-{month:02d}] Maximum temperature-corrected CF on the CPU fallback path: {cf_max:.4f}")
                                # ==========================================

                            except Exception as torch_err:
                                print(f"[{year}-{month:02d}] Warning: Torch path failed; falling back to CPU: {torch_err}")
                                safe_log_write(log_file_path, f"[{year}-{month:02d}] Torch failed; CPU fallback used: {torch_err}")
                                cf_df_chunk = calculate_cf_for_chunk(ds_chunk_to_use, year, month, content_land_site)

                                # ========== Temperature and wind-speed data processing (CPU path) ==========
                                if 't2m' in ds_chunk_to_use and 'u10' in ds_chunk_to_use and 'v10' in ds_chunk_to_use:
                                    # Extract temperature and wind-speed data
                                    t2m = ds_chunk_to_use['t2m'].values  # 2 m temperature (K)
                                    u10 = ds_chunk_to_use['u10'].values  # 10 m zonal wind speed (m/s)
                                    v10 = ds_chunk_to_use['v10'].values  # 10 m meridional wind speed (m/s)

                                    # 1. Compute 10 m wind speed
                                    v10_combined = np.sqrt(u10 ** 2 + v10 ** 2)

                                    # 2. Convert wind speed from 10 m to 2 m using a wind-shear exponent of 0.143
                                    v2 = v10_combined * (2.0 / 10.0) ** 0.143

                                    # 3. Convert temperature from kelvin to degrees Celsius
                                    t_celsius = t2m - 273.15

                                    # 4. Recover POA multiplied by the loss factor from cf_df_chunk
                                    POA = cf_df_chunk.values

                                    # 5. Estimate surface temperature as T = 4.3 + 0.943*t + 0.028*POA - 1.528*v
                                    T_surface = 4.3 + 0.943 * t_celsius + 0.028 * POA - 1.528 * v2

                                    # 6. Compute corrected CF = POA * loss_factor * (1 - 0.005 * (T - 298.15))
                                    T_kelvin = T_surface + 273.15
                                    f_T_dual = calculate_correction_factor_gpu(T_kelvin)
                                    cf_corrected = POA * (1 + f_T_dual) / 1000.0

                                    # 7. Update cf_df_chunk
                                    cf_df_chunk = pd.DataFrame(cf_corrected, index=cf_df_chunk.index,
                                                               columns=cf_df_chunk.columns)
                                    cf_max = cf_df_chunk.values.max()
                                    print(f"[{year}-{month:02d}] Maximum temperature-corrected CF on the CPU fallback path: {cf_max:.4f}")
                                # ==========================================
                        else:
                            cf_df_chunk = calculate_cf_for_chunk(ds_chunk_to_use, year, month, content_land_site)

                            # ========== Temperature and wind-speed data processing (pure CPU path) ==========
                            if 't2m' in ds_chunk_to_use and 'u10' in ds_chunk_to_use and 'v10' in ds_chunk_to_use:
                                # Extract temperature and wind-speed data
                                t2m = ds_chunk_to_use['t2m'].values  # 2 m temperature (K)
                                u10 = ds_chunk_to_use['u10'].values  # 10 m zonal wind speed (m/s)
                                v10 = ds_chunk_to_use['v10'].values  # 10 m meridional wind speed (m/s)

                                v10_combined = np.sqrt(u10 ** 2 + v10 ** 2)
                                v2 = v10_combined * (2.0 / 10.0) ** 0.143

                                t_celsius = t2m - 273.15
                                POA = cf_df_chunk.values

                                # Estimate surface temperature as T = 4.3 + 0.943*t + 0.028*POA - 1.528*v
                                T_surface = 4.3 + 0.943 * t_celsius + 0.028 * POA - 1.528 * v2

                                # Compute corrected CF = POA * loss_factor * (1 - 0.005 * (T - 298.15))
                                T_kelvin = T_surface + 273.15
                                f_T_dual = calculate_correction_factor_gpu(T_kelvin)
                                cf_corrected = POA * (1 + f_T_dual)/ 1000.0

                                # Update cf_df_chunk
                                cf_df_chunk = pd.DataFrame(cf_corrected, index=cf_df_chunk.index,
                                                           columns=cf_df_chunk.columns)
                                cf_max = cf_df_chunk.values.max()
                                print(f"[{year}-{month:02d}] Maximum temperature-corrected CF on the CPU fallback path: {cf_max:.4f}")
                            # ==========================================

                        if is_incremental_mode:
                            new_months_data[month] = cf_df_chunk
                        else:
                            table = pa.Table.from_pandas(cf_df_chunk)
                            if first_chunk:
                                writer = pq.ParquetWriter(temp_output_path, table.schema)
                                first_chunk = False
                            writer.write_table(table)

                        chunk_success = True
                        if attempt > 1:
                            retry_success_msg = f"[{year}-{month:02d}] Retry succeeded on attempt {attempt}"
                            print(f"\n{retry_success_msg}\n")
                            safe_log_write(log_file_path, retry_success_msg)
                        break

                    except Exception as chunk_error:
                        if attempt < max_attempts:
                            retry_msg = f"[{year}-{month:02d}] Attempt {attempt} failed; retrying: {chunk_error}"
                            print(f"\nWarning: {retry_msg}\n")
                            safe_log_write(log_file_path, retry_msg)
                            if 'ds_chunk_to_use' in locals():
                                del ds_chunk_to_use
                            gc.collect()
                            time.sleep(2)
                        else:
                            skipped_chunks.append(f"{year}-{month:02d}")
                            error_msg = f"[{year}-{month:02d}] Abandoning this chunk after {max_attempts} attempts: {chunk_error}"
                            print(f"\nWarning: {error_msg}\n")
                            safe_log_write(log_file_path, error_msg)
                    finally:
                        if chunk_success or attempt == max_attempts:
                            if 'ds_chunk_to_use' in locals():
                                del ds_chunk_to_use
                            try:
                                if ds_ssrd_retry is not None:
                                    ds_ssrd_retry.close()
                                if ds_albedo_retry is not None:
                                    ds_albedo_retry.close()
                            except Exception:
                                pass
                            try:
                                if ds_local_month is not None:
                                    ds_local_month.close()
                            except Exception:
                                pass
                            try:
                                if local_month_file is not None and Path(local_month_file).exists():
                                    Path(local_month_file).unlink()
                            except Exception:
                                pass
                            gc.collect()

        if is_incremental_mode:
            if new_months_data:
                if merge_months_to_existing(final_output_path, new_months_data, year, log_file_path):
                    total_time_hours = (time.time() - t1) / 3600
                    merged_count = len(new_months_data)
                    print(f"[{process_name}] SUCCESS: {year} incremental merge completed after filling {merged_count} months")
                    return {'year': year, 'status': 'success',
                            'message': f'Incremental merge succeeded after filling {merged_count} months; elapsed time: {total_time_hours:.3f} hours',
                            'skipped_chunks': skipped_chunks}
                else:
                    return {'year': year, 'status': 'error', 'message': 'Incremental merge failed',
                            'skipped_chunks': skipped_chunks}
            else:
                return {'year': year, 'status': 'error', 'message': 'No new months are available for merging',
                        'skipped_chunks': skipped_chunks}
        else:
            if writer is None:
                return {'year': year, 'status': 'error', 'message': 'All data blocks are corrupted or cannot be processed',
                        'skipped_chunks': skipped_chunks}
            is_success = True
            total_time_hours = (time.time() - t1) / 3600
            print(f"[{process_name}] SUCCESS: {year} processing completed")
            return {'year': year, 'status': 'success', 'message': f'Processing succeeded; elapsed time: {total_time_hours:.3f} hours',
                    'skipped_chunks': skipped_chunks}

    except Exception as e:
        return {'year': year, 'status': 'error', 'message': f'Critical error: {e}', 'skipped_chunks': skipped_chunks}
    finally:
        if writer:
            writer.close()
            if is_success and not is_incremental_mode:
                try:
                    temp_output_path.rename(final_output_path)
                    print(f"[{process_name}] Renamed: {final_output_path.name}")
                except Exception as e:
                    error_msg = f"Rename failed: {e}"
                    print(f"[{process_name}] Error: {error_msg}")
                    safe_log_write(log_file_path, f"[{year}] {error_msg}")
        if temp_output_path.exists():
            temp_output_path.unlink()
        gc.collect()


def main():
    time_start = time.time()

    # ========== Data path configuration ==========
    land_site_path = r"E:\Zixin\CodeCAL\site_matching_world.csv"

    # SSRD data path
    ssrd_folder = Path(r"E:\Var4Solar\SSRD")

    # Albedo data path (aluvp and aluvd)
    albedo_folder = Path(r"E:\Var4Solar\Albedo")

    # Temperature and wind-speed data path
    wind_temp_folder = Path(r"F:\Var4Solar")

    save_folder = Path(r"E:\solar\cf_true")

    # Year range can be restricted through GPU_TEST_YEARS, for example "2020" or "2020,2021"
    # if GPU_TEST_YEARS:
    #     years_to_process = [int(x) for x in GPU_TEST_YEARS.split(',') if x.strip()]
    # else:
    #     years_to_process = list(range(1980, 2000))
    years_to_process = list(range(1950, 2025))
    num_processes = 4

    save_folder.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {save_folder}")

    log_file_path = save_folder / "data_corruption_log_realtime.txt"
    with open(log_file_path, 'w', encoding='utf-8') as log_f:
        log_f.write("=" * 80 + "\n")
        log_f.write("Data Corruption Real-time Log\n")
        log_f.write("=" * 80 + "\n")
        log_f.write(f"Task start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"Processing year range: {years_to_process[0]} - {years_to_process[-1]}\n")
        log_f.write(f"Number of parallel processes: {num_processes}\n")
        log_f.write("=" * 80 + "\n")
        log_f.write("Notes:\n")
        log_f.write("- This log is updated in real time and flushed after each error\n")
        log_f.write("- Recorded errors are retained even if the program is interrupted with Ctrl+C\n")
        log_f.write("- The log can be opened as read-only while the program is running\n")
        log_f.write("- Log format: [timestamp] [year-month] error: details\n")
        log_f.write("- Incremental merging is supported when output files exist but are missing months\n")
        log_f.write("=" * 80 + "\n\n")
    print(f"Real-time error log: {log_file_path}")

    print("Loading and sorting site information...")
    df_sites = pd.read_csv(land_site_path).sort_values(['grid_lat_idx', 'grid_lon_idx'])

    # Restrict the number of trial-run sites when requested
    if GPU_TEST_MAX_POINTS and GPU_TEST_MAX_POINTS > 0:
        df_sites = df_sites.iloc[:GPU_TEST_MAX_POINTS]
        print(f"Number of trial-run sites: {len(df_sites)}")

    site_indices = {
        'lat': df_sites['grid_lat_idx'].values,
        'lon': df_sites['grid_lon_idx'].values
    }
    content_land_site = list(zip(df_sites['site_lon'].values, df_sites['site_lat'].values))
    print(f"Loaded {len(df_sites)} sites.")

    print("Reading data-file paths...")
    # Build the SSRD file list
    ssrd_files = sorted(ssrd_folder.glob("*.nc"))
    ssrd_files_dict = {}
    for f in ssrd_files:
        # Extract the four-digit year from the filename
        match = re.search(r'(19|20)\d{2}', f.stem)
        if match:
            year = int(match.group())
            ssrd_files_dict[year] = str(f)

    # Build the albedo file list (aluvp and aluvd)
    albedo_files = sorted(albedo_folder.glob("*.nc"))
    albedo_files_dict = {}
    for f in albedo_files:
        # Extract the four-digit year from the filename
        match = re.search(r'(19|20)\d{2}', f.stem)
        if match:
            year = int(match.group())
            albedo_files_dict[year] = str(f)

    # Build the temperature and wind-speed file list
    wind_temp_files = sorted(wind_temp_folder.glob("*.nc"))
    wind_temp_files_dict = {}
    for f in wind_temp_files:
        # Extract the four-digit year from the filename
        match = re.search(r'(19|20)\d{2}', f.stem)
        if match:
            year = int(match.group())
            wind_temp_files_dict[year] = str(f)

    print(f"  - Found {len(ssrd_files_dict)} SSRD files")
    print(f"  - Found {len(albedo_files_dict)} albedo files")
    print(f"  - Found {len(wind_temp_files_dict)} temperature/wind-speed files")

    file_lists = {
        'ssrd_files': ssrd_files_dict,
        'albedo_files': albedo_files_dict,
        'wind_temp_files': wind_temp_files_dict
    }

    print(f"\nPreparing to start {num_processes} processes for {len(years_to_process)} years...")
    worker_func = partial(
        process_year_chunked,
        file_lists=file_lists,
        site_indices=site_indices,
        content_land_site=content_land_site,
        save_folder=save_folder,
        log_file_path=log_file_path
    )

    all_results = []
    try:
        with mp.Pool(processes=num_processes) as pool:
            for result in tqdm(pool.imap_unordered(worker_func, years_to_process), total=len(years_to_process),
                               desc="Overall progress"):
                all_results.append(result)
                print(f"\n[{result['year']}] status: {result['status']} - {result['message']}")
    except KeyboardInterrupt:
        print("\n\nUser interruption detected (Ctrl+C); exiting safely...")
        safe_log_write(log_file_path, "=" * 50)
        safe_log_write(log_file_path, "Task interrupted by the user (Ctrl+C)")
        safe_log_write(log_file_path, f"Completed {len(all_results)} years of processing")
        safe_log_write(log_file_path, "=" * 50)
        print(f"Recorded errors have been saved to: {log_file_path}")
        print("Completed years will be skipped automatically on the next run")
        return

    print("\n" + "=" * 60)
    print("All years processed; result summary:")
    print("=" * 60)
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    skipped_count = sum(1 for r in all_results if r['status'] == 'skipped')
    error_count = sum(1 for r in all_results if r['status'] == 'error')
    all_skipped_chunks = []
    for r in all_results:
        if r['skipped_chunks']:
            all_skipped_chunks.extend(r['skipped_chunks'])
    print(f"Success: {success_count} years")
    print(f"Skipped (already exists): {skipped_count} years")
    print(f"Failed: {error_count} years")
    print(f"Detected {len(all_skipped_chunks)} corrupted data blocks.")
    if error_count > 0:
        print("\nFailure details:")
        for r in all_results:
            if r['status'] == 'error':
                print(f"  - [{r['year']}] {r['message']}")

    with open(log_file_path, 'a', encoding='utf-8') as log_f:
        log_f.write("\n" + "=" * 80 + "\n")
        log_f.write("Task completion summary\n")
        log_f.write("=" * 80 + "\n")
        log_f.write(f"Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"Total elapsed time: {(time.time() - time_start) / 3600:.2f} hours\n")
        log_f.write(f"Success: {success_count} years\n")
        log_f.write(f"Skipped (already exists): {skipped_count} years\n")
        log_f.write(f"Failed: {error_count} years\n")
        log_f.write(f"Detected {len(all_skipped_chunks)} corrupted data blocks\n")
        if all_skipped_chunks:
            log_f.write("\nCorrupted data-block list sorted by year-month:\n")
            log_f.write("-" * 80 + "\n")
            for chunk_id in sorted(all_skipped_chunks):
                log_f.write(f"{chunk_id}\n")
        if error_count > 0:
            log_f.write("\nFailed-year details:\n")
            log_f.write("-" * 80 + "\n")
            for r in all_results:
                if r['status'] == 'error':
                    log_f.write(f"[{r['year']}] {r['message']}\n")
        log_f.write("=" * 80 + "\n")

    print(f"\nFull log written to: {log_file_path}")
    print(f"\nAll tasks completed; total elapsed time: {(time.time() - time_start) / 3600:.2f} hours")

    # Generate a GPU monitoring summary by aggregating monthly CSV peak and mean values
    if TORCH_ENABLE_MONITOR and TORCH_MONITOR_CSV_DIR:
        monitor_dir = Path(TORCH_MONITOR_CSV_DIR)
        if monitor_dir.exists():
            summary_file = monitor_dir / "gpu_monitor_summary.txt"
            try:
                import csv
                from collections import defaultdict
                stats = defaultdict(lambda: {'gpu_vals': [], 'mem_vals': [], 'used_vals': [], 'total_mem': None})
                for csv_path in sorted(monitor_dir.glob("gpu_monitor_*.csv")):
                    try:
                        with open(csv_path, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                dev = row.get('device')
                                if dev is None:
                                    continue
                                dev_key = f"cuda:{dev}"
                                try:
                                    gpu_util = float(row.get('util_gpu_%', '0'))
                                    mem_util = float(row.get('util_mem_%', '0'))
                                    mem_used = float(row.get('mem_used_MB', '0'))
                                    mem_total = float(row.get('mem_total_MB', '0'))
                                except Exception:
                                    continue
                                stats[dev_key]['gpu_vals'].append(gpu_util)
                                stats[dev_key]['mem_vals'].append(mem_util)
                                stats[dev_key]['used_vals'].append(mem_used)
                                if stats[dev_key]['total_mem'] is None:
                                    stats[dev_key]['total_mem'] = mem_total
                    except Exception:
                        pass
                with open(summary_file, 'w', encoding='utf-8') as sf:
                    sf.write("GPU Monitor Summary\n")
                    sf.write("===================\n")
                    for dev, d in sorted(stats.items()):
                        if not d['gpu_vals']:
                            continue
                        avg_gpu = sum(d['gpu_vals']) / len(d['gpu_vals'])
                        peak_gpu = max(d['gpu_vals'])
                        peak_used = max(d['used_vals']) if d['used_vals'] else 0
                        avg_used = sum(d['used_vals']) / len(d['used_vals']) if d['used_vals'] else 0
                        total_mem = d['total_mem'] or 0
                        sf.write(
                            f"{dev} | mean GPU utilization: {avg_gpu:.1f}% | peak GPU utilization: {peak_gpu:.1f}% | mean memory: {avg_used:.1f}MB | peak memory: {peak_used:.1f}/{total_mem:.1f}MB\n")
                print(f"GPU monitoring summary generated: {summary_file}")
            except Exception as e:
                print(f"Warning: failed to generate GPU monitoring summary: {e}")


if __name__ == "__main__":
    main()
