import numpy as np
import pandas as pd
from scipy import stats
from astropy.timeseries import BoxLeastSquares, LombScargle
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

DATA = '.'

def process_star(flux_raw, cen_col_raw, cen_row_raw, t, bad_mask, master_trend):
    f = flux_raw.copy()
    f[bad_mask] = np.nan
    
    median_f = np.nanmedian(f)
    if median_f == 0 or np.isnan(median_f):
        return {'std': 0, 'skew': 0, 'kurt': 0, 'p5': 0, 'p95': 0, 
                'bls_period': 0, 'bls_depth': 0, 'bls_duration': 0, 'centroid_shift': 0,
                'odd_even_depth_diff': 0, 'ls_primary_freq': 0, 'ls_primary_power': 0, 
                'flare_count': 0, 'max_jump': 0, 'centroid_var_col': 0, 
                'centroid_var_row': 0, 'bad_data_ratio': 0}
        
    f_norm = (f / median_f) - 1
    
    # Subtract Master CCD Trend FIRST
    f_norm_clean = f_norm - master_trend
    
    # 1. Base statistical features
    feat = {
        'std': np.nanstd(f_norm_clean),
        'skew': float(stats.skew(f_norm_clean, nan_policy='omit')),
        'kurt': float(stats.kurtosis(f_norm_clean, nan_policy='omit')),
        'p5': np.nanpercentile(f_norm_clean, 5),
        'p95': np.nanpercentile(f_norm_clean, 95),
    }
    
    # 2. Detrending (Remove slow variations)
    f_series = pd.Series(f_norm_clean)
    trend = f_series.interpolate(limit_direction='both').rolling(window=47, center=True).median().values
    f_det = f_norm_clean - trend
    
    valid = ~np.isnan(f_det) & ~np.isnan(t) & ~np.isnan(cen_col_raw) & ~np.isnan(cen_row_raw)
    
    if valid.sum() < 100:
        feat.update({'bls_period': 0, 'bls_depth': 0, 'bls_duration': 0, 'centroid_shift': 0,
                     'odd_even_depth_diff': 0, 'ls_primary_freq': 0, 'ls_primary_power': 0, 
                     'flare_count': 0, 'max_jump': 0, 'centroid_var_col': 0, 
                     'centroid_var_row': 0, 'bad_data_ratio': 0})
        return feat
        
    t_clean = t[valid]
    f_clean = f_det[valid]
    col_clean = cen_col_raw[valid]
    row_clean = cen_row_raw[valid]
    
    # 3. BLS Search
    model = BoxLeastSquares(t_clean, f_clean)
    results = model.autopower(0.2, minimum_period=0.5, maximum_period=13.0, minimum_n_transit=2)
    best = np.argmax(results.power)
    per = results.period[best]
    dur = results.duration[best]
    t0 = results.transit_time[best]
    
    feat['bls_period'] = per
    feat['bls_depth'] = results.depth[best]
    feat['bls_duration'] = dur
    
    # 4. Centroid Shift
    phase = (t_clean - t0 + 0.5 * per) % per - 0.5 * per
    in_tr = np.abs(phase) < (dur / 2)
    out_tr = ~in_tr
    if in_tr.sum() > 0 and out_tr.sum() > 0:
        shift_col = np.nanmedian(col_clean[in_tr]) - np.nanmedian(col_clean[out_tr])
        shift_row = np.nanmedian(row_clean[in_tr]) - np.nanmedian(row_clean[out_tr])
        feat['centroid_shift'] = np.sqrt(shift_col**2 + shift_row**2)
    else:
        feat['centroid_shift'] = 0.0
        
    # 5. Odd/Even Transit Depth Difference
    if in_tr.sum() > 0 and out_tr.sum() > 0:
        baseline = np.nanmedian(f_clean[out_tr])
        transit_num = np.round((t_clean - t0) / per)
        is_odd = (transit_num % 2 != 0) & in_tr
        is_even = (transit_num % 2 == 0) & in_tr
        
        if is_odd.sum() > 0 and is_even.sum() > 0:
            depth_odd = baseline - np.nanmedian(f_clean[is_odd])
            depth_even = baseline - np.nanmedian(f_clean[is_even])
            feat['odd_even_depth_diff'] = float(np.abs(depth_odd - depth_even))
        else:
            feat['odd_even_depth_diff'] = 0.0
    else:
        feat['odd_even_depth_diff'] = 0.0
        
    # 6. Lomb-Scargle (Pulsations & Spots)
    ls = LombScargle(t_clean, f_clean)
    freq, power = ls.autopower(minimum_frequency=0.1, maximum_frequency=24.0)
    best_idx_ls = np.argmax(power)
    feat['ls_primary_freq'] = freq[best_idx_ls]
    feat['ls_primary_power'] = power[best_idx_ls]

    # 7. Flare Detection
    threshold = 3 * feat['std']
    is_spike = f_det > threshold
    tail_threshold = 1 * feat['std']
    has_tail = np.roll(f_det, -1) > tail_threshold
    valid_flares = is_spike & has_tail
    feat['flare_count'] = int(np.sum(valid_flares))
    
    # 8. Instrument Glitch Features
    if len(f_clean) > 1:
        feat['max_jump'] = float(np.nanmax(np.abs(np.diff(f_clean))))
    else:
        feat['max_jump'] = 0.0
        
    feat['centroid_var_col'] = float(np.nanvar(col_clean))
    feat['centroid_var_row'] = float(np.nanvar(row_clean))
    feat['bad_data_ratio'] = float(bad_mask.sum() / len(bad_mask))
        
    return feat

def build_dataset(split='train'):
    print(f"Loading {split} data...")
    flux = np.load(f'{DATA}/{split}_flux.npy', mmap_mode='r')
    cen = np.load(f'{DATA}/{split}_centroid.npy', mmap_mode='r')
    stars = pd.read_csv(f'{DATA}/{split}_stars.csv')
    
    if split == 'train':
        cad = pd.read_csv(f'{DATA}/train_cadences.csv')
    else:
        cad = pd.read_csv(f'{DATA}/test_cadences.csv')
        
    t = cad.time_btjd.values
    quality = cad.quality.values
    bad_mask = (quality & 1 > 0) | (quality & 2 > 0) | (quality & 4 > 0)
    
    # Calculate Master CCD Trends
    print("Calculating master CCD trends...")
    ccd_trends = {}
    for cam in range(1, 5):
        for ccd in range(1, 5):
            mask = (stars['camera'] == cam) & (stars['ccd'] == ccd)
            if mask.sum() > 0:
                group_flux = flux[mask]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    group_medians = np.nanmedian(group_flux, axis=1, keepdims=True)
                    group_norm = (group_flux / group_medians) - 1
                    master_trend = np.nanmedian(group_norm, axis=0)
                ccd_trends[(cam, ccd)] = master_trend

    print(f"Extracting features for {len(stars)} stars...")
    features = []
    
    for i in tqdm(range(len(stars))):
        cam = stars.loc[i, 'camera']
        ccd = stars.loc[i, 'ccd']
        master_trend = ccd_trends.get((cam, ccd), np.zeros(len(t)))
        
        star_feats = process_star(flux[i], cen[i, 0], cen[i, 1], t, bad_mask, master_trend)
        features.append(star_feats)
        
    df_features = pd.DataFrame(features)
    df_final = pd.concat([stars, df_features], axis=1)
    
    out_file = f'{split}_features.csv'
    df_final.to_csv(out_file, index=False)
    print(f"Saved to {out_file}\n")

if __name__ == '__main__':
    build_dataset('train')
    build_dataset('test')