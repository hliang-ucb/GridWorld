import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from scipy.signal import argrelmin
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import nystrom_holmqvist as NH

EVENT_CODES = {'trial_start': 9,
            'center_on': 22,
            'fixation_on': 23,
            'fail_to_fixate': 24,
            'maze_on': 25,
            'joystick_on': 26,
            'small_reward': 27,
            'reach_target': 28,
            'maze_off': 29,
            'trial_end': 18              
            }


# ------------------------------------------------------------------
# Eye data / Gaze pattern
# ------------------------------------------------------------------
def get_classified_gaze(date, start='maze_on', stop='joystick_on',
                         data_dir='D:/Newworld/Data', freq=1000,
                         xy_bound=25, v_max=1000, dur_max=2000,
                         **center_kwargs):
  
    mat = loadmat(f'{data_dir}/London_{date}.mat', simplify_cells=True)
    eyedata = pd.DataFrame(mat['eye'])
    eventcode = pd.DataFrame(mat['eventcode'])
    behavior = pd.DataFrame(mat['behavior'])

    fixations, saccades = get_gaze(
        eyedata, eventcode, behavior, start, stop,
        freq=freq, xy_bound=xy_bound, v_max=v_max,
    )
    return classify_center(fixations, saccades, **center_kwargs)


def get_gaze(eyedata, eventcode, behavior, start, stop,
             freq=1000, xy_bound=25, v_max=1000, dur_max=2000):
    """
    Extract and classify fixations and saccades for all trials.

    Two-pass design:
      Pass 1 — collect raw eye traces from every trial, compute a single
               adaptive velocity threshold pooled across the whole session.
      Pass 2 — classify each trial using that fixed session threshold so
               every trial is held to the same standard regardless of how
               many blinks or saccades it contains individually.
    """
    start_code = EVENT_CODES[start]
    stop_code  = EVENT_CODES[stop]
    trials     = behavior.Trial.unique()

    # ------------------------------------------------------------------
    # Pass 1: collect traces + compute session-level threshold
    # ------------------------------------------------------------------
    model = NH.NystromHolmqvist(freq=freq, edge=xy_bound, max_vel=v_max)

    trace_cache = {}          # trial → (x, y, epoch_on, epoch_off)
    x_list, y_list = [], []

    for trial in trials:
        try:
            epoch_on  = int(eventcode.query('(Number==@start_code) & (Trial==@trial)').Time.values[0])
            epoch_off = int(eventcode.query('(Number==@stop_code)  & (Trial==@trial)').Time.values[0])
            trace = eyedata.query('Trial==@trial').Eye.values[0]
            x, y  = trace[:, 0], trace[:, 1]
            trace_cache[trial] = (x, y, epoch_on, epoch_off)
            x_list.append(x)
            y_list.append(y)
        except Exception as e:
            print(f'Trial {trial}: skipped in pass 1 ({type(e).__name__}: {e})')

    # fits session threshold and caches it on the model
    model.compute_session_threshold(x_list, y_list)

    # ------------------------------------------------------------------
    # Pass 2: classify each trial with the fixed session threshold
    # ------------------------------------------------------------------
    fixation_list = []
    saccade_list  = []

    for trial, (x, y, epoch_on, epoch_off) in trace_cache.items():
        try:
            # model.fit() picks up session_thresh_onset/offset automatically
            result = model.fit(x, y)

            trial_fixations = result.fixations.query(
                '(t_off > @epoch_on) & (t_on < @epoch_off)'
            ).copy()
            trial_saccades = result.saccades.query(
                '(t_off > @epoch_on) & (t_on < @epoch_off)'
            ).copy()

            for df in (trial_fixations, trial_saccades):
                df.insert(0, 'Trial', trial)
                df[start] = epoch_on
                df[stop]  = epoch_off

            fixation_list.append(trial_fixations)
            saccade_list.append(trial_saccades)

        except Exception as e:
            print(f'Trial {trial}: skipped in pass 2 ({type(e).__name__}: {e})')

    # ------------------------------------------------------------------
    # Merge, join behavior, apply post-hoc filters
    # ------------------------------------------------------------------
    fixation_list = [df for df in fixation_list if not df.empty]
    saccade_list  = [df for df in saccade_list  if not df.empty]

    all_fixations = pd.concat(fixation_list, ignore_index=True)
    all_saccades  = pd.concat(saccade_list,  ignore_index=True)

    trial_behavior = behavior.drop_duplicates('Trial')
    all_fixations  = all_fixations.merge(trial_behavior, on='Trial')
    all_saccades   = all_saccades.merge(trial_behavior,  on='Trial')

    all_fixations = all_fixations[
        (all_fixations.x_mean.abs() < xy_bound) &
        (all_fixations.y_mean.abs() < xy_bound) &
        (all_fixations.duration     < dur_max)
    ]
    all_fixations['n_fixations'] = all_fixations.groupby('Trial')['Trial'].transform('size')
    all_saccades  = all_saccades[all_saccades.v_mean < v_max]

    return all_fixations, all_saccades


# ------------------------------------------------------------------
# Center / periphery classification
# ------------------------------------------------------------------

def _estimate_center_xy(X, trim_frac=0.05, n_iter=5, tol=1e-3):
    """
    Iterative estimate of the true center location to handle calibration drift
    """
    center_xy = np.zeros(2)
    n_trim = max(int(len(X) * trim_frac), 5)

    for _ in range(n_iter):
        r = np.hypot(X[:, 0] - center_xy[0], X[:, 1] - center_xy[1])
        closest = X[np.argsort(r)[:n_trim]]
        new_center = closest.mean(axis=0)
        if np.hypot(*(new_center - center_xy)) < tol:
            center_xy = new_center
            break
        center_xy = new_center

    return center_xy


def _center_cluster_radial(X, center_xy, r_max, bw_method, n_grid, valley_order,
                            trim_frac=0.05, n_iter=5):
    """
    Collapse center-vs-periphery to 1D: compute radial distance from
    center_xy, then find the first density valley past the initial peak
    in a KDE of that radial distance. 
    """
    def _valley_pass(x0, y0):
        r = np.hypot(X[:, 0] - x0, X[:, 1] - y0)
        r_max_ = r_max if r_max is not None else np.percentile(r, 99)
        grid = np.linspace(0, r_max_, n_grid)
        density = gaussian_kde(r, bw_method=bw_method)(grid)
        minima_idx = argrelmin(density, order=valley_order)[0]
        if len(minima_idx) == 0:
            raise ValueError(
                'No valley found in the radial density profile -- try adjusting '
                'r_max or bw_method, or inspect np.hypot(x_mean, y_mean) directly.'
            )
        return r, grid[minima_idx[0]]

    auto = isinstance(center_xy, str) and center_xy == 'auto'
    if center_xy is None:
        x0, y0 = 0.0, 0.0
    elif auto:
        x0, y0 = _estimate_center_xy(X, trim_frac=trim_frac, n_iter=n_iter)
    else:
        x0, y0 = center_xy

    r, center_radius = _valley_pass(x0, y0)
    center_mask = r < center_radius

    if auto:
        x0, y0 = X[center_mask].mean(axis=0)
        r, center_radius = _valley_pass(x0, y0)
        center_mask = r < center_radius

    return center_mask, center_radius, (x0, y0)


def classify_center(fixations, saccades, method='radial',
                     center_xy='auto', r_max=None, bw_method=None,
                     n_grid=500, valley_order=5,
                     search_radius=None,
                     n_components=None, k_range=range(1, 8),
                     min_cluster_size=30, std_thresh=3, max_iter=20):
    
    fixations = fixations.copy()
    saccades = saccades.copy()
    X = fixations[['x_mean', 'y_mean']].values

    center_mask, found_radius, resolved_center_xy = _center_cluster_radial(
        X, center_xy, r_max, bw_method, n_grid, valley_order
    )
    fixations['cluster'] = center_mask.astype(int)
    fixations['center'] = center_mask
    fixations.attrs['center_radius'] = found_radius

    center_x, center_y = fixations.loc[fixations['center'], ['x_mean', 'y_mean']].mean()

    # Recenter position columns so the estimated true center becomes the
    # origin -- downstream code no longer has to subtract center_x/center_y
    # itself, and it's explicit that (0, 0) now means "true center".
    for col in ('x_on', 'x_off', 'x_mean'):
        if col in fixations.columns:
            fixations[col] = fixations[col] - center_x
        if col in saccades.columns:
            saccades[col] = saccades[col] - center_x
    for col in ('y_on', 'y_off', 'y_mean'):
        if col in fixations.columns:
            fixations[col] = fixations[col] - center_y
        if col in saccades.columns:
            saccades[col] = saccades[col] - center_y

    fixations.attrs['center_xy'] = (center_x, center_y)  # offset that was subtracted

    r_fixation = np.hypot(
        fixations.loc[fixations['center'], 'x_mean'],
        fixations.loc[fixations['center'], 'y_mean'],
    )
    center_radius = np.percentile(r_fixation, 95)

    saccades['on_center'] = np.hypot(saccades.x_on, saccades.y_on) < center_radius
    saccades['off_center'] = np.hypot(saccades.x_off, saccades.y_off) < center_radius
    saccades['direction'] = (
        np.where(saccades.on_center, 'center', 'periphery') + '_to_' +
        np.where(saccades.off_center, 'center', 'periphery')
    )

    return fixations, saccades



def plot_saccades(saccades, edge=15, line_alpha=0.1, dot_alpha=0.1):
    
    fig, axes = plt.subplots(2,2,figsize=(8,8),gridspec_kw={'wspace': 0.4, 'hspace': 0.4})
    
    for n, direction in enumerate(saccades.direction.unique()):
        
        df = saccades.query('direction==@direction')
        segments = np.array([
            [[row.x_on, row.y_on], [row.x_off, row.y_off]]
            for _, row in df.iterrows()
        ])
        
        lc = LineCollection(
            segments,
            alpha=line_alpha,
            linewidths=2,
            color='grey',
            zorder=1
        )
        
        ax = axes[np.divmod(n,2)]
        ax.add_collection(lc)
        
        sns.scatterplot(data=df, x='x_on', y='y_on', label='on', ax=ax,
                        linewidth=0, alpha=dot_alpha, zorder=2)
        sns.scatterplot(data=df, x='x_off', y='y_off', label='off', ax=ax,
                        linewidth=0, alpha=dot_alpha, zorder=2)
        ax.autoscale()
        ax.set_xlim(-edge,edge)
        ax.set_ylim(-edge,edge)
        ax.set_title(direction)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
    sns.despine()