"""
Nyström & Holmqvist (2010) Adaptive Velocity Algorithm
for saccade, fixation, and glissade detection.

Reference:
    Nyström, M., & Holmqvist, K. (2010). An adaptive algorithm for fixation,
    saccade, and glissade detection in eyetracking data.
    Behavior Research Methods, 42(1), 188–204.

Implementation notes:
    - Designed for 1000Hz monocular or binocular data
    - Velocity computed via Savitzky-Golay filter (or simple finite difference)
    - Iterative threshold refinement based on fixation noise floor
    - Hysteresis: separate onset/offset thresholds
    - Glissade detection: post-saccadic low-velocity oscillation
    - Short fixation merging between same-direction saccades
"""

import numpy as np
import pandas as pd
import scipy.signal as ss
from dataclasses import dataclass, field
from typing import Optional
from scipy.interpolate import CubicSpline, interp1d

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EyeEvent:
    t_on: int           # onset index (samples)
    t_off: int          # offset index (samples)
    duration: float     # duration in ms (assumes 1000Hz; scale if needed)
    x_on: float
    y_on: float
    x_off: float
    y_off: float
    x_mean: float
    y_mean: float
    v_peak: float
    v_mean: float
    distance: float     # degrees (Euclidean start→end)
    amplitude: float    # degrees (total path length)
    event_type: str     # 'saccade', 'fixation', 'glissade'


@dataclass
class NHResult:
    saccades:  pd.DataFrame
    fixations: pd.DataFrame
    blinks:    pd.DataFrame
    # glissades: pd.DataFrame
    labels:    np.ndarray   # per-sample: 0=fixation, 1=saccade, 2=glissade
    velocity:  np.ndarray
    threshold_onset:  float
    threshold_offset: float


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

class NystromHolmqvist:
    """
    Parameters
    ----------
    freq : int
        Sampling frequency in Hz (default 1000).
    edge: int
        Visual degree of screen edge (default 20).
    min_saccade_duration : int
        Minimum saccade duration in samples (default 6ms at 1kHz).
    min_fixation_duration : int
        Minimum fixation duration in samples (default 40ms at 1kHz).
    min_glissade_duration : int
        Minimum glissade duration in samples (default 4ms at 1kHz).
    max_glissade_duration : int
        Maximum glissade duration in samples (default 80ms at 1kHz).
    onset_threshold_factor : float
        Multiplier on noise SD for saccade onset threshold (default 6.0).
        Paper uses a data-driven estimate; this scales it.
    offset_ratio : float
        Offset threshold = onset_threshold * offset_ratio (default 0.5).
        Implements hysteresis — offset is lower than onset.
    max_iter : int
        Maximum iterations for threshold refinement (default 100).
    savgol_window : int
        Savitzky-Golay filter window length in samples (default 7).
    savgol_order : int
        Savitzky-Golay polynomial order (default 2).
    merge_interval : int
        Merge fixations separated by saccades < this duration (samples).
        Set to 0 to disable merging.
    max_vel: int
        Biologically plausible saccade velocity, exceeding this counts as artifact
    """

    def __init__(
        self,
        freq: int = 1000,
        edge: int = 30,
        min_saccade_duration: int = 6,
        min_fixation_duration: int = 40,
        min_glissade_duration: int = 4,
        max_glissade_duration: int = 80,
        onset_threshold_factor: float = 6.0,
        offset_ratio: float = 0.5,
        max_iter: int = 100,
        savgol_window: int = 7,
        savgol_order: int = 2,
        merge_interval: int = 75,
        max_vel: int = 1500,
        blink_pad_ms: int = 10,
        interp_method: str = 'pchip',
    ):
        self.freq = freq
        self.edge = edge
        self.min_saccade_duration = min_saccade_duration
        self.min_fixation_duration = min_fixation_duration
        self.min_glissade_duration = min_glissade_duration
        self.max_glissade_duration = max_glissade_duration
        self.onset_threshold_factor = onset_threshold_factor
        self.offset_ratio = offset_ratio
        self.max_iter = max_iter
        self.savgol_window = savgol_window
        self.savgol_order = savgol_order
        self.merge_interval = merge_interval
        self.max_vel = max_vel
        self.blink_pad_ms = blink_pad_ms
        self.interp_method = interp_method


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------


    def fit(self, x: np.ndarray, y: np.ndarray) -> NHResult:
        """
        Detect saccades, fixations, and glissades in a continuous eye trace.

        Parameters
        ----------
        x, y : np.ndarray
            Eye position in degrees of visual angle. NaN = missing/blink.
            Length must match.

        Returns
        -------
        NHResult
        """
        x, y = np.asarray(x, float), np.asarray(y, float)
        assert len(x) == len(y), "x and y must have equal length"

        # 1. Compute velocity
        vel = self._velocity(x, y)

        # 2. Iterative adaptive threshold
        thresh_onset, thresh_offset = self._adaptive_threshold(vel)

        # 3. Detect saccade candidates via hysteresis thresholding
        saccades = self._detect_saccades(vel, thresh_onset, thresh_offset)

        # 4. Detect glissades immediately after each saccade
        # saccades, glissades = self._detect_glissades(vel, saccades, thresh_offset)

        # 5. Build fixations as intervals between saccades (+glissades)
        # all_events_sorted = sorted(saccades + glissades, key=lambda e: e[0])
        fixations = self._build_fixations(saccades, len(x))
        # fixations = [f for f in fixations
        #              if (f[1] - f[0]) >= self.min_fixation_duration]

        # 6. Merge short fixations between same-direction saccades
        if self.merge_interval > 0:
            saccades, fixations = self._merge_fixations(
                saccades, fixations, x, y, vel)

        # 7. Find missing/blinks, and remove that from saccades and fixations
        fixations, saccades, blinks = self._find_blinks(vel, fixations, saccades, x, y)

        # 8. Build label trace
        labels = self._build_labels(len(x), saccades, blinks)

        # 9. Build output DataFrames
        sac_df  = self._build_df(saccades,  x, y, vel, 'saccade')

        fix_df  = self._build_df(fixations, x, y, vel, 'fixation')

        fix_df = fix_df.query('(duration>@self.min_fixation_duration)')
        bli_df = self._build_df(blinks, x, y, vel, 'blink')
        # gli_df  = self._build_df(glissades, x, y, vel, 'glissade')


        return NHResult(
            saccades=sac_df,
            fixations=fix_df,
            blinks=bli_df,
            # glissades=gli_df,
            labels=labels,
            velocity=vel,
            threshold_onset=thresh_onset,
            threshold_offset=thresh_offset,
        )

    # ------------------------------------------------------------------
    # Step 1: Velocity
    # ------------------------------------------------------------------

    def _velocity(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        2D velocity magnitude via Savitzky-Golay first derivative.
        NaN regions are interpolated linearly before filtering,
        then masked again afterward.
        """

        # clip out-of-range gaze positions to NaN
        xi, yi = x.copy().astype(float), y.copy().astype(float)
        out_of_range = (np.abs(xi) >= self.edge) | (np.abs(yi) >= self.edge)
        xi[out_of_range] = np.nan
        yi[out_of_range] = np.nan
 
        # record original NaN mask (includes out-of-range) before interpolation
        nan_mask = np.isnan(xi) | np.isnan(yi)
 
        # interpolate all NaN gaps for smooth velocity computation
        xi, yi, _ = self._interpolate_missing(xi, yi)
 
        # if any NaNs remain (e.g. gap at trace edges), fall back to linear interp
        still_nan = np.isnan(xi) | np.isnan(yi)
        if still_nan.any():
            t = np.arange(len(xi))
            valid = ~still_nan
            if valid.sum() > 1:
                xi = np.interp(t, t[valid], xi[valid])
                yi = np.interp(t, t[valid], yi[valid])

        dx = ss.savgol_filter(xi, window_length=self.savgol_window,
                              polyorder=self.savgol_order, deriv=1,
                              delta=1.0 / self.freq)
        dy = ss.savgol_filter(yi, window_length=self.savgol_window,
                              polyorder=self.savgol_order, deriv=1,
                              delta=1.0 / self.freq)

        vel = np.sqrt(dx**2 + dy**2)          # deg/s

        # when the gaze position is lost track or velocity exceeds maximum biological speed
        vel[vel>self.max_vel] = np.nan

        return vel

    # ------------------------------------------------------------------
    # Step 2: Adaptive threshold (core innovation of the paper)
    # ------------------------------------------------------------------

    def _adaptive_threshold(self, vel: np.ndarray):
        """
        Iteratively estimate the fixation noise floor and derive thresholds.
        Idea is that velocity during fixations should be close to zero, 
        if not converging meaning that below threshold, which is supposed to be fixations, have high velocity

        Algorithm (Nyström & Holmqvist 2010, Section 2.3):
        1. Start with a high initial threshold (mean + 3*std of all velocity)
        2. Mark samples below threshold as "candidate fixation"
        3. Recompute mean and std from candidate fixation samples only
        4. New threshold = mean + onset_threshold_factor * std
        5. Repeat until threshold converges

        The offset threshold = onset_threshold * offset_ratio (hysteresis).
        """
        valid = vel[~np.isnan(vel)]

        # initial high threshold to exclude obvious saccades
        thresh = np.mean(valid) + 3.0 * np.std(valid)

        for _ in range(self.max_iter):
            fixation_samples = valid[valid < thresh]
            if len(fixation_samples) < 10:
                break  # degenerate case

            mu  = np.mean(fixation_samples)
            std = np.std(fixation_samples)
            new_thresh = mu + self.onset_threshold_factor * std

            if np.abs(new_thresh - thresh) < 1e-6:
                break
            thresh = new_thresh

        thresh_onset  = thresh
        thresh_offset = thresh * self.offset_ratio
        return thresh_onset, thresh_offset

    # ------------------------------------------------------------------
    # Step 3: Hysteresis thresholding
    # ------------------------------------------------------------------

    def _detect_saccades(self, vel, thresh_onset, thresh_offset):
        """
        Detect saccade intervals using two thresholds (hysteresis):
        - Onset:  velocity crosses ABOVE thresh_onset
        - Offset: velocity falls BELOW thresh_offset

        This avoids chattering at the boundary between fixation and saccade.
        """
        in_saccade = False
        onset = None
        saccades = []
        n = len(vel)

        for i in range(n):
            v = vel[i]
            if np.isnan(v):
                if in_saccade:          # blink during saccade — close it
                    saccades.append((onset, i - 1))
                    in_saccade = False
                continue

            if not in_saccade:
                if v >= thresh_onset:
                    onset = i
                    in_saccade = True
            else:
                if v < thresh_offset:
                    saccades.append((onset, i))
                    in_saccade = False

        if in_saccade and onset is not None:
            saccades.append((onset, n - 1))

        # only include saccades longer than min duration
        saccades = [s for s in saccades
                    if (s[1] - s[0]) >= self.min_saccade_duration]

        return saccades


    # ------------------------------------------------------------------
    # Step 4: Glissade detection
    # ------------------------------------------------------------------

    def _detect_glissades(self, vel, saccades, thresh_offset):
        """
        Glissades are low-velocity oscillations immediately following a saccade,
        before the eye settles into the next fixation.

        Detection (Nyström & Holmqvist 2010, Section 2.5):
        - After saccade offset, check if velocity shows a secondary peak
          above thresh_offset within max_glissade_duration samples
        - If so, extend saccade offset to end of glissade, mark as glissade

        Returns updated saccade list and separate glissade list.
        """
        updated_saccades = []
        glissades = []
        n = len(vel)

        for (t_on, t_off) in saccades:
            # search window immediately after saccade
            search_end = min(t_off + self.max_glissade_duration, n - 1)
            post_vel = vel[t_off:search_end]

            # glissade: velocity dips below threshold then rises above it again
            below = post_vel < thresh_offset
            above = post_vel >= thresh_offset

            # find transitions: below → above (secondary peak onset)
            secondary_onset = None
            secondary_offset = None

            in_below = False
            for j, (b, a) in enumerate(zip(below, above)):
                if b and not in_below:
                    in_below = True
                if in_below and a:
                    secondary_onset = t_off + j
                    in_below = False
                    # find end of secondary peak
                    for k in range(j, len(post_vel)):
                        if post_vel[k] < thresh_offset:
                            secondary_offset = t_off + k
                            break
                    if secondary_offset is None:
                        secondary_offset = search_end
                    break

            if (secondary_onset is not None and
                    secondary_offset is not None and
                    (secondary_offset - secondary_onset) >= self.min_glissade_duration):
                glissades.append((secondary_onset, secondary_offset))

            updated_saccades.append((t_on, t_off))

        return updated_saccades, glissades

    # ------------------------------------------------------------------
    # Step 5: Build fixations
    # ------------------------------------------------------------------

    def _build_fixations(self, events, n_samples):
        """Fixations are the gaps between saccades and glissades."""
        fixations = []
        prev_end = 0
        for (t_on, t_off) in events:
            if t_on > prev_end:
                fixations.append((prev_end, t_on))
            prev_end = t_off
        if prev_end < n_samples - 1:
            fixations.append((prev_end, n_samples - 1))
        return fixations

    # ------------------------------------------------------------------
    # Step 6: Merge short fixations
    # ------------------------------------------------------------------

    def _merge_fixations(self, saccades, fixations, x, y, vel):
        """
        Merge fixations that are separated by a very short saccade
        (< merge_interval samples) AND where both flanking saccades
        travel in roughly the same direction — indicates a single
        interrupted fixation, not a true sequential fixation pair.
        """
        if len(saccades) < 2:
            return saccades, fixations

        merged_saccades = list(saccades)
        merged_fixations = list(fixations)

        i = 0
        while i < len(merged_saccades) - 1:
            s1 = merged_saccades[i]
            s2 = merged_saccades[i + 1]

            # find the fixation between s1 and s2
            between = [f for f in merged_fixations
                       if f[0] >= s1[1] and f[1] <= s2[0]]

            if not between:
                i += 1
                continue

            fix = between[0]
            fix_dur = fix[1] - fix[0]

            if fix_dur < self.merge_interval:
                # check direction similarity
                d1 = self._saccade_direction(s1, x, y)
                d2 = self._saccade_direction(s2, x, y)
                angle_diff = np.abs(np.degrees(np.arctan2(
                    np.sin(d1 - d2), np.cos(d1 - d2))))

                if angle_diff < 45:
                    # merge: remove the short fixation and bridge saccades
                    merged_fixations.remove(fix)
                    merged_saccades[i] = (s1[0], s2[1])
                    merged_saccades.pop(i + 1)
                    continue
            i += 1

        return merged_saccades, merged_fixations

    def _saccade_direction(self, saccade, x, y):
        t_on, t_off = saccade
        dx = x[t_off] - x[t_on]
        dy = y[t_off] - y[t_on]
        return np.arctan2(dy, dx)


    # ------------------------------------------------------------------
    # Step 7: Find missing/blinks 
    # ------------------------------------------------------------------

    def _find_blinks(self, vel, fixations, saccades, x, y):

        
        blinks = []

        # first find fixations outside of the screen edge
        for fix_idx, (t_on, t_off) in enumerate(fixations):
            
            x_mean = x[t_on:t_off].mean()
            y_mean = y[t_on:t_off].mean()
            
            if np.abs(x_mean)>self.edge or np.abs(y_mean)>self.edge:
                fixations.pop(fix_idx)
                blink_on = t_on
                blink_off = t_off
                
                try:
                    # check preceding saccades
                    sac_idx = np.where(np.isin(saccades, t_on))[0][0]
                    sac_on, sac_off = saccades[sac_idx]
                    preceding_vel = vel[sac_on:sac_off].mean()
                    if np.abs(preceding_vel)>self.max_vel:
                        saccades.pop(sac_idx)
                        blink_on = sac_on
                    preceding = 1
                except:
                    preceding = 0

                # check following saccades
                try:
                    sac_idx = np.where(np.isin(saccades, t_off))[0][0]
                    sac_on, sac_off = saccades[sac_idx]
                    following_vel = vel[sac_on:sac_off].mean()
                    if np.abs(following_vel)>self.max_vel:
                        saccades.pop(sac_idx)
                        blink_off = sac_off
                    following = 1
                except:
                    following = 0

                blinks.append((blink_on, blink_off))

                # if preceding: 
                #     if following:
                #         print(fix_idx, "Preceding and following blinks")
                #     else:
                #         print(fix_idx, "Preceding blinks") 
                # else:
                #     if following:
                #         print(fix_idx, "Following blinks")
                #     else:
                #         print(fix_idx, "Neither")

        # second, find saccades that are outside of the screen
        for sac_idx, (t_on, t_off) in enumerate(saccades):
            
            x_max = np.abs(x[t_on:t_off]).max()
            y_max = np.abs(y[t_on:t_off]).max()
            
            if x_max>self.edge or y_max>self.edge:
                saccades.pop(sac_idx)
                blink_on = t_on
                blink_off = t_off
                blinks.append((blink_on, blink_off))

        return fixations, saccades, blinks


    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _interpolate_missing(self, x, y):
        """
        Interpolate ALL NaN gaps in eye position traces.
 
        Parameters
        ----------
        x, y : np.ndarray
            Eye position with NaN for missing samples.
 
        Returns
        -------
        x_clean, y_clean : np.ndarray
            Interpolated traces.
        gap_mask : np.ndarray bool
            True where data was originally missing (before padding).
        """
 
        pad_samples = int(self.blink_pad_ms * self.freq / 1000)
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
 
        gap_mask = np.isnan(x) | np.isnan(y)
        padded_mask = _pad_mask(gap_mask, pad_samples)
        x[padded_mask] = np.nan
        y[padded_mask] = np.nan
 
        gaps = _find_gaps(padded_mask)
        x_clean = x.copy()
        y_clean = y.copy()
 
        for (gap_start, gap_end) in gaps:
            left  = gap_start - 1
            right = gap_end
 
            if left < 0 or right >= len(x):
                continue
            if np.isnan(x[left]) or np.isnan(x[right]):
                continue
 
            t_gap = np.arange(gap_start, gap_end)
            x_clean[gap_start:gap_end] = _interpolate_segment(
                left, right, t_gap, x, self.interp_method)
            y_clean[gap_start:gap_end] = _interpolate_segment(
                left, right, t_gap, y, self.interp_method)
 
        return x_clean, y_clean, gap_mask


    def _build_labels(self, n, saccades, blinks):
        labels = np.zeros(n, dtype=int)   # 0 = fixation
        for (t_on, t_off) in saccades:
            labels[t_on:t_off] = 1        # 1 = saccade
        for (t_on, t_off) in blinks:
            labels[t_on:t_off] = 2        # 2 = blinks
        # for (t_on, t_off) in glissades:
        #     labels[t_on:t_off] = 3        # 3 = glissade

        return labels

    def _build_df(self, events, x, y, vel, event_type):
        if not events:
            return pd.DataFrame(columns=[
                't_on', 't_off', 'duration', 'x_on', 'y_on',
                'x_off', 'y_off', 'x_mean', 'y_mean',
                'v_peak', 'v_mean', 'distance', 'amplitude', 'event_type'])

        rows = []
        for (t_on, t_off) in events:
            dur = (t_off - t_on) / self.freq * 1000  # ms
            seg_x = x[t_on:t_off]
            seg_y = y[t_on:t_off]
            seg_v = vel[t_on:t_off]

            distance = np.sqrt((x[t_off] - x[t_on])**2 +
                                (y[t_off] - y[t_on])**2)
            # amplitude = total path length
            dx = np.diff(seg_x)
            dy = np.diff(seg_y)
            amplitude = np.nansum(np.sqrt(dx**2 + dy**2))

            rows.append({
                't_on':       t_on,
                't_off':      t_off,
                'duration':   dur,
                'x_on':       x[t_on],
                'y_on':       y[t_on],
                'x_off':      x[t_off] if t_off < len(x) else np.nan,
                'y_off':      y[t_off] if t_off < len(y) else np.nan,
                'x_mean':     np.nanmean(seg_x),
                'y_mean':     np.nanmean(seg_y),
                'v_peak':     np.nanmax(seg_v) if len(seg_v) else np.nan,
                'v_mean':     np.nanmean(seg_v) if len(seg_v) else np.nan,
                'distance':   distance,
                'amplitude':  amplitude,
                'event_type': event_type,
            })

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience wrapper matching your existing code's interface
# ---------------------------------------------------------------------------

def detect_saccades_NH(x, y, freq=1000, **kwargs):
    """
    Drop-in wrapper returning (saccades_df, fixations_df, glissades_df, labels, velocity).

    Parameters
    ----------
    x, y : array-like
        Eye position in degrees. NaN for missing samples.
    freq : int
        Sampling frequency in Hz.
    **kwargs
        Passed to NystromHolmqvist constructor.

    Example
    -------
    sac, fix, gli, labels, vel = detect_saccades_NH(eye_x, eye_y, freq=1000)
    """
    model = NystromHolmqvist(freq=freq, **kwargs)
    result = model.fit(np.asarray(x), np.asarray(y))
    return result


# ---------------------------------------------------------------------------
# Helper functions for interpolating missing values in eye tracking
# ---------------------------------------------------------------------------

def _pad_mask(mask, pad):
    """Expand True regions by pad samples on each side."""
    if pad == 0:
        return mask.copy()
    padded = mask.copy()
    indices = np.where(mask)[0]
    for i in indices:
        lo = max(0, i - pad)
        hi = min(len(mask), i + pad + 1)
        padded[lo:hi] = True
    return padded

def _find_gaps(mask):
    """Return list of (start, end) indices for contiguous True runs."""
    gaps = []
    in_gap = False
    start = None
    for i, v in enumerate(mask):
        if v and not in_gap:
            start = i
            in_gap = True
        elif not v and in_gap:
            gaps.append((start, i))
            in_gap = False
    if in_gap:
        gaps.append((start, len(mask)))
    return gaps

def _interpolate_segment(left, right, t_gap, signal, method):
    """Interpolate between two anchor points."""
    # use a few samples on each side as anchors for smoother fit
    n_anchor = 5
    t_left  = np.arange(max(0, left - n_anchor), left + 1)
    t_right = np.arange(right, min(len(signal), right + n_anchor + 1))

    t_known = np.concatenate([t_left, t_right])
    v_known = signal[t_known]

    # drop any NaNs in anchor region
    valid = ~np.isnan(v_known)
    t_known = t_known[valid]
    v_known = v_known[valid]

    if len(t_known) < 2:
        return np.full(len(t_gap), np.nan)

    if method == 'cubic' and len(t_known) >= 4:
        interp = CubicSpline(t_known, v_known, extrapolate=False)
    elif method == 'pchip' and len(t_known) >= 2:
        from scipy.interpolate import PchipInterpolator
        interp = PchipInterpolator(t_known, v_known, extrapolate=False)
    else:
        interp = interp1d(t_known, v_known, kind='linear',
                        bounds_error=False, fill_value=np.nan)

    return interp(t_gap)