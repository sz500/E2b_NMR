import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os
import sys
import re

def exponential_decay(t, M0, T2, C):
    """
    Exponential decay function (T2 relaxation): M_z = M_0 * exp(-t/T2) + C
    
    Args:
        t: Time in ms
        M0: Maximum magnetization (initial value)
        T2: T2 relaxation time constant (spin-spin relaxation)
        C: Constant offset (background signal)
    
    Returns:
        M_z: Magnetization at time t
    """
    return M0 * np.exp(-t / T2) + C

def get_mirrored_figures_path(filepath):
    """
    Get the output figures path that mirrors the data directory structure.
    
    Args:
        filepath: Path to the data file
    
    Returns:
        output_dir: Path to save figures with same subfolder structure as data
    """
    # Find E2b_NMR directory
    file_abs = os.path.abspath(filepath)
    
    # Look for E2b_NMR in the path
    parts = file_abs.split(os.sep)
    e2b_index = None
    for i, part in enumerate(parts):
        if part == 'E2b_NMR':
            e2b_index = i
            break
    
    if e2b_index is None:
        # Fallback: just use a 'figures' subdirectory in current location
        return os.path.join(os.getcwd(), 'figures')
    
    # Get the path relative to E2b_NMR directory
    e2b_root = os.sep.join(parts[:e2b_index + 1])
    
    # Get the directory structure after 'data'
    data_index = None
    for i in range(e2b_index, len(parts)):
        if parts[i] == 'data':
            data_index = i
            break
    
    if data_index is None:
        return os.path.join(os.getcwd(), 'figures')
    
    # Get relative path after 'data'
    relative_path = os.sep.join(parts[data_index + 1:-1])  # Exclude filename
    
    # Create figures path with same structure
    figures_path = os.path.join(e2b_root, 'figures', relative_path)
    
    return figures_path

def find_peaks_in_data(time, voltage, prominence=0.1, distance=50, height_threshold=None, min_width=40):
    """
    Find peaks in the voltage data.
    
    Args:
        time: Time array
        voltage: Voltage array
        prominence: Minimum prominence of peaks
        distance: Minimum distance between peaks (in samples)
        height_threshold: Minimum height threshold (default: 10% of max voltage)
        min_width: Minimum peak width in samples (filters out narrow noise spikes)
    
    Returns:
        peak_indices: Indices of peaks
        peak_times: Time values at peaks
        peak_voltages: Voltage values at peaks
    """
    # Auto-set height threshold if not provided (10% of max voltage)
    if height_threshold is None:
        height_threshold = np.max(voltage) * 0.01
    
    peak_indices, properties = find_peaks(voltage, prominence=prominence, distance=distance, height=height_threshold, width=min_width)
    peak_times = time[peak_indices]
    peak_voltages = voltage[peak_indices]
    
    return peak_indices, peak_times, peak_voltages

def fit_curve(filepath, prominence=0.5, distance=100, height_threshold=None, min_width=40, skip_peaks=None):
    """
    Perform exponential recovery curve fitting on NMR T2 data.
    
    Args:
        filepath: Path to the CSV file
        prominence: Peak prominence threshold
        distance: Minimum distance between peaks
        height_threshold: Minimum peak height (default: 10% of max voltage)
        min_width: Minimum peak width in samples (default: 40)
        skip_peaks: List of peak indices to skip (1-based, e.g., [1, 3] skips 1st and 3rd)
    
    Returns:
        M0: Fitted M0 parameter
        T2: Fitted T2 parameter
        T2_err: Error in T2 parameter
        peaks: Dictionary with peak information
    """
    # Read the CSV file - handle both old format (with units/blank rows) and new format
    try:
        # Try reading with proper headers (new format: header + data)
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
    except Exception as e:
        # Fallback for old format with skiprows
        try:
            df = pd.read_csv(filepath, skiprows=3)
        except:
            print(f"Error reading {filepath}: {e}")
            return None, None, None, None
    
    # Extract time and voltage data and convert to numeric
    try:
        time = pd.to_numeric(df['Time'], errors='coerce').values  # Time column (ms)
        voltage = pd.to_numeric(df['Channel A'], errors='coerce').values  # Channel A column (V)
        
        # Remove NaN values that result from coercion
        mask = ~(np.isnan(time) | np.isnan(voltage))
        time = time[mask]
        voltage = voltage[mask]
        
        if len(time) == 0 or len(voltage) == 0:
            print(f"Error reading {filepath}: No valid numeric data found")
            return None, None, None, None
    except Exception as e:
        print(f"Error converting data to numeric: {e}")
        return None, None, None, None
    
    # Find peaks
    peak_indices, peak_times, peak_voltages = find_peaks_in_data(time, voltage, prominence=prominence, distance=distance, height_threshold=height_threshold, min_width=min_width)
    
    # Check if enough peaks found
    if len(peak_times) < 3:
        print(f"Warning: {os.path.basename(filepath)} - Not enough peaks found (need at least 3).")
        return None, None, None, None
    
    # Filter out peaks to skip (convert 1-based to 0-based indexing)
    if skip_peaks:
        skip_indices = set(int(idx) - 1 for idx in skip_peaks if 1 <= int(idx) <= len(peak_times))
        mask = np.array([i not in skip_indices for i in range(len(peak_times))])
        peak_times_fit = peak_times[mask]
        peak_voltages_fit = peak_voltages[mask]
        peak_times = peak_times[mask]
        peak_voltages = peak_voltages[mask]
        
        if len(peak_times_fit) < 3:
            print(f"Warning: {os.path.basename(filepath)} - Not enough peaks after skipping (need at least 3).")
            return None, None, None, None
    else:
        # Use all detected peaks for fitting
        peak_times_fit = peak_times
        peak_voltages_fit = peak_voltages
    
    # Perform curve fitting
    try:
        # Initial guess: M0 = max voltage - min voltage (amplitude), T2 = estimated from decay slope
        M0_guess = np.max(peak_voltages_fit) - np.min(peak_voltages_fit)
        C_guess = np.min(peak_voltages_fit) - 0.1 * M0_guess  # Small offset below minimum
        
        # Estimate T2 from the slope of decay between first and last peak
        # Using: M_z = M_0 * exp(-t/T2) + C → ln((M_z - C)/M_0) = -t/T2
        adjusted_first = peak_voltages_fit[0] - C_guess
        adjusted_last = peak_voltages_fit[-1] - C_guess
        
        if adjusted_first > 0 and adjusted_last > 0:
            ln_ratio = np.log(adjusted_last / adjusted_first)
            time_diff = peak_times_fit[-1] - peak_times_fit[0]
            if time_diff > 0:
                T2_guess = -time_diff / ln_ratio
                # Ensure T2_guess is positive and reasonable
                T2_guess = abs(T2_guess)
            else:
                T2_guess = 200
        else:
            T2_guess = 200
        
        # Add constant measurement error to all data points
        measurement_error = 0.02 # V (constant error for all measurements)
        sigma = np.full_like(peak_voltages_fit, measurement_error)
        
        popt, pcov = curve_fit(exponential_decay, peak_times_fit, peak_voltages_fit, 
                               p0=[M0_guess, T2_guess, C_guess], maxfev=10000, sigma=sigma, absolute_sigma=True)
        
        M0, T2, C = popt
        T2_err = np.sqrt(np.diag(pcov))[1]  # Standard error in T2 only
        # Calculate residuals and reduced chi-square for diagnostics
        residuals = peak_voltages_fit - exponential_decay(peak_times_fit, M0, T2, C)
        chi2 = np.sum((residuals / sigma) ** 2)
        reduced_chi2 = chi2 / (len(peak_voltages_fit) - 3)
        
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        return None, None, None, None
    
    # Generate fitted curve
    t_smooth = np.linspace(peak_times.min(), peak_times.max(), 500)
    fitted_curve = exponential_decay(t_smooth, M0, T2, C)
    
    # Generate error bands for T2 uncertainty
    # Upper and lower curves using ±1 std dev in T2
    T2_upper = T2 + T2_err
    T2_lower = T2 - T2_err
    curve_upper = exponential_decay(t_smooth, M0, T2_upper, C)
    curve_lower = exponential_decay(t_smooth, M0, T2_lower, C)
    
    # Create and save the plot with mirrored directory structure
    figures_dir = get_mirrored_figures_path(filepath)
    os.makedirs(figures_dir, exist_ok=True)
    filename = os.path.basename(filepath)
    output_name = os.path.splitext(filename)[0] + '_fitted.png'
    output_path = os.path.join(figures_dir, output_name)
    
    with plt.rc_context({
        'font.family': 'serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'legend.fontsize': 8,
        'savefig.bbox': 'tight'
    }):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(4.8, 5.0),
            gridspec_kw={'height_ratios': [3, 1]}, sharex=True
        )

        # Raw trace in the background for data context
        ax1.plot(time, voltage, color='gray', alpha=0.25, linewidth=0.8, zorder=0, label='Original data')

        # Peaks used for fitting
        ax1.errorbar(
            peak_times_fit,
            peak_voltages_fit,
            yerr=sigma,
            fmt='o',
            color='black',
            ecolor='black',
            elinewidth=0.8,
            capsize=2,
            capthick=0.8,
            markersize=3,
            markeredgewidth=1,
            zorder=2,
            label='Peak amplitudes'
        )

        # Fit and uncertainty band
        ax1.fill_between(
            t_smooth,
            curve_lower,
            curve_upper,
            color='steelblue',
            alpha=0.12,
            zorder=1,
            label=f'T2 uncertainty (±1σ: ±{T2_err:.2g} ms)'
        )
        ax1.plot(
            t_smooth,
            fitted_curve,
            linestyle='--',
            color='steelblue',
            linewidth=1.5,
            zorder=3,
            label=f'Fit: $T_2 = {T2:.3g} \\pm {T2_err:.2g}$ ms'
        )

        ax1.set_ylabel(r'Signal Amplitude $\propto$ $M_{xy}$ (V)')
        ax1.minorticks_on()
        ax1.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.25)
        ax1.legend(frameon=False)

        # Residuals plot
        ax2.errorbar(
            peak_times_fit,
            residuals,
            yerr=sigma,
            fmt='o',
            color='black',
            ecolor='black',
            elinewidth=0.8,
            capsize=2,
            capthick=0.8,
            markersize=3,
            markeredgewidth=1
        )
        ax2.axhline(0, color='steelblue',linestyle='--', linewidth=1.0)
        ax2.set_xlabel(r'Delay Time 2$\tau$ (ms)')
        ax2.set_ylabel('Residuals (V)')
        ax2.minorticks_on()
        ax2.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.25)

        fig.tight_layout()
        fig.savefig(output_path, dpi=600)
        plt.show()
        plt.close(fig)
    
    print(f"✓ {filename}")
    print(f"  M0 = {M0:.6f} V")
    print(f"  T2 = {T2:.3g} ± {T2_err:.3g} ms")
    print(f"  C  = {C:.6f} V")
    
    peaks = {
        'times': peak_times,
        'voltages': peak_voltages,
        'count': len(peak_times)
    }
    
    return M0, T2, T2_err, peaks

def fit_multiple_files(data_folder='data/T2', prominence=0.5, distance=50, height_threshold=None, min_width=40, skip_peaks=None):
    """
    Fit curves for all CSV files in a folder.
    
    Args:
        data_folder: Folder containing CSV files
        prominence: Peak prominence threshold
        distance: Minimum distance between peaks
        height_threshold: Minimum peak height (default: 10% of max voltage)
        skip_peaks: List of peak indices to skip (1-based)
        min_width: Minimum peak width in samples (default: 40)
    """
    import glob
    
    csv_files = glob.glob(os.path.join(data_folder, '**/*.csv'), recursive=True)
    
    if not csv_files:
        print(f"No CSV files found in {data_folder}")
        return
    
    results = []
    for filepath in csv_files:
        print(f"\nProcessing: {filepath}")
        M0, T2, T2_err, peaks = fit_curve(filepath, prominence=prominence, distance=distance, height_threshold=height_threshold, min_width=min_width, skip_peaks=skip_peaks)
        
        if M0 is not None and T2 is not None:
            filename = os.path.basename(filepath)
            results.append({
                'filename': filename,
                'filepath': filepath,
                'M0': M0,
                'T2': T2,
                'T2_err': T2_err,
                'peak_count': peaks['count']
            })
    
    # Summary table
    if results:
        print(f"\n{'='*100}")
        print("SUMMARY OF ALL FITS")
        print(f"{'='*100}")
        print(f"{'Filename':<40} {'M0 (V)':<15} {'T2 (ms)':<30}")
        print(f"{'-'*100}")
        for result in results:
            T2_str = f"{result['T2']:.4f} ± {result['T2_err']:.4f}"
            print(f"{result['filename']:<40} {result['M0']:<15.6f} {T2_str:<30}")
        print(f"{'='*100}\n")

# Main execution
if __name__ == '__main__':

    prominence = 0.13
    distance = 500
    height_threshold = 0.0
   
    min_width = 0.5
    skip_peaks = None # use 1, if want to skip the first; use 2,3,5 if want to skip 2nd, 3rd, 5th
    
    if len(sys.argv) < 2:
        print("Curve Fitting Tool for NMR T2 Data")
        print("=" * 70)
        print("Usage:")
        print("  python T2_exp_fit.py all [folder] [prom] [dist] [height] [width] [skip]     - Fit all files")
        print("  python T2_exp_fit.py <filepath> [prom] [dist] [height] [width] [skip]       - Fit single file")
        print("=" * 70)
        print("\nParameters:")
        print("  folder:     Data folder path (default: data/T2)")
        print("  prom:       Peak prominence (default: 0.1)")
        print("  dist:       Peak distance in samples (default: 50)")
        print("  height:     Minimum peak height in volts (default: 0.25)")
        print("  width:      Minimum peak width in samples (default: 40)")
        print("  skip:       Peak indices to skip (1-based, comma-separated, e.g., 1,3)")
        print("\nExample:")
        print("  python T2_exp_fit.py all")
        print("  python T2_exp_fit.py all data/T2")
        print("  python T2_exp_fit.py all data/T2 0.5 100 0.2 50")
        print("  python T2_exp_fit.py data/T2/file.csv")
        print("  python T2_exp_fit.py data/T2/file.csv 0.5 100 0.2 50 1,3")
        sys.exit(0)
    
    arg = sys.argv[1].strip().strip("'\"")
    
    # Parse skip_peaks from last argument if it looks like a number or comma-separated numbers
    if len(sys.argv) > 2:
        last_arg = sys.argv[-1].strip().strip("'\"")
        # Check if it's a number or comma-separated numbers and not a file path
        if not os.path.exists(last_arg) and not '/' in last_arg:
            try:
                # Handle both single number and comma-separated numbers
                if ',' in last_arg:
                    skip_peaks = [int(x.strip()) for x in last_arg.split(',')]
                else:
                    skip_peaks = [int(last_arg)]
            except ValueError:
                skip_peaks = None
    
    if arg.lower() == 'all':
        # Handle 'all' command with optional folder and parameters
        data_folder = 'data/T2'
        if len(sys.argv) > 2:
            # Check if next argument is a folder path
            arg2 = sys.argv[2].strip().strip("'\"")
            if os.path.isdir(arg2) or '/' in arg2 or arg2.startswith('data'):
                data_folder = arg2
                # Check if prominence is specified after folder
                if len(sys.argv) > 3:
                    try:
                        prominence = float(sys.argv[3])
                    except ValueError:
                        pass
                if len(sys.argv) > 4:
                    try:
                        height_threshold = float(sys.argv[5])
                    except ValueError:
                        pass
                if len(sys.argv) > 5:
                    try:
                        min_width = int(sys.argv[6])
                    except ValueError:
                        pass
                if len(sys.argv) > 6:
                    try:
                        min_width = int(sys.argv[6])
                    except ValueError:
                        pass
            else:
                # Treat as prominence parameter
                try:
                    prominence = float(arg2)
                    if len(sys.argv) > 3:
                        try:
                            distance = int(sys.argv[3])
                        except ValueError:
                            pass
                    if len(sys.argv) > 4:
                        try:
                            height_threshold = float(sys.argv[4])
                        except ValueError:
                            pass
                    if len(sys.argv) > 5:
                        try:
                            min_width = int(sys.argv[5])
                        except ValueError:
                            pass
                except ValueError:
                    pass
        
        fit_multiple_files(data_folder, prominence=prominence, distance=distance, height_threshold=height_threshold, min_width=min_width, skip_peaks=skip_peaks)
    else:
        # Handle single file with optional parameters
        if os.path.exists(arg):
            # Check if next argument is prominence parameter
            if len(sys.argv) > 2:
                arg2 = sys.argv[2].strip().strip("'\"")
                try:
                    # Try to parse as prominence
                    prominence = float(arg2)
                    # Then check for distance
                    if len(sys.argv) > 3:
                        try:
                            distance = int(sys.argv[3])
                        except ValueError:
                            pass
                    if len(sys.argv) > 4:
                        try:
                            height_threshold = float(sys.argv[4])
                        except ValueError:
                            pass
                    if len(sys.argv) > 5:
                        try:
                            min_width = int(sys.argv[5])
                        except ValueError:
                            pass
                except ValueError:
                    pass
            
            fit_curve(arg, prominence=prominence, distance=distance, height_threshold=height_threshold, min_width=min_width, skip_peaks=skip_peaks)
        else:
            print(f"Error: File '{arg}' not found.")
