import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os
import sys
import glob

def linear_model(t, slope, intercept):
    """
    Linear model: y = slope*t + intercept
    
    Args:
        t: Time in ms
        slope: Slope of the line (1/T2 for T2 relaxation)
        intercept: Y-intercept
    
    Returns:
        Linear fit value at time t
    """
    return slope * t + intercept

def find_peaks_in_data(time, voltage, prominence=0.1, distance=50, height_threshold=None, min_width=100):
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
    # Auto-set height threshold if not provided (8% of max voltage)
    if height_threshold is None:
        height_threshold = np.max(voltage) * 0.08
    
    peak_indices, properties = find_peaks(voltage, prominence=prominence, distance=distance, height=height_threshold, width=min_width)
    peak_times = time[peak_indices]
    peak_voltages = voltage[peak_indices]
    
    return peak_indices, peak_times, peak_voltages

def fit_t2_linearized(filepath, prominence=0.5, distance=50, height_threshold=None, figures_dir='figures'):
    """
    Perform T2 relaxation curve fitting using linearization: ln(M0/Mz) vs t
    T2 = 1 / slope
    
    Args:
        filepath: Path to the CSV file
        prominence: Peak prominence threshold
        distance: Minimum distance between peaks
        height_threshold: Minimum peak height (default: 8% of max voltage)
        figures_dir: Directory to save figures
    
    Returns:
        T2: Fitted T2 relaxation time
        T2_err: Standard error in T2
        slope: Slope of linearized fit (1/T2)
        slope_err: Standard error in slope
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
            return None, None, None, None, None
    
    # Extract time and voltage data
    time = df['Time'].values  # Time column (ms)
    voltage = df['Channel A'].values  # Channel A column (V)
    
    # Find peaks
    peak_indices, peak_times, peak_voltages = find_peaks_in_data(time, voltage, prominence=prominence, distance=distance, height_threshold=height_threshold)
    
    # Check if enough peaks found
    if len(peak_times) < 3:
        print(f"Warning: {os.path.basename(filepath)} - Not enough peaks found (need at least 3).")
        return None, None, None, None, None
    
    # Calculate M0 as the first peak (highest amplitude)
    M0 = peak_voltages[0]
    
    # Calculate ln(M0/Mz) for each peak
    # Avoid log of values <= 0
    peak_voltages_fit = peak_voltages[peak_voltages > 0]
    peak_times_fit = peak_times[peak_voltages > 0]
    
    if len(peak_times_fit) < 3:
        print(f"Warning: {os.path.basename(filepath)} - Not enough positive peaks.")
        return None, None, None, None, None
    
    ln_ratio = np.log(M0 / peak_voltages_fit)
    
    # Perform linear curve fitting: ln(M0/Mz) = (1/T2) * t
    try:
        # Initial guess for slope
        slope_guess = (ln_ratio[-1] - ln_ratio[0]) / (peak_times_fit[-1] - peak_times_fit[0])
        intercept_guess = ln_ratio[0]
        
        # Add measurement error (in log space, this is approximate)
        measurement_error = 0.05  # approximate error in ln scale
        sigma = np.full_like(ln_ratio, measurement_error)
        
        popt, pcov = curve_fit(linear_model, peak_times_fit, ln_ratio, 
                               p0=[slope_guess, intercept_guess], maxfev=10000, sigma=sigma, absolute_sigma=True)
        
        slope, intercept = popt
        slope_err = np.sqrt(np.diag(pcov))[0]
        
        # T2 = 1 / slope (convert from 1/ms to ms)
        T2 = 1.0 / slope if slope != 0 else None
        # Error propagation: T2_err = T2_err/slope_err * |T2|
        T2_err = (slope_err / (slope ** 2)) if slope != 0 else None
        
        # Calculate residuals and reduced chi-square
        residuals = ln_ratio - linear_model(peak_times_fit, slope, intercept)
        chi2 = np.sum((residuals / sigma) ** 2)
        reduced_chi2 = chi2 / (len(peak_times_fit) - 2)
        
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        return None, None, None, None, None
    
    # Generate fitted curve
    t_smooth = np.linspace(peak_times_fit.min(), peak_times_fit.max(), 500)
    fitted_curve = linear_model(t_smooth, slope, intercept)
    
    # Generate error bands for slope uncertainty
    slope_upper = slope + slope_err
    slope_lower = slope - slope_err
    curve_upper = linear_model(t_smooth, slope_upper, intercept)
    curve_lower = linear_model(t_smooth, slope_lower, intercept)
    
    # Create and save the plot (without displaying)
    os.makedirs(figures_dir, exist_ok=True)
    fitted_dir = os.path.join(figures_dir, 'T2_lin_fitted')
    os.makedirs(fitted_dir, exist_ok=True)
    filename = os.path.basename(filepath)
    output_name = os.path.splitext(filename)[0] + '_linearized.png'
    output_path = os.path.join(fitted_dir, output_name)
    
    plt.figure(figsize=(12, 7))
    plt.scatter(peak_times_fit, ln_ratio, color='red', s=50, label='ln(M₀/Mz) data points', zorder=3)
    # Shade the error band around the fitted curve
    plt.fill_between(t_smooth, curve_lower, curve_upper, color='blue', alpha=0.15, label=f'Slope uncertainty (±1σ: ±{slope_err:.6f})')
    plt.plot(t_smooth, fitted_curve, 'b-', linewidth=2, label=f'Linear fit: $y = \\frac{{1}}{{T_2}} \\cdot t$')
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('ln(M₀/Mz)', fontsize=12)
    
    # Title with T2 value
    title_str = f'T2 Relaxation (Linearized): {filename}\n'
    if T2 is not None and T2_err is not None:
        title_str += f'$T_2 = {T2:.4f} \\pm {T2_err:.4f}$ ms, Slope = $1/T_2 = {slope:.6f}$ ms⁻¹'
    else:
        title_str += f'Slope = {slope:.6f} ms⁻¹'
    
    plt.title(title_str, fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    # plt.savefig(output_path, dpi=300)
    plt.close()
    
    peaks = {
        'times': peak_times_fit,
        'ln_ratio': ln_ratio,
        'M0': M0,
        'count': len(peak_times_fit)
    }
    
    return T2, T2_err, slope, slope_err, peaks

def fit_t2_linearized_multiple_files(data_folder='data/T2', prominence=0.5, distance=50, height_threshold=None, figures_dir='figures'):
    """
    Fit T2 relaxation for all CSV files using linearization method.
    
    Args:
        data_folder: Folder containing CSV files
        prominence: Peak prominence threshold
        distance: Minimum distance between peaks
        height_threshold: Minimum peak height (default: 8% of max voltage)
        figures_dir: Directory to save figures
    """
    csv_files = glob.glob(os.path.join(data_folder, '**/*.csv'), recursive=True)
    
    if not csv_files:
        print(f"No CSV files found in {data_folder}")
        return
    
    results = []
    for filepath in csv_files:
        print(f"\nProcessing: {filepath}")
        T2, T2_err, slope, slope_err, peaks = fit_t2_linearized(filepath, prominence=prominence, distance=distance, height_threshold=height_threshold, figures_dir=figures_dir)
        
        if T2 is not None and T2_err is not None:
            filename = os.path.basename(filepath)
            results.append({
                'filename': filename,
                'filepath': filepath,
                'T2': T2,
                'T2_err': T2_err,
                'slope': slope,
                'slope_err': slope_err,
                'M0': peaks['M0'],
                'peak_count': peaks['count']
            })
    
    # Summary table
    if results:
        print(f"\n{'='*110}")
        print("SUMMARY OF T2 FITS (LINEARIZED METHOD)")
        print(f"{'='*110}")
        print(f"{'Filename':<40} {'T2 (ms)':<30} {'Slope (ms⁻¹)':<25}")
        print(f"{'-'*110}")
        for result in results:
            T2_str = f"{result['T2']:.4f} ± {result['T2_err']:.4f}"
            slope_str = f"{result['slope']:.6f} ± {result['slope_err']:.6f}"
            print(f"{result['filename']:<40} {T2_str:<30} {slope_str:<25}")
        print(f"{'='*110}\n")

# Main execution
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("T2 Relaxation Fitting Tool (Linearized Method)")
        print("=" * 70)
        print("Fits ln(M0/Mz) vs t to extract T2 = 1/slope")
        print("=" * 70)
        print("Usage:")
        print("  python fit_t2_linearized.py all [folder] [subfolder]      - Fit all files")
        print("  python fit_t2_linearized.py <filepath> [subfolder]        - Fit single file")
        print("  python fit_t2_linearized.py <filepath> [prom] [dist]      - Fit with custom params")
        print("=" * 70)
        print("\nParameters:")
        print("  folder:     Data folder path (default: data/T2)")
        print("  subfolder:  Subfolder inside 'figures' to save plots (default: T2_lin_fitted)")
        print("  prom:       Peak prominence (default: 0.1)")
        print("  dist:       Peak distance in samples (default: 50)")
        print("\nExample:")
        print("  python fit_t2_linearized.py all")
        print("  python fit_t2_linearized.py all data/T2")
        print("  python fit_t2_linearized.py all data/T2 lin_fits")
        print("  python fit_t2_linearized.py data/T2/file.csv")
        print("  python fit_t2_linearized.py data/T2/file.csv my_lin_fits")
        print("  python fit_t2_linearized.py data/T2/file.csv 0.2 100")
        sys.exit(0)
    
    arg = sys.argv[1].strip().strip("'\"")
    prominence = 0.1
    distance = 50
    height_threshold = None
    figures_subfolder = 'figures'  # Default subfolder
    
    if arg.lower() == 'all':
        # Handle 'all' command with optional folder and subfolder
        data_folder = 'data/T2'
        if len(sys.argv) > 2:
            # Check if next argument is a folder path or a subfolder name
            arg2 = sys.argv[2].strip().strip("'\"")
            if os.path.isdir(arg2) or '/' in arg2 or arg2.startswith('data'):
                data_folder = arg2
                # Check if subfolder is specified after folder
                if len(sys.argv) > 3:
                    subfolder_name = sys.argv[3].strip().strip("'\"")
                    figures_subfolder = os.path.join('figures', subfolder_name)
            else:
                # Treat as subfolder name
                subfolder_name = arg2
                figures_subfolder = os.path.join('figures', subfolder_name)
        
        fit_t2_linearized_multiple_files(data_folder, prominence=prominence, distance=distance, height_threshold=height_threshold, figures_dir=figures_subfolder)
    else:
        # Handle single file with optional subfolder and parameters
        if os.path.exists(arg):
            # Check if next argument is a subfolder or prominence parameter
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
                except ValueError:
                    # Not a number, treat as subfolder name
                    subfolder_name = arg2
                    figures_subfolder = os.path.join('figures', subfolder_name)
                    # Check for prominence after subfolder
                    if len(sys.argv) > 3:
                        try:
                            prominence = float(sys.argv[3])
                        except ValueError:
                            pass
                    if len(sys.argv) > 4:
                        try:
                            distance = int(sys.argv[4])
                        except ValueError:
                            pass
            
            fit_t2_linearized(arg, prominence=prominence, distance=distance, height_threshold=height_threshold, figures_dir=figures_subfolder)
        else:
            print(f"Error: File '{arg}' not found.")
