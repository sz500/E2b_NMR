import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# def plot_t2_vs_parameter(t2_values, t2_errors, parameter_values, parameter_name='temperature', output_path='figures', solution_name=None):


#     # Sort by parameter values
#     sorted_indices = np.argsort(parameter_values)
#     parameter_values_sorted = parameter_values[sorted_indices]
#     t2_values_sorted = t2_values[sorted_indices]
#     t2_errors_sorted = t2_errors[sorted_indices]

#     # Create plot
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.errorbar(parameter_values_sorted, t2_values_sorted, yerr=t2_errors_sorted,
#                 fmt='x', color = 'black', ecolor = 'black', capsize=5, markersize=7, markeredgewidth = 1, linewidth=1.5)
#     ax.plot(parameter_values_sorted, t2_values_sorted, color='cornflowerblue', linewidth=1.5)

#     if parameter_name.lower() == 'temperature':
#         ax.set_xlabel('Temperature (°C)', fontsize=12)
#         plot_name = 'T2_vs_temperature.png'
#     else:
#         ax.set_xlabel('Concentration (%)', fontsize=12)
#         plot_name = 'T2_vs_concentration.png'

#     if solution_name:
#         solution_tag = str(solution_name).replace(' ', '_')
#         plot_name = plot_name.replace('.png', f'_{solution_tag}.png')

#     ax.set_ylabel('T2 (ms)', fontsize=12)
#     if solution_name:
#         ax.set_title(f'T2 vs. {parameter_name.capitalize()} ({solution_name})', fontsize=13)
#     else:
#         ax.set_title(f'T2 vs. {parameter_name.capitalize()}', fontsize=13)
#     ax.grid(True, alpha=0.3)

#     # Save plot
#     os.makedirs(output_path, exist_ok=True)
#     output_file = os.path.join(output_path, plot_name)
#     plt.tight_layout()
#     plt.savefig(output_file, dpi=300)
#     plt.show()
#     plt.close()

#     print(f"\n✓ Plot saved: {output_file}")
#     print(f"\nPlot details:")

def plot_t2_vs_parameter(t2_values, t2_errors, parameter_values, parameter_name='temperature', output_path='figures', solution_name=None):
    # ...existing code...
    """
    Plot T2 values against manually provided concentration or temperature values.

    Args:
        t2_values: List or array of T2 values (ms)
        t2_errors: List or array of T2 error values (ms)
        parameter_values: List or array of concentration or temperature values
        parameter_name: 'temperature' or 'concentration'
        output_path: Directory to save the plot
        solution_name: Optional solution label (e.g., 'CuSO4', 'Glycerol')
    """
    # Convert to numpy arrays
    t2_values = np.array(t2_values)
    t2_errors = np.array(t2_errors)
    parameter_values = np.array(parameter_values)

    # Validate inputs
    if len(t2_values) != len(parameter_values):
        print(f"Error: Length of t2_values ({len(t2_values)}) must match length of parameter_values ({len(parameter_values)})")
        return

    if len(t2_errors) != len(t2_values):
        print(f"Error: Length of t2_errors ({len(t2_errors)}) must match length of t2_values ({len(t2_values)})")
        return
    # Sort by parameter values
    sorted_indices = np.argsort(parameter_values)
    parameter_values_sorted = parameter_values[sorted_indices]
    t2_values_sorted = t2_values[sorted_indices]
    t2_errors_sorted = t2_errors[sorted_indices]

    # Publication-style plotting defaults (local to this figure)
    plt.rcParams.update({
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
    })

    # Single-column journal-friendly size
    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    ax.errorbar(
        parameter_values_sorted,
        t2_values_sorted,
        yerr=t2_errors_sorted,
        fmt='o',                      # marker only, no black connecting line
        color='black',
        ecolor='black',
        elinewidth=0.8,
        capsize=2,
        capthick=0.8,
        markersize=3,
        markeredgewidth=1,
        zorder=2
    )
    ax.plot(
        parameter_values_sorted,
        t2_values_sorted,
        linestyle='-',
        color='steelblue',
        linewidth=1.5,
        zorder=1
    )

    ######################### linear fit ############################
    slope, intercept = np.polyfit(parameter_values_sorted, t2_values_sorted, 1)
    y_fit = slope * np.array(parameter_values_sorted) + intercept
    # 2. Calculate R-squared
    residuals = t2_values_sorted - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((t2_values_sorted - np.mean(t2_values_sorted))**2)
    r_squared = 1 - (ss_res / ss_tot)
    x_min = np.min(parameter_values_sorted)
    x_max = np.max(parameter_values_sorted)
    margin = (x_max - x_min) * 0.10
    x_fit_extended = np.linspace(x_min - margin, x_max + margin, 100)
    y_fit_extended = slope * x_fit_extended + intercept
    ax.plot(
        x_fit_extended,
        y_fit_extended,
        linestyle='-.',
        color='gray',
        linewidth=1.5,
        label=f'Linear Fit ($R^2 = {r_squared:.3f}$)',
        zorder=1
    )
    # r^2
    ax.text(0.05, 0.95, f'$R^2 = {r_squared:.4f}$',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    ###################################################################

    if parameter_name.lower() == 'temperature':
        ax.set_xlabel('1 / Temperature (K$^{-1}$)')
        # ax.set_xlabel('Temperature (°C)')
        plot_name = 'T2_vs_temperature'
    else:
        ax.set_xlabel('Concentration (%)')
        plot_name = 'T2_vs_concentration'

    if solution_name:
        solution_tag = str(solution_name).replace(' ', '_')
        plot_name = f'{plot_name}_{solution_tag}'

    ax.set_ylabel('1/$T_2$ (ms$^{-1}$)')
    # ax.set_ylabel('ln($T_2$/T) (ln(ms K$^{-1}$))')
    # ax.set_ylabel('ln($T_2$) (ln(ms))')
    # For publication, usually prefer caption over plot title:
    # if solution_name:
    #     ax.set_title(f'T2 vs. {parameter_name.capitalize()} ({solution_name})')
    # else:
    #     ax.set_title(f'T2 vs. {parameter_name.capitalize()}')

    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.25)

    # Save plot
    os.makedirs(output_path, exist_ok=True)
    output_png = os.path.join(output_path, f'{plot_name}.png')
    output_pdf = os.path.join(output_path, f'{plot_name}.pdf')  # vector format for papers
    fig.tight_layout()
    fig.savefig(output_png, dpi=600)
    fig.savefig(output_pdf)
    plt.show()
    plt.close(fig)

    print(f"\n✓ Plot saved: {output_png}")
    print(f"✓ Plot saved: {output_pdf}")
    print(f"  Parameter: {parameter_name}")
    print(f"  Number of points: {len(parameter_values)}")
    print(f"  {parameter_name.capitalize()} range: {parameter_values_sorted.min():.3f} - {parameter_values_sorted.max():.3f}")
    print(f"  T2 range: {t2_values_sorted.min():.6f} - {t2_values_sorted.max():.6f} ms")

# Main execution
if __name__ == '__main__':
    # T2 vs conc Gly roomT
    # T = np.array([1.5, 3.05, 6.1, 12.5, 25, 50, 75, 90, 100])
    # T2 = np.array([360 , 419, 533.3010, 434.9676, 347.5364, 261.7863, 97.2445, 55.9497, 19.3596  ])
    # T2_error = np.array([86, 68, 162.2010, 41.1700, 7.4054, 6.2695, 2.4361, 2.4691, 1.0929]) / T2
    # T2 = np.log(T2)
    # parameter_name = 'concentration'
    # solution_name = 'Gly T=24°C'

    # # T2 vs conc CuSO4 roomT
    T2 = np.array([26.4621, 15.7133, 10.7210, 4.6863, 2.3691, 1.0891, 0.7125, 0.6010,0.5876])
    T2_error = np.array([1.555, 0.8991, 1.3321, 0.2879, 0.1004, 0.028, 0.0245, 0.00156, 0.0224])
    T = np.array([1.5, 3.1, 6.25, 12.5, 25, 50, 75, 90, 100])
    T2_error /= T2**2
    T2 = 1/T2
    parameter_name = 'concentration'
    solution_name = 'CuSO4 T=24°C'


    # # T2 vs T CuSO4_1.5
    # T2 = np.array([22.7331, 26.4632, 30.086232, 30.6613, 35.4139, 37.0505, 37.6528])
    # T = np.array([0 , 24, 30, 35, 40, 45, 50]) + 273
    # T2_error = np.array([0.9566, 1.5551 , 1.7953, 0.7226, 0.8302, 0.4913, 0.4885]) / T2
    # y = np.log(T2/T)
    # x = 1/T
    # T = x
    # T2 = y
    # parameter_name = 'temperature'
    # solution_name = 'CuSO4 1.5%'


    # T2 vs T gly roomT
    # T2 = np.array([7.2590, 19.3596, 24.0629, 31.7927, 38.0412, 57.3390 , 49.9855])
    # T = np.array([0 , 24, 30, 35, 40, 45, 50])+273
    # T2_error = np.array([0.2667, 1.0929, 0.9867, 0.4488, 0.5918, 1.3372, 0.7331]) / T2
    # y = np.log(T2/T)
    # x = 1/T
    # T = x
    # T2 = y
    # parameter_name = 'temperature'
    # solution_name = 'Glycerol 100%'

    # T2 vs T CuSO4_50 roomT
    # T2 = np.array([0.7708, 1.0891, 1.2051, 1.2133, 1.3773,  1.3273, 1.5067 ])
    # T = np.array([0 , 24, 30, 35, 40, 45, 50])+273
    # T2_error = np.array([0.0236, 0.0280, 0.0355, 0.0311, 0.0456, 0.0352, 0.0466  ]) / T2
    # y = np.log(T2/T)
    # x = 1/T
    # T = x
    # T2 = y
    # parameter_name = 'temperature'
    # solution_name = 'CuSO4 50%'



    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, 'figures', 't2_results')


    # ====================================
    # Create the plot
    # ====================================
    plot_t2_vs_parameter(T2, T2_error, T, parameter_name, output_dir, solution_name)


    if len(sys.argv) < 2:
        print("Manual T2 Plotting Tool")
        print("=" * 70)
        print("This script plots T2 values against manually provided parameter lists.")
        print("\nUsage:")
        print("  Edit this script directly to input your data in the 'Example Usage' section")
        print("\nExample:")
        print("  # Define your data")
        print("  T2 = [120.5, 95.3, 78.2, 65.4, 55.8]")
        print("  T2_error = [1.2, 0.95, 0.88, 0.76, 0.65]")
        print("  temperatures = [24, 35, 45, 55, 65]")
        print("")
        print("  # Then call:")
        print("  plot_t2_vs_parameter(T2, T2_error, temperatures, 'temperature', 'figures')")
        print("\nOR use it as a module in your own script:")
        print("  from plot_T2_vs_T_C import plot_t2_vs_parameter")
        print("  plot_t2_vs_parameter([120.5, 95.3], [1.2, 0.95], [24, 35], 'temperature')")
        sys.exit(0)

    # ====================================
    # EDIT YOUR DATA HERE
    # ====================================

    # Example 1: T2 vs Temperature for Glycerol
    # T2 = [120.5, 95.3, 78.2, 65.4, 55.8]
    # T2_error = [1.2, 0.95, 0.88, 0.76, 0.65]
    # T = [24, 35, 45, 55, 65]  # Temperature in K
    # parameter_name = 'temperature'
    # output_dir = 'figures'

