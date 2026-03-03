import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def plot_t2_vs_parameter(t2_values, t2_errors, parameter_values, parameter_name='temperature', output_path='figures', solution_name=None):
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
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(parameter_values_sorted, t2_values_sorted, yerr=t2_errors_sorted, 
                fmt='o-', capsize=5, markersize=8, linewidth=2)
    
    if parameter_name.lower() == 'temperature':
        ax.set_xlabel('Temperature (K)', fontsize=12)
        plot_name = 'T2_vs_temperature.png'
    else:
        ax.set_xlabel('Concentration (M)', fontsize=12)
        plot_name = 'T2_vs_concentration.png'

    if solution_name:
        solution_tag = str(solution_name).replace(' ', '_')
        plot_name = plot_name.replace('.png', f'_{solution_tag}.png')
    
    ax.set_ylabel('T2 (ms)', fontsize=12)
    if solution_name:
        ax.set_title(f'T2 Relaxation Time vs {parameter_name.capitalize()} ({solution_name})', fontsize=13)
    else:
        ax.set_title(f'T2 Relaxation Time vs {parameter_name.capitalize()}', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, plot_name)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()
    plt.close()
    
    print(f"\n✓ Plot saved: {output_file}")
    print(f"\nPlot details:")
    print(f"  Parameter: {parameter_name}")
    print(f"  Number of points: {len(parameter_values)}")
    print(f"  {parameter_name.capitalize()} range: {parameter_values_sorted.min():.3f} - {parameter_values_sorted.max():.3f}")
    print(f"  T2 range: {t2_values_sorted.min():.6f} - {t2_values_sorted.max():.6f} ms")

# Main execution
if __name__ == '__main__':

    # T2 vs conc Gly roomT 
    T2 = [1615, 1389, 1212, 1067, 288, 97, 56, 20  ]
    T2_error = [383, 199, 204, 84, 11, 5, 4, 2]
    T = [1.5, 3.05, 6.1, 12.5, 50, 75, 90, 100]  # Concentration in M
    parameter_name = 'concentration'
    solution_name = 'Gly_T=24'


    # # T2 vs conc CuSO4 roomT 
    # T2 = [26.4621, 15.7133, 10.7210, 4.6863, 2.3691, 1.0891, 0.7125, 0.6010,0.5876]
    # T2_error = [1.555, 0.8991, 1.3321, 0.2879, 0.1004, 0.028, 0.0245, 0.00156, 0.0224]
    # T = [1.5, 3.1, 6.25, 12.5, 25, 50, 75, 90, 100]  # Concentration in M
    # parameter_name = 'concentration'
    # solution_name = 'CuSO4_T=24'


    # # # T2 vs T CuSO4_1.5  
    # T2 = [22.7331, 26.4632, 30.086232, 32.1376, 30.0709, 31.8151, 40.7262]
    # T2_error = [0.9566, 1.5551 , 1.7953, 0.5868, 1.1895, 0.4863, 0.9242]
    # T = [0 , 24, 30, 35, 40, 45, 50]  # Concentration in M
    # parameter_name = 'temperature'
    # solution_name = 'CuSO4_1.5'
   

    # T2 vs T gly roomT 
    # T2 = [7.2590, 20.2995, 24.0629, 35.8778, 43.0505, 52.9749, 59.1369]
    # T2_error = [0.2667, 2.3591, 0.9867, 0.4488, 1.0277, 0.7477, 1.1146]
    # T = [0 , 24, 30, 35, 40, 45, 50]  # Concentration in M
    # parameter_name = 'temperature'
    # solution_name = 'Glycerol_100'

    # T2 vs T CuSO4_50 roomT 
    # T2 = [0.7708, 1.0891, 1.2051, 1.2133, 1.3773,  1.3273, 1.5067 ]

    # T2_error = [0.0236, 0.0280, 0.0355, 0.0311, 0.0456, 0.0352, 0.0466  ]

    # T = [0 , 24, 30, 35, 40, 45, 50]  # Concentration in M
    # parameter_name = 'temperature'
    # solution_name = 'CuSO4_50'



    output_dir = 'figures/t2_results'

    
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
    
