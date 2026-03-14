import pandas as pd
import matplotlib.pyplot as plt

file1 = 'strange_T2_trial.csv'

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
fig, ax1 = plt.gcf(), plt.gca()

try:
    df1 = pd.read_csv(file1, skiprows=[1])
    ax1.plot(df1['Time'], df1['Channel A'], label='Receiver Output', alpha=0.8, color='b')
    ax1.plot(df1['Time'], df1['Channel B']/2, label='Mixer Output', alpha=0.2,color='r')
    ax1.set_title('NMR Signal Trace: Delay = 1.4ms')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_xlabel('time (ms)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)
except FileNotFoundError:
    print(f"Error: {file1} not found in the current folder.")

# try:
#     df2 = pd.read_csv(file2, skiprows=[1])
#     ax2.plot(df2['Time'], df2['Channel A'], label='Channel A', alpha=0.8)
#     # ax2.plot(df2['Time'], df2['Channel B'], label='Channel B', alpha=0.8)
#     ax2.set_title('NMR Signal Trace: Delay = 1.6ms')
#     ax2.set_xlabel('Time (ms)')
#     ax2.set_ylabel('Voltage (V)')
#     ax2.legend(loc='upper right')
#     ax2.grid(True, linestyle='--', alpha=0.7)
# except FileNotFoundError:
#     print(f"Error: {file2} not found in the current folder.")

plt.tight_layout()
plt.savefig('strange_peak_T2.png', dpi=300)
plt.show()