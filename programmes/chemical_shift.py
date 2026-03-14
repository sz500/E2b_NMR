import numpy as np

sample_a_khz = np.array([0.28, 0.45, 1.22, 1.38])

sample_b_khz = np.array([0.28, 0.38, 0.49, 1.23, 1.45])

delta_ref = -113
ref_khz = 0.091

spec_mhz = 21.67  # spectrometer frequency

chemical_shifts_a = (sample_a_khz - ref_khz) / spec_mhz * 1000 + delta_ref
chemical_shifts_b = (sample_b_khz - ref_khz) / spec_mhz * 1000 + delta_ref

print("Sample A shifts (ppm):", chemical_shifts_a)
print("Sample B shifts (ppm):", chemical_shifts_b)