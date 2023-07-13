from Simulasi import validasi
beta = 0.0027
density = 25
time_start = 8
time_end = 8.05
runs = 1
validasi(beta, density, time_start, time_end, runs)

# # Menentukan area-area di MFD (L1, L2, L3, L4)
# import numpy as np
# import matplotlib.pyplot as plt

# # Buat fungsi untuk L1, L2, L3, L4 dari MFD
# def determine_areas(flow_rates, densities):
#     # Fit a curve to the scatter plot data (you can use curve fitting techniques)
    
#     mfd_curve = fit_mfd_curve(flow_rates, densities)

#     # Determine the boundaries between the areas
#     L1_boundary = define_L1_boundary(mfd_curve)
#     L2_boundary = define_L2_boundary(mfd_curve)
#     L3_boundary = define_L3_boundary(mfd_curve)
#     L4_boundary = define_L4_boundary(mfd_curve)
    

#     # Classify data points based on the determined boundaries
#     areas = classify_data_points(flow_rates, densities, L1_boundary, L2_boundary, L3_boundary, L4_boundary)

#     return areas

# # Function to fit a curve to the scatter plot data (you can replace this with your own curve fitting method)
# def fit_mfd_curve(flow_rates, densities):
#     # Perform curve fitting and return the fitted curve
#     # Replace this with your curve fitting code
#     return fit_mfd_curve

# # Function to define the boundary for L1 area
# def define_L1_boundary(mfd_curve):
#     # Determine the boundary for L1 based on the characteristics of the curve
#     # Replace this with your own logic to define the L1 boundary
#     L1_boundary = ...

#     return L1_boundary

# # Function to define the boundary for L2 area
# def define_L2_boundary(mfd_curve):
#     # Determine the boundary for L2 based on the characteristics of the curve
#     # Replace this with your own logic to define the L2 boundary
#     L2_boundary = ...

#     return L2_boundary

# # Function to define the boundary for L3 area
# def define_L3_boundary(mfd_curve):
#     # Determine the boundary for L3 based on the characteristics of the curve
#     # Replace this with your own logic to define the L3 boundary
#     L3_boundary = ...

#     return L3_boundary

# # Function to define the boundary for L4 area
# def define_L4_boundary(mfd_curve):
#     # Determine the boundary for L4 based on the characteristics of the curve
#     # Replace this with your own logic to define the L4 boundary
#     L4_boundary = ...

#     return L4_boundary

# # Function to classify data points based on the determined boundaries
# def classify_data_points(flow_rates, densities, L1_boundary, L2_boundary, L3_boundary, L4_boundary):
#     areas = []

#     for flow_rate, density in zip(flow_rates, densities):
#         # Classify the data point into L1, L2, L3, or L4 based on the determined boundaries
#         if density < L1_boundary:
#             areas.append("L1")
#         elif density < L2_boundary:
#             areas.append("L2")
#         elif density < L3_boundary:
#             areas.append("L3")
#         elif density < L4_boundary:
#             areas.append("L4")
#         else:
#             areas.append("L5")
#     return areas

# # Example usage
# # Assuming you have flow rates and densities as numpy arrays
# flow_rates = np.array([...])
# densities = np.array([...])

# areas = determine_areas(flow_rates, densities)

# # Print the areas
# for area in areas:
#     print(area)

# # Plotting the MFD curve and the classified areas
# plt.scatter(densities, flow_rates, label="Data Points")
# plt.plot(densities, mfd_curve, color="red", label="MFD Curve")
# plt.legend()
# plt.xlabel("Density")
# plt.ylabel("Flow Rate")
# plt.show()
