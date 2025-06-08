import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_linear_kernel(data):
    linear_data = data[data['kernel'] == 'linear']

    if linear_data.empty:
        print("No data available for kernel: linear")
        return

    # Average accuracy in case multiple gamma values exist for same C
    grouped = linear_data.groupby('C')['test_accuracy'].mean().reset_index()
    C_vals = grouped['C'].values
    acc_vals = grouped['test_accuracy'].values

    plt.figure(figsize=(8, 6))
    plt.plot(C_vals, acc_vals, marker='o', linestyle='-')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs C (Linear Kernel)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_kernel_surface(data, kernel_name):
    kernel_data = data[data['kernel'] == kernel_name]

    if kernel_data.empty:
        print(f"No data available for kernel: {kernel_name}")
        return

    C_raw = kernel_data['C'].values
    gamma_raw = kernel_data['gamma'].values
    test_accuracy = kernel_data['test_accuracy'].values

    log_C = np.log10(C_raw)
    log_gamma = np.log10(gamma_raw)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(log_C, log_gamma, test_accuracy, cmap='viridis', linewidth=0.2)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Set ticks and labels to actual values
    unique_C = np.sort(np.unique(C_raw))
    unique_gamma = np.sort(np.unique(gamma_raw))

    ax.set_xticks(np.log10(unique_C))
    ax.set_xticklabels([str(c) for c in unique_C])

    ax.set_yticks(np.log10(unique_gamma))
    ax.set_yticklabels([str(g) for g in unique_gamma])

    ax.set_xlabel('C')
    ax.set_ylabel('gamma')
    ax.set_zlabel('Accuracy')
    ax.set_title(f'3D Accuracy Surface - Kernel: {kernel_name}')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_file.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        # Skip the first row (CPU info)
        data = pd.read_csv(file_path, skiprows=1)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    expected_columns = ['C', 'gamma', 'kernel', 'test_accuracy']
    if not all(col in data.columns for col in expected_columns):
        print(f"Error: Data file '{file_path}' does not contain expected columns.")
        sys.exit(1)

    available_kernels = data['kernel'].unique()
    print(f"Kernels found in file: {', '.join(available_kernels)}")

    for kernel in available_kernels:
        if kernel == 'linear':
            plot_linear_kernel(data)
        else:
            plot_kernel_surface(data, kernel)
