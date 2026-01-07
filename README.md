# Master's Thesis Figures

This repository contains the resources, scripts, and notebooks used to generate the figures for my master's thesis. Below is an overview of the main folders and their contents.

## Repository Structure

### 3D_points/
Contains scripts and visualizations related to 3D point analysis. This includes diagrams for ResNet and VGG models, as well as various 3D plots and VRML files for visualization.

### antisym_bipolarity/
Resources for experiments on antisymmetric bipolarity. Includes Jupyter notebooks and PDFs for ResNet and VGG experiments, sorted by various parameters.

### antisym_bipolarity_with_gauge_constraint/
Contains resources for experiments involving antisymmetric bipolarity with gauge constraints.

### antisym_experiment/
General resources and scripts for antisymmetric experiments.

### dct_breakdown/
Resources for analyzing the breakdown of Discrete Cosine Transform (DCT).

### ERF/
Contains resources for Effective Receptive Field (ERF) analysis.

### expanded_weights/
Resources for visualizing expanded weights in the models.

### kernels/
Contains resources related to kernel generation and analysis.

### layer_wise_discrete_kernel_type_count/
Resources for analyzing kernel types on a layer-wise basis.

### motivation_diagrams/
Diagrams illustrating the motivation and key concepts behind the thesis.

### nmf/
Resources for Non-negative Matrix Factorization (NMF) experiments and analysis.

### orrientation_conv/
Contains resources for experiments related to orientation convolution.

### propagation/
Resources for propagation experiments, including subfolders for multidirectional, single-pixel, and unipolar propagation.

### random_init_beta/
Experiments involving random initialization of beta values.

### rgb_visualization/
Resources for RGB visualizations.

### utils/
Contains utility scripts and functions used across various experiments. Below are the primary functions utilized for the thesis:

- **get_filter(model, layer, sev=False)**: Extracts the filters from a specified convolutional layer in a given model.

- **getDominantAngle(filters)**: Computes the dominant angle of the filters based on their symmetry and antisymmetry properties.

- **getSobelTF(filters)**: Applies Sobel filters to compute gradients in the horizontal and vertical directions for the given filters.

- **getSymAntiSymTF(filter)**: Decomposes a filter into its symmetric and antisymmetric components.

- **topKfilters(model, layer_num, k=10, sev=False)**: Identifies the top K filters in a specified layer based on their magnitude.

- **topKchannels(model, layer_num, f_num, k=10, sev=False)**: Identifies the top K channels in a specified filter based on their magnitude.

- **dct2(a)**: Computes the 2D Discrete Cosine Transform (DCT) of an input array.

- **idct2(a)**: Computes the 2D Inverse Discrete Cosine Transform (IDCT) of an input array.

- **plot_filter_x(beta2, ax=None)**: Visualizes a 3D plot of a filter based on the given beta2 parameter.

### weight_orrientation_similarity/
Resources for analyzing weight orientation similarity.

## Usage

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the desired directory and open the relevant Jupyter notebooks or scripts.
3. Follow the instructions in the notebooks to reproduce the figures.

## Requirements

- Python 3.x
- Jupyter Notebook
- Required Python packages (install using `requirements.txt` if available).

## License

This repository is for academic purposes related to my master's thesis. Feel free to use the contents as needed.

## Contact

For any questions or clarifications, feel free to reach out to me.
