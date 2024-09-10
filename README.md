# Neuron Segmentation and Fluorescence Analysis for Protein Expression Analysis
This repository contains code for analyzing neurite and soma regions in bioimaging data, focusing on segmenting neuronal structures and measuring fluorescence intensity. The code is implemented using various image processing techniques such as watershed segmentation, GrabCut, and CLAHE enhancement, as well as region-based fluorescence quantification.

## Image Processing 
#### Watershed Segmentation:
- Used to separate neuronal regions (somas and neurites) from background and nearby structures.
#### GrabCut Refinement: 
- Refines the segmentation output to improve accuracy in separating neurite and soma regions.
#### CLAHE Enhancement: 
- Enhances image contrast using Contrast Limited Adaptive Histogram Equalization.
#### Fluorescence Intensity Calculation: 
- Measures the mean fluorescence intensity in soma, neurite, and background regions.
#### Morphological Operations: 
- Identifies and refines neurite skeletons, removing small branches for a clearer representation of the neurite structure.
### Key Functions
#### process_fluorescence_channel(image_channel, cell_mask, background_mask): 
- Calculates the fluorescence intensity of the cell regions and background, normalizing the result for accurate comparison.
#### apply_watershed(image): 
- Segments neurite and soma regions using the watershed algorithm based on distance transforms.
#### apply_grabcut(image, binary_mask): 
- Refines the segmentation using the GrabCut algorithm, improving separation between foreground (neurites/somas) and background.
#### apply_clahe(image): 
- Enhances the image contrast using CLAHE to improve visibility of the cell structures.
#### apply_clahe_selectively(original, enhanced, mask): 
- Combines original and enhanced images based on a binary mask, preserving key regions while enhancing contrast.
### Workflow
#### 1. Preprocessing: 
- Images are loaded and converted to grayscale if necessary. Noise is reduced using denoising techniques.
#### 2. Segmentation:
- Watershed segmentation is applied to identify somas and neurites.
- The output is refined using GrabCut for more precise segmentation.
#### 3. Skeletonization: 
- Neurite structures are skeletonized for further morphological analysis, ensuring small branches are removed to highlight major structures.
#### 4. Fluorescence Analysis: 
- The fluorescence intensity of somas, neurites, and background regions is calculated, and the results are normalized to provide comparative measures.
#### 5. Results Visualization: 
- The segmentation and analysis results are visualized using overlays of somas and neurites on the original images.


## Data Analysis: Fluorescence Intensity Across Substrate Conditions

This script is designed to analyze the relationship between fluorescence intensity and different neurite substrate conditions (Thin, Mushroom, Stubby) based on the data stored in a CSV file (`soma_neurite_separate_data.csv`) from the image processing code. It focuses on the mean fluorescence intensity across multiple channels and substrate conditions.

   
### 1. **Categorizing Substrate and Pitch Groups**:
   - **`P Group`**: The pitch values are categorized into groups such as `p4`, `p10`, `p30`, and `flat`.
   - **`Image Group`**: The substrate conditions are categorized into `Thin`, `Mushroom`, and `Stubby` based on the first letter of the image name.
   - The code also replicates the 'flat' data for all `Image Group` categories.

### 2. **Data Cleaning and Grouping**:
   - Rows with undefined image groups are removed.
   - The data is grouped by `Image Group`, `P Group`, and `Channel`, and the mean fluorescence intensity is calculated for each group.

### 3. **Heatmap Visualization**:
   - The script visualizes the data using heatmaps for three fluorescence channels (Paxillin, Integrin, and green channel).

### 4. **Output**:
   - The generated heatmaps show the mean fluorescence intensity for Paxillin (channel 0), Integrin (channel 1), and another channel across different substrate conditions (`Thin`, `Mushroom`, `Stubby`) and pitch values (`p4`, `p10`, `p30`, `flat`).
   - Heatmaps are saved as PDF files (`figure_0.pdf`, `figure_1.pdf`, `figure_2.pdf`).

