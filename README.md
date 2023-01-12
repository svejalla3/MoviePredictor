# CS7641-project
Website for class project
https://aditya-khadilkar.github.io/cs7641.github.io/

This repository also contains the script for the final project. The layout and contents are explained below:

### Table of Contents:
- **data:** All notebooks and scripts created to collect and analyze the IMDb movie dataset
- **data-preprocess:** All notebooks used to preprocess the IMDb movies dataset extracted from the API
    - **feature_dictionaries:** All feature dictionaries used to create one-hot representations of movies
- **PCA:** All required files and script to run the PCA algorithm for dimentionality reduction
- **models:** All scripts conraining model definitions and evaluation pipelines
- **metrics:** Locations were all final metrics are saved, across all models and experiments
- **index.html:** Website for this project

## TODO - Final Report
- [ ] Consolidate pipelines across PCA, GMM, Kmeans, and all futue model definitions
- [ ] Remove rare genres, pick same number of example per genre to address class imbalance issues
- [ ] Select & implement new unsupervised techniques to try
- [ ] Try leveraging GMM soft clustering capabilities for multi-label genre classification
- [ ] Consoidate evaluation pipelines
- [ ] Try multimodal approach? -- stretch goal
- [ ] Run experiements
- [ ] Cross validation (if necessary)
- [ ] Create final report

