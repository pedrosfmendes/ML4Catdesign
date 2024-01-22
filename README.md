# ML4Catdesign
#This data folder relates to the publication 
#‘Machine learning for catalyst design: data matters’ 
#by Pedro S.F. Mendes*, Florence H. Vermeire*, Thibaut Van Haute, and Joris W. Thybaut. 
#*joint first authors

#All data for the figures can be generated using the python scripts in the ‘scripts’ folder. If you run the scripts to reproduce the results, please make sure the appropriate environment is set including scikit-learn v1.0.2. The data and model results generated for this publication can be found in the ‘results’ folder. Data and results are stored as (1) dictionaries with features as keys and pandas DataFrames as values or (2) as pandas DataFrames in pickle files, they can be opened using python. Below, an overview is given of which files correspond to which figures in the publication.

#Figure 1:
#Script: scripts/size.py
#Data: results/size/SyntheticDataset_3M.pickle
#Data test sets: results/size/SyntheticDataset_test_50k.pickle
#Results: results/size/results_size.pickle

#Figure 2:
#Script: scripts/size_ocm.py
#Data: results/size_ocm/ocm_all.pickle
#Data test sets: results/size_ocm/ocm_test_1k.pickle
#Results: results/size_ocm/results_ocm_size.pickle

#Figure 3:
#Script: scripts/features_ocm.py
#Data: results/features_ocm/ocm_all.pickle
#Results: results/features_ocm/results_ocm_features.pickle

#Figure 4:
#Script: scripts/design.py
#Data: results/design/DesignSpace_100k.pickle
#Results: results/design/results_ocm_features.pickle

#Figure 5:
#Script: scripts/error.py
#Data: results/error/Error_100k.pickle
#Results: results/error/results_error.pickle
#(Data also used for Figure S3)

#Figure 6:
#Script: scripts/corr_linear.py
#Data: results/corr_linear/CorrLinear_100k.pickle
#Results: results/corr_linear/results_corr_linear.pickle

#Table 1:
#Script: scripts/corr_arrhenius_base.py
#Data: results/corr_arrhenius/CorrArrhenius_100k_base.pickle
#Results: results/corr_arrhenius/results_corr_arrhenius_base.pickle
#Script: scripts/corr_arrhenius.py
#Data: results/corr_arrhenius/CorrArrhenius_100k.pickle
#Results: results/corr_arrhenius/results_corr_arrhenius.pickle
