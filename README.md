# COMET-pytorch
COMET(Chemically Omnipotent Molecular Encoder from Transformer)

# Branch Version  
- [X] branch_v1_weighted_sampling_dataloader
    - [X] Data Loader : data loader sampling masking atom based on their inverse occurence probability  
    - [X] Logging : record macro f1 score, confusion matrix, weight histogram  
    
- [X] branch_v2_weighted_sampling_dataset_fixed_size_batch     
    - [X] Data : dataset consist more balanced molecule sample with more abundant rare symbol.  
    - [X] Data Set : Normalize molecular property with mean & std value. It is very fast with pandas operation.    
    - [X] Data Set : It preprocess each molecule and hold their Adjacency Matrix and Feature Matrix. Also each molecule is parsed into fixed size vector.    
    - [X] Inside Iteration : masking indices were selected based on the symbol distribution and return A, X, masked_A, masked_X, masked_idx, P  
    - [X] Ground Truth : previous ground truth matrix is indexed inside the training iteration.  
    - [ ] ~~Loss : weighted cross-entropy loss applied.~~  
    - [ ] Put Masking Task in Batch_process fn
    - [ ] Normalize A matrix in dataloader or iteration (Note. masked A should be generate from the Normalized A)
    - [ ] Crop A matrix with max-atom length in order to increase speed
     
- [ ] branch_v3_radius_masking_dataloader  
    - [ ] Data Loader : Firstly sampling center atom with occurence distribution. Secondly, it find out adjacent atom by multiplying A matrix with r(radius) times. Construct index set and truncate with num_masking  
    
- [ ] branch_v4_adj_masking_dataloader  
    - [ ] Data Loader : Masked A would provide  
    - [ ] Model : A should be calculate from previous A.   
    
# Dataset  
Total Number of Molecules in Raw Zinc Dataset : 531,354,040

|    Name   |        Train Size | Train Coverage |      Valid Size | Valid Coverage | Sampling Rate |
|:---------:|------------------:|---------------:|----------------:|---------------:|--------------:|
|  COMET_L  | 19.9M(19,919,005) |                | 4.9M(4,980,881) |                |          4.7% |
|  COMET_M  |  5.9M (5,975,109) |                | 1.4M(1,494,480) |                |          1.4% |
|  COMET_S  |  1.9M (1,979,256) |                | 0.5M  (495,380) |                |         0.47% |
| COMET_XXS |  197K   (197,189) |                |  49K   (49,514) |                |        0.047% |

# Reference  
Register conda environment to jupyter notebook : https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook  
Install RDkit : https://anaconda.org/rdkit/rdkit  

Handling Large Dataset : https://machinelearningmastery.com/large-data-files-machine-learning/  
Neat Tutorial to use HDF5 with python : https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/  
Convert String into HDF5 encoding : https://stackoverflow.com/questions/23220513/storing-a-list-of-strings-to-a-hdf5-dataset-from-python  
Loading List of HDF5 files with pytorch Dataset : https://discuss.pytorch.org/t/loading-huge-data-functionality/346/9   

Installing TensorboardX : https://github.com/lanpa/tensorboardX  
```
git clone https://github.com/lanpa/tensorboardX && cd tensorboardX && python setup.py install
```  

Compress and Extract datasetfile : https://www.cyberciti.biz/faq/how-do-i-compress-a-whole-linux-or-unix-directory/  
compress : ```tar -zcvf dataset.tar.gz dataset```  
extrace : ```tar -zxvf dataset.tar.gz```  

