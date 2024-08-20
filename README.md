# EM-pipeline
light-weighted multi-gpu, multi-core pipeline engine


## Installation
```
conda create -n "em_pipeline" python==3.8
source activate em_pipeline

git clone git@github.com:donglaiw/em_pipeline.git
cd em_pipeline
conda env create -f environment.yml
cd ..


git clone git@github.com:PytorchConnectomics/em_util.git
cd em_util
pip install --editable .
cd ..

git clone -b affuint8 git@github.com:donglaiw/waterz.git
cd waterz
pip install --editable .

git clone git@github.com:donglaiw/zwatershed.git
cd zwatershed
pip install --editable .
```

## Case 1: Zebrafinch 100um cube segmentation
- step 1. voxel to supervoxel (region graph)
    - Generate waterz chunk segmentation
      ```
      python main.py -c em_pipeline/data/j0126.yaml -t waterz
      ```   
    - Link waterz chunk segmentation into a global segmentation
      ```
      python main.py -c em_pipeline/data/j0126.yaml -t waterz-stats
      python main.py -c em_pipeline/data/j0126.yaml -t rg-border
      python main.py -c em_pipeline/data/j0126.yaml -t rg-all
      ```

- step 2. region graph algorithm
    - Find soma segment ids
    - Grow each cell as much as possible
    - Resolve false splits for each cells
    - Resolve false merges among cells
    
