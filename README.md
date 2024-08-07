# EM-pipeline
light-weighted multi-gpu, multi-core pipeline engine


## Installation
```
conda create -n "em_pipeline" python==3.8
source activate em_pipeline
conda install pip

git clone git@github.com:donglaiw/em_pipeline.git
cd em_pipeline
pip install --editable .
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
```