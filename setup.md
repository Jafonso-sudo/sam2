conda env create -f environment.yml 
conda activate sam2
conda install cuda -c nvidia/label/cuda-11.8.0
<!-- conda install nvidia/label/cuda-11.8.0::cuda -->
pip install -e .
pip install -e ".[notebooks]"


conda env remove --name sam2