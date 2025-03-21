- Enviroment setup
```
conda create -n mai
conda activate mai
pip install -r requirements.txt
conda install -c conda-forge cudatoolkit cudnn
python3 -m pip install 'tensorflow[and-cuda]'
```
