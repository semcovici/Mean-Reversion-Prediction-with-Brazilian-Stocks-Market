# Mean Reversion Prediction with Brazilian Stocks Market

!!! This project is under development !!!

## About the study

[under development]


## How to replicate the experiments

### Dummy model

For the dummy model, follow the steps below:


Run `generate_dummy_results.py`. The results will be in `reports/test_results/`.

### Probabilistic models

[under development]

### Tensorflow models

For models based on TensorFlow, follow the steps below:

1. Create a virtual environment with Python 3.10.12:
   ```sh
   conda create --name env-tcc-tf-gpu python=3.10.12
   ```
2. Install TensorFlow and its dependencies by following the instructions at: [TensorFlow Installation Guide](https://www.tensorflow.org/install/pip)
3. Install the remaining libraries with:
   ```sh
   pip install -r requirements_nn.txt
   ```
4. Run `generate_LSTM_results.py` and `generate_MLP_results.py`. The results will be in `reports/test_results/` and the models created will be in `models/`.




conda create --name env-tcc-tf-gpu python=3.12.7
conda activate env-tcc-tf-gpu
conda install conda-forge::tensorflow
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia        
conda install scikit-learn
python -m pip install pykan
conda install matplotlib
conda install tqdm



### KAN models

[under development]


## Results

### How to generate the analysis of results


### Results


