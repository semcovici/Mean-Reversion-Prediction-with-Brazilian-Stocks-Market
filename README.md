# Mean Reversion Prediction with Brazilian Stocks Market

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
