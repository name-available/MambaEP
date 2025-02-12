# MambaEP: Efficient Spatio-Temporal Forecasting with Mamba Block and Fourier-Enhanced Temporal Modeling

## Abstract
Spatio-temporal forecasting plays a vital role in a range of scientific and practical applications, from weather prediction to traffic forecasting. However, existing models often struggle with balancing accuracy, efficiency, and scalability. This paper presents MambaEP, an efficient framework for spatio-temporal forecasting that integrates the Mamba Block with Fourier-enhanced temporal modeling. By combining local and global spatial feature extraction with multi-scale temporal processing, MambaEP achieves superior prediction accuracy while maintaining computational efficiency. Extensive experiments on benchmark datasets, including SEVIR, RainNet, and MovingMNIST, demonstrate that MambaEP outperforms state-of-the-art methods in both accuracy and efficiency, achieving faster convergence and smaller model sizes. Our model's ability to efficiently handle long-range dependencies, identify extreme cases, and reduce computational costs positions it as a promising solution for spatio-temporal forecasting tasks.

<p align="center" width="100%">
  <img src='figure/1Architecture.bmp' width="100%">
</p>

## Getting Started
1. Install Python 3.8. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```

2. Prepare Data. You can obtain experimental datasets from the following links.

| Dataset       | Task   | Categories | Link   |
| :-----------: | :----:| :------:| :---------: |
| WeatherBench | Predict future natural representation | Natural Scene | [[Google Cloud]](https://drive.google.com/drive/folders/1sPCg8nMuDa0bAWsHPwskKkPOzaVcBneD) |
| SEVIR | Predict future natural representation | Natural Scene | [[AWS Opendata]](https://registry.opendata.aws/sevir/) |
|   RainNet    | Predict future natural representation | Natural Scene |    [[Github Link]](https://github.com/neuralchen/RainNet)    |
| Moving MNIST |         Predict future image          | Synthetic | [[Link]](https://www.cs.toronto.edu/~nitish/unsupervised_video/) |

3. Use the following instructions to quickly run the code.

```python
python train_main.py --data_path Dataset/NavierStokes_V1e-5_N1200_T20.mat --num_epochs 100 --batch_size 5
```

