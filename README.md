# MambaEP: Efficient Spatio-Temporal Forecasting with Mamba Block and Fourier-Enhanced Temporal Modeling

## Abstract
Spatio-temporal forecasting plays a vital role in a range of scientific and practical applications, from weather prediction to traffic forecasting. However, existing models often struggle with balancing accuracy, efficiency, and scalability. This paper presents MambaEP, an efficient framework for spatio-temporal forecasting that integrates the Mamba Block with Fourier-enhanced temporal modeling. By combining local and global spatial feature extraction with multi-scale temporal processing, MambaEP achieves superior prediction accuracy while maintaining computational efficiency. Extensive experiments on benchmark datasets, including SEVIR, RainNet, and MovingMNIST, demonstrate that MambaEP outperforms state-of-the-art methods in both accuracy and efficiency, achieving faster convergence and smaller model sizes. Our model's ability to efficiently handle long-range dependencies, identify extreme cases, and reduce computational costs positions it as a promising solution for spatio-temporal forecasting tasks.

<p align="center" width="100%">
  <img src='figure/1Architecture.bmp' width="100%">
</p>



## Main Result

Figure below present the evaluation outcomes of MambaEP alongside ten state-of-the-art (SOTA) baselines across six datasets. On the Moving MNIST dataset, MambaEP achieves the lowest MSE (scaled by 100), outperforming prominent methods such as EarthFarsser, Earthformer, and SimVP. Similarly, on the Component of Wind dataset, MambaEP achieves the lowest MSE and MAE scores, demonstrating superior accuracy compared to other approaches. Overall, MambaEP attains the best average performance ranking of 2.09, highlighting its consistent effectiveness across all datasets. These results validate the model's ability to capture and model spatial features effectively through the integration of the Mamba Block.

<p align="center" width="100%">
  <img src='figure/3MainResult.bmp' width="100%">
</p>

## Getting Started

1. Install Python 3.10. For convenience, execute the following command.

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

3. Use the following instructions to quickly run the code. (Using Moving MNIST as a demonstration example)

```python
python main.py --data_path dataset --num_epochs 100 --batch_size 4 --val_batch_size 4 --lr 0.001
```

