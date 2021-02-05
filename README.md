# Forecasting with Spectral Mixture kernels
This is a companion repo for [my blog post](https://sebastiancallh.github.io/post/spectral-mixture-kernels/) which takes a look at the Spectral Mixture kernel as presented in [Gaussian Process Kernels for Pattern Discovery and Extrapolation](https://arxiv.org/abs/1302.4245).

## How to run
Make sure you run in a virtual environment, then `pip install -r requirements.txt` and you should be good to go. 
To fit the models and produce plots run the scripts in the [scripts](scripts) folder and provide the command line arguments you like.
For instance `python scripts/fit_mauna_loa.py --num-restarts 3 --lr 0.1 --num-iters 500 --kernel smk` will fit a model using only the SM kernel using three random restarts and 500 iterations per restart.  
