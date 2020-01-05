## The objectives of this tutorial
* Implement the forward and backward passes as well as the neural network training procedure
* Implement the widely-used optimizers and training tricks including dropout
* Get familiar with TensorFlow by training and designing a network on your own
* Visualize the learned weights and activation maps of a ConvNet
* Use Grad-CAM to visualize and reason why ConvNet makes certain predictions

# replace your_virtual_env with the virtual env name you want
virtualenv -p $(which python3) your_virtual_env
source your_virtual_env/bin/activate

# install dependencies other than tensorflow
pip3 install -r requirements.txt
# or
pip3 install numpy jupyter ipykernel opencv-python matplotlib

# install tensorflow (cpu version, recommended)
pip3 install tensorflow

# install tensorflow (gpu version)
# run this command only if your device supports gpu running
pip3 install tensorflow-gpu

deactivate # Exit the virtual environment
```

## Work with IPython Notebook
To start working on this, simply run the following command to start an ipython kernel.
```shell
# add your virtual environment to jupyter notebook
source your_virtual_env/bin/activate
python -m ipykernel install --user --name=your_virtual_env

# port is only needed if you want to work on more than one notebooks
jupyter notebook --port=your_port_number

```
and then work on each problem with their corresponding `.ipynb` notebooks.
Check the python environment you are using on the top right corner.
If the name of environment doesn't match, change it to your virtual environment in "Kernel>Change kernel".

