# FLOW19_MLSchool
Code for the hands-on sessions of the 2019 FLOW School on Machine Learning 

## Regression session
### To run the notebook
* Using Docker (as root):  
```docker run -i -t -p 8888:8888 -v "$PWD:/home/" continuumio/anaconda3 /bin/bash -c "/opt/conda/bin/jupyter notebook --notebook-dir=/home/ --ip='0.0.0.0' --port=8888 --no-browser --allow-root"```  
* Using local Anaconda installation (with Jupyter notebook installed)   
  Clone the repository, then launch from terminal  
  ```jupyter-notebook```  
  Browser should open automatically  
* Using Anaconda installation (with Jupyter notebook installed) on a remote computer  
  Clone the repository on the remote computer, then launch from remote terminal  
  ```jupyter-notebook --no-browser--port=XXXX```  
  The port needs to be forwarded to connect to the notebook from the local computer. Connect to the remote computer using
  ```ssh -N -f -L localhost:YYYY:localhost:XXXX remoteuser@remotehost```  
  Connect to the notebook using the browser at the URL *"localhost:YYYY"*

**fit-sine.ipynb**  
Noisy samples from a sine functions are taken and fit with a polynomial functions of increasing order. The concepts of overfitting and L2 regularization, as well as stochastic gradient descent (SGD) are introduced. 
