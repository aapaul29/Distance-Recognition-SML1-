# SML Project 1
This folder contains the code template for project 1.

## Environment Setup
### Local Installation (Recommended)
We recommend developing locally. Please set up an Anaconda environment running `python3.10`. Please check out the installation guide [here](https://lec.inf.ethz.ch/sml/2026/other/anaconda_guide_sml.pdf).

If you are using Windows, we recommend to use either the VS code terminal or the Anaconda terminal, which is installed with Anaconda.

Please activate your project 1 environment by using:
```
conda activate <environment_name>
```
Then navigate to the folder containing the project files and run:
```
pip install --upgrade pip
pip install -r requirements.txt
```
If you require any additional packages, run:
```
pip install <package_name>
```

Make sure to extract all the data in the `./data` folder.

### JupyterHub
> **Note:** You have a **50-hour total time limit** on JupyterHub for the entire semester. We recommend developing and debugging locally first, and only using JupyterHub when necessary.

For those who are running the project on JupyterHub, no setup is needed. Most packages have already been installed to this environment.

If you want to check the packages that have been installed, open a [terminal](https://jupyterlab.readthedocs.io/en/latest/user/terminal.html) and run:
```
pip list
```
If you require additional packages, please open a terminal and run:
```
pip install <package_name>
```

Please note that in this environment, you will not be able to directly access the data due to server-side limitations. You can rest assured that the `load_dataset` function does as described in the project description.

#### Known Limitations of JupyterHub
There exists a resource limit of 4GB RAM per person. This means that certain code will cause the process to be killed.

Specifically, using a `downsample_factor` of less than 5 with `load_rgb: True` will cause your code to crash, with the following error message:
```
[INFO]: Configs are loaded with:
 {'data_dir': PosixPath('/data/project1'), 'load_rgb': True, 'downsample_factor': 1}
Killed
```

## Running Code
### Local Installation
To run your solution locally, first make sure you have activated your conda environment. Then open a terminal and run:
```
python <./path/to>main.py
```

### JupyterHub
To run your solution on JupyterHub, first, navigate to the folder containing your implementation of the solution, `main.py`. Then save any changes to your solution. Finally, open a [terminal](https://jupyterlab.readthedocs.io/en/latest/user/terminal.html) and run:
```
python main.py
```
Another option is to copy and paste your code into a Jupyter Notebook cell and run it from there.
