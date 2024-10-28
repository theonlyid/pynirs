# PYNIRS - a library for analysis of NIRS and IP timeseries data

Version: 0.0.1

Author: Ali Zaidi

## Installation

**NOTE: Make sure you are not in a folder updated by One-Drive / Box!**


The code needs to be cloned and run locally. You cannot install the package from the One-Drive foder.
<ol>

<li>
First you must close the package to a local folder. Use the command prompt to navigate to a local folder
Copy the package contents to a local folder. From the package root directory, run the following commands:

```
$ git clone file://"path-to-repo"
```

where path-to-repo is the path to the remote folder on One-Drive.
</li>

<li>
Then you need to install the code. Run the following commands from inside the cloned folder:

```
$ conda env create -f src/deps/environment.py
$ pip install .
```
</li>

This will install the necessary dependencies. You can now load the library in python.

<li>
To load and use the package in python, first activate the virtual environment,and then run python. You can do that from the shell:

```bash
$ conda activate pynirs
$ python

```

From Python you can load the library and seek help:
```python
> import pynirs
> help(pynirs)
```
</li>


# Updating the package contents

Now and again there will be updates to the code. To download and install updates, run the following commands:

```
$ git pull
$ pip install .
```

This will update  the source code and install the package.

## Test import

On first install, it would be good to test the package imports. 


## Examples

An example notebook is present in docs/notebooks. You can run it to see how the code works.

## Documentation

Documentatation is under construction. Basic documentation can be found by opening docs/pynirs/index.html. 

## Uninstalling / Reinstalling the package

To uninstall the package use the following command: `pip uninstall pynirs`

Reinstall it with pip while in the root folder: `pip install .`


#### Using PyNIRS with Matlab

Since Matlab needs to access python libraries, it is important to configure some paths.

1. From the Start Menu, launch the Anaconda Powershell prompt (Start Menu > Anaconda3 (64-bit) > Anaconda Powershell Prompt (anaconda3))
and activate the conda environment you installed the library in  (conda activate [environment-name]).

2. Navigate to the library's root directory. This is the directory where the contents of this repository reside.

3. Run ```python setup_matlab.py```. Wait for the prompt to confirm a 'Success!'. This should write a 'Matlab.bat' file in the current folder.

4. Copy the statement above the last line of the installation prompt. It starts with ```pyversion```. This needs to be entered in Matlab, as described below.

Finally, Matlab's own python interpreter needs to be configured.

1. Close all current instances of Matlab, and launch a new instance by double clicking "Matlab.bat" in the root folder.

2. In the Matlab prompt paste the command you copied and press return. Type ```pyversion``` and press return. It should show something like:

   ```java
   >> pyversion

         version: '3.11.9'
      executable: 'C:\Users\Continuum\anaconda3\python.exe'
         library: 'C:\Users\Continuum\anaconda3\python37.dll'
            home: 'C:\Users\Continuum\anaconda3'
         isloaded: 0
   ```


For an example on using the library with MATLAB, see the 'example_hbc_clean.m' located in docs/notebooks/matlab. 