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

On first install, it would be good to test the package. 


## Examples

An example notebook is present in docs/notebooks. You can run it to see how the code works.

## Documentation

Documentatation is under construction. Basic documentation can be found by opening docs/pynirs/index.html. 

## Uninstalling / Reinstalling the package

To uninstall the package use the following command: `pip uninstall pynirs`

Reinstall it with pip while in the root folder: `pip install .`

