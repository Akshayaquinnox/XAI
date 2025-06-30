# Instructions

 ## Download the Zip File:
  - Go to the repository on GitHub https://github.com/Akshayaquinnox/XAI
  - Download the code from the main branch as a zip folder
  - Unzip the folder


## Install Python and pip (if not already installed)(we used Python 3.10 and pip 24.2 or Python 3.12.0 and pip 23.2.1):

 - Python: Download and install from https://www.python.org/downloads/
 - pip: Usually comes with Python. If not, follow instructions from https://pip.pypa.io/en/stable/installation/

## Create a virtual environment:
  ``` python -m venv venv ```
 -  Activate the virtual environment:

On MacOS/Linux:

bash
source venv/bin/activate


On Windows:

```bash
.\venv\Scripts\activate
```

Install required libraries
```bash
pip install shap pandas numpy scikit-learn matplotlib
```
. Run the notebook
```bash
jupyter notebook shap_analysis.ipynb
```

