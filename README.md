XAI - Explainable Artificial Intelligence
 This repository implements a movie-genre classification pipeline that extracts film metadata from
 DBpedia via SPARQL queries and uses multiple XAI techniques (SHAP, permutation importance, surrogate
 models) to interpret a Gradient Boosting classifier's decision-making process. The project demonstrates
 how director and actor metadata drive genre predictions between comedies and dramas, providing
 transparency into model behavior through complementary interpretability methods.
 Setup Instructions
 Prerequisites
 Python 3.10+ or Python 3.12.0
 pip 24.2+ or pip 23.2.1
 Conda (recommended)
 Installation
 1. Clone the repository
 bash
 git clone https://github.com/Akshayaquinnox/XAI.git
 git
 cd
 clone https://github.com/Akshayaquinnox/XAI.git
 cd XAI
 XAI
 2. Create conda environment
 bash
 conda create -n xai-env python
 conda create -n xai-env 
conda activate xai-env
 conda activate xai-env
 python= =3.10
 3.10
 3. Install required packages
 bash
 pip 
pip install
 install -r requirements.txt-r requirements.txt
 Or install packages manually:
 bash
 pip 
pip install
 install shap pandas numpy scikit-learn matplotlib jupyter rdflib sparqlwrapper
 shap pandas numpy scikit-learn matplotlib jupyter rdflib sparqlwrapper
 Running the Analysis
 1. Start Jupyter Notebook
bash
 jupyter notebook
 jupyter notebook
 2. Open the main notebook
 shap_analysis.ipynb
 shap_analysis.ipynb
 3. Run all cells to execute the SHAP analysis
 Alternative Setup (Virtual Environment)
 If you prefer using venv instead of conda:
 bash
 python -m venv venv
 python -m venv venv
 # Activate environment
 # Activate environment
 # On macOS/Linux:
 source
 # On macOS/Linux:
 source venv/bin/activate
 venv/bin/activate
 # On Windows:
 .
 # On Windows:
 .\ \venv
 venv\ \Scripts
 Scripts\ \activate
 activate
 # Install dependencies
 pip 
# Install dependencies
 pip install
 install shap pandas numpy scikit-learn matplotlib jupyter rdflib sparqlwrapper
 shap pandas numpy scikit-learn matplotlib jupyter rdflib sparqlwrapper
 Project Structure
 XAI/
 ├── shap_analysis.ipynb    # Main analysis notebook
 XAI/
 ├── shap_analysis.ipynb    # Main analysis notebook
 ├── requirements.txt       
├── requirements.txt       
└── README.md             
└── README.md             
Usage
 # Python dependencies
 # Python dependencies
 # This file
 # This file
 The notebook demonstrates:
 DBpedia data extraction using SPARQL queries for movie metadata
 Feature engineering from RDF triples (directors, actors, producers, distributors)
 Gradient Boosting classifier training for comedy vs drama classification
 Four interpretability techniques: Gini importance, permutation importance, SHAP values, and
 surrogate decision trees
 Comprehensive model explanation revealing how director and actor metadata influence predictions
 Requirements
Core dependencies include:
 shap - For explainable AI analysis
 pandas - Data manipulation
 numpy - Numerical computations
 scikit-learn - Machine learning models (Gradient Boosting)
 matplotlib - Data visualization
 jupyter - Interactive notebook environment
 rdflib - RDF data processing
 sparqlwrapper - SPARQL query execution for DBpedia
