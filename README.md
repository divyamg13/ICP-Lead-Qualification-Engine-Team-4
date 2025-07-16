# ICP Lead Qualification Engine

## Project Overview

This project focuses on automating **lead qualification** using **AI-driven classification** and **ICP (Ideal Customer Profile) matching**. Designed during a Practice School-I internship at **CloudDefense.AI**, this system identifies high-quality leads through a combination of **synthetic data generation**, **attribute-based similarity scoring**, and **machine learning models** (Logistic Regression and Neural Network). The goal is to optimize outreach strategies for B2B sales pipelines by surfacing the most relevant contacts for each ICP template.

## Repository Contents

### Core Notebooks

| Notebook | Description |
|----------|-------------|
| `data_generator.ipynb` | Generates synthetic lead data with labeled ICP matches using realistic B2B attributes |
| `LogisticRegressionICPClassifier.ipynb` | Preprocesses data and trains a logistic regression model for ICP classification |
| `NeuralNetworkICPClassifier.ipynb` | Implements a PyTorch-based neural network (`ICPNet`) for non-linear ICP classification |

## Features

- **Best Match Algorithm**  
  Matches leads to the closest ICP using attribute-level Jaccard similarity and range overlap scoring.

- **Logistic Regression Classifier**  
  Simple, interpretable binary classifier using TF-IDF, one-hot encoding, and scaling.

- **ICPNet Neural Network**  
  Feedforward neural net with 14 → 24 → 1 architecture using ReLU and sigmoid activations.

- **Multi-label Format**  
  Each lead is associated with multiple ICPs (`ICP1` to `ICP5`), allowing flexible match scenarios.

- **Synthetic Dataset Generation**  
  Supports large-scale model training using fake but realistic data.

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone <your-repo-url>
   cd ICP-Lead-Qualification
   ```

2. **Create and Activate Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run Notebooks**

   Launch JupyterLab or Jupyter Notebook:

   ```bash
   jupyter lab
   ```

   Run the notebooks in the following order:
   - `data_generator.ipynb`
   - `LogisticRegressionICPClassifier.ipynb`
   - `NeuralNetworkICPClassifier.ipynb`

## Requirements

You can create a `requirements.txt` file like:

```txt
pandas
scikit-learn
matplotlib
numpy
torch
seaborn
```

## ICP Template Structure

Each ICP is a JSON-like object defining the target segment:

```json
{
  "industry": ["Healthcare", "MedTech"],
  "engagement_rate": "65-95",
  "company_size_employees": "100-800",
  "annual_revenue_usd": "10M-40M",
  "headquarters_location": "India",
  "technology_stack": ["Python", "AWS"],
  "target_designations": ["CTO", "Head of AI"],
  "pain_points": ["Compliance", "Real-time analytics"]
}
```

## Output

After model inference, leads predicted to match an ICP are saved/exported with selected fields:
- First name
- Last name
- Title
- Company
- Location
- Phone/email

These leads are ready for downstream sales workflows like email, LinkedIn, or CRM ingestion.

## Evaluation Results

| Model            | Accuracy (Avg) | Strengths                         |
|------------------|----------------|-----------------------------------|
| Logistic Regression | 95–98%         | Fast, interpretable, reliable      |
| Neural Network      | 85–90%         | Flexible, learns complex patterns  |

## Future Enhancements

- Add more real-world ICP templates across sectors
- Integrate with enrichment and CRM tools
- Support cross-validation and grid search for hyperparameter tuning

## Authors

- **Divyam Gupta** (2023A7PS0423G)  
- **Ragav Krishna Ramesh** (2023A7PS0415G)

## Acknowledgments

We are grateful to **CloudDefense.AI**, our mentor **Mr. Varendra Maurya**, and **Dr. Sharan Gopal** (PS Faculty) for their support during the Practice School-I internship.

