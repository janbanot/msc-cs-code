# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a university repository for the "Języki Skryptowe w Analizie Danych" (JSwAD) course containing Python laboratory exercises covering data analysis, machine learning, and text processing. The code is written in Polish with Polish comments and outputs.

## Project Structure

- **Lab Files**: Individual Python scripts (`lab1.py` through `lab6.py`) containing completed exercises
- **Notebooks**: Jupyter notebooks for more complex analysis (`lab5_JanBanot.ipynb`, `lab5-1_JanBanot.ipynb`, `lab5_numpy.ipynb`)
- **Data**: CSV files in `data/` directory containing datasets for analysis
- **Kaggle**: Competition-related files in `kaggle/` subdirectory

## Common Development Commands

### Running Lab Exercises
```bash
python lab1.py    # Basic Python functions and algorithms
python lab2.py    # Object-oriented programming with classes
python lab3.py    # Pandas data analysis
python lab4a.py   # Machine learning with scikit-learn
python lab6.py    # Currently empty/in development
```

### Working with Notebooks
```bash
jupyter notebook lab5_JanBanot.ipynb      # NumPy exercises
jupyter notebook lab5-1_JanBanot.ipynb    # Text analysis with NLTK
```

## Code Architecture

### Lab 1 (`lab1.py`)
Basic Python functions including:
- Mathematical calculations (factorial, quadratic equations)
- Interactive games and algorithms
- Tax calculation system

### Lab 2 (`lab2.py`)
Object-oriented programming featuring:
- Financial classes (`BankAccount`, `Fraction`, `ComplexNumber`)
- Animal hierarchy with inheritance (`Animal`, `Dog`, `Bird`, `Cat`)
- File I/O and simulation systems
- Cat toy preference simulation with random behavior

### Lab 3 (`lab3.py`)
Pandas data manipulation:
- CSV data loading and cleaning
- Data filtering and aggregation
- DataFrame merging and concatenation
- Analysis of automobile and alcohol consumption datasets

### Lab 4a (`lab4a.py`)
Machine learning experiments:
- Synthetic dataset generation using `make_classification`
- Model comparison: Gaussian Naive Bayes, SVM, Random Forest
- Hyperparameter tuning for SVM (C values) and Random Forest (n_estimators)
- Performance evaluation with accuracy, F1-score, and ROC AUC metrics

### Lab 5 Notebooks
- **lab5_JanBanot.ipynb**: NumPy array operations, polynomial evaluation, discrete random variables
- **lab5-1_JanBanot.ipynb**: Text frequency analysis using NLTK, Polish stopwords processing, constitutional text analysis with visualization

## Development Notes

- All code uses Polish language for comments, variable names, and output messages
- The repository follows a simple structure with numbered lab files
- Data files are stored in relative paths (`sem2/JSWAD/data/`)
- Code includes comprehensive examples of Python data science stack: pandas, numpy, scikit-learn, matplotlib, nltk

## Data Dependencies

The code expects the following data files to be present in `data/`:
- `automobile_data.csv` - Car specifications dataset
- `world_alcohol_data.csv` - Global alcohol consumption data
- `cat_toys.txt` - List of cat toys for simulation exercises

## Key Libraries Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **nltk**: Natural language processing
- **requests**: HTTP requests for data fetching