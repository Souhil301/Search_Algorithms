# SEARCH ALGORITHMS PATTERN IN TEXT

# Description
The Bioinformatics Pattern Matcher is a performance testing framework designed to evaluate various pattern-matching algorithms commonly used in bioinformatics, particularly for genomic sequence analysis. It allows users to run tests on different algorithms, compare their performance, and visualize the results. The framework includes both built-in test cases (best, worst, random) and the ability for users to define their own custom test cases.

This project implements a suite of classical string-searching algorithms and provides tools to measure their efficiency, including:

Morris-Pratt (MP)

Boyer-Moore (BM)

Rabin-Karp (RK)

Aho-Corasick (AC)

Commentz-Walter (CW)

Features
Testing Algorithms: Run a single algorithm or compare multiple algorithms on various test cases.

Custom Test Cases: Users can define custom text and pattern lengths for more flexible testing.

Performance Metrics: Collect and display results including execution time, comparisons made, and occurrences found.

Data Visualization: Display results as tables and plots, and export them as CSV files.

Interactive User Interface: Easy-to-use menu-driven interface for configuring tests, running tests, and viewing results.


## Prerequisites

Before running the application, make sure you have the following installed:

- **Python 3.x**: You can download Python from [here](https://www.python.org/downloads/).
- **pip**: This is the Python package manager, and it is included with Python installations.

## Setup

To get started with this project, follow these steps:

# 1. Clone the Repository

First, clone the repository to your local machine:

access the root directory of the project



# 2. Set up the Virtual Environment : type

## Create a virtual environment in your project directory to isolate the dependencies:

python3 -m venv venv

"""
## Activate the virtual environment:

### On macOS/Linux:
source venv/bin/activate

### On Windows:
venv\Scripts\activate


# 3. Install Dependencies
pip install -r requirements.txt


# 4. Running the Application
python app.py
### or
python3 app.py



## 👥 Authors

Built by:

**Oussama BOUSSAHLA**  
GitHub: [@BSHLoussama](https://github.com/BSHLoussama)  
Email: oussamaboussahla2017@gmail.com

**Souhil MOKEDEM**  
GitHub: [@Souhil301](https://github.com/Souhil301)  
Email: mokeddemsouhil968@gmail.com

---

## 📝 License

MIT License — feel free to use, share, and modify.

