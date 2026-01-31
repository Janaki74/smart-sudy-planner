# app.py - Complete Smart Study Planner with Chatbot (Single File)
import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime, timedelta
import random
import time
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ---------- Configuration ----------
st.set_page_config(
    page_title="Smart Study Planner",
    page_icon="📘",
    layout="wide"
)

# ---------- File Paths ----------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

USERS_FILE = os.path.join(DATA_DIR, "users.csv")
DATA_FILE = os.path.join(DATA_DIR, "study_tracker_large_features.csv")
PROGRESS_FILE = os.path.join(DATA_DIR, "user_progress.json")
STUDY_LOG_FILE = os.path.join(DATA_DIR, "study_logs.json")
TOPIC_PROGRESS_FILE = os.path.join(DATA_DIR, "topic_progress.json")
COMPLETED_NOTES_FILE = os.path.join(DATA_DIR, "completed_notes.json")
CHAT_HISTORY_FILE = os.path.join(DATA_DIR, "chat_history.json")

# ---------- Topic Notes Database ----------
TOPIC_NOTES = {
    # Maths Topics
    "Calculus": {
        "notes": [
            "📌 **Derivatives**: Rate of change of a function",
            "📌 **Integrals**: Area under a curve",
            "📌 **Limits**: Behavior of functions at points",
            "📌 **Differentiation Rules**: Product, quotient, chain rules",
            "📌 **Integration Techniques**: Substitution, integration by parts"
        ],
        "resources": [
            "📚 Textbook: Chapter 3-5", 
            "🎥 Video: Khan Academy Calculus",
            "📝 Notes: [Calculus Complete Notes PDF](https://example.com/calculus_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=WUvTyaaNkzM&list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr"
    },
    "Algebra": {
        "notes": [
            "📌 **Linear Equations**: y = mx + b form",
            "📌 **Quadratic Equations**: ax² + bx + c = 0",
            "📌 **Polynomials**: Degree, roots, factorization",
            "📌 **Matrices**: Addition, multiplication, determinants",
            "📌 **Vectors**: Magnitude, direction, operations"
        ],
        "resources": [
            "📚 Textbook: Chapter 1-2", 
            "🎥 Video: 3Blue1Brown Algebra",
            "📝 Notes: [Linear Algebra Notes PDF](https://example.com/algebra_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=ZK3O402wf1c&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
    },
    "Vectors": {
        "notes": [
            "📌 **Vector Components**: i, j, k notation",
            "📌 **Dot Product**: a · b = |a||b|cosθ",
            "📌 **Cross Product**: a × b = |a||b|sinθ n",
            "📌 **Vector Projection**: Projection of a onto b",
            "📌 **Vector Spaces**: Basis, dimension, linear independence"
        ],
        "resources": [
            "📚 Textbook: Chapter 7", 
            "🎥 Video: MIT OCW Vectors",
            "📝 Notes: [Vectors and Matrices PDF](https://example.com/vectors_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
    },
    "Matrices": {
        "notes": [
            "📌 **Matrix Operations**: Addition, subtraction, multiplication",
            "📌 **Determinants**: Properties and calculation",
            "📌 **Inverse Matrices**: Finding and properties",
            "📌 **Eigenvalues & Eigenvectors**: Definition and calculation",
            "📌 **Linear Transformations**: Matrix representation"
        ],
        "resources": [
            "📚 Textbook: Chapter 8", 
            "🎥 Video: Gilbert Strang Lectures",
            "📝 Notes: [Matrix Theory Complete PDF](https://example.com/matrices_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=ZK3O402wf1c&list=PLUl4u3cNGP61iQEFiWLE21EJCxwmWvvek"
    },
    
    # Python Topics
    "Functions": {
        "notes": [
            "📌 **Function Definition**: def function_name(parameters):",
            "📌 **Parameters vs Arguments**: Positional, keyword, default",
            "📌 **Return Statement**: Returning values from functions",
            "📌 **Lambda Functions**: Anonymous inline functions",
            "📌 **Scope**: Local, global, and nonlocal variables"
        ],
        "resources": [
            "📚 Python Documentation", 
            "🎥 Corey Schafer Python Functions",
            "📝 Notes: [Python Functions Cheat Sheet](https://example.com/python_functions.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=9Os0o3wzS_I"
    },
    "Pandas": {
        "notes": [
            "📌 **DataFrames**: 2D labeled data structure",
            "📌 **Series**: 1D labeled array",
            "📌 **Data Selection**: loc[], iloc[], conditional",
            "📌 **GroupBy**: Split-apply-combine operations",
            "📌 **Merging/Joining**: concat, merge, join operations"
        ],
        "resources": [
            "📚 Pandas Documentation", 
            "🎥 Data School Pandas",
            "📝 Notes: [Pandas Complete Guide PDF](https://example.com/pandas_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=vmEHCJofslg&list=PL-osiE80TeTsWmV9i9c58mdDCSskIFdDS"
    },
    "Matplotlib": {
        "notes": [
            "📌 **Pyplot Interface**: plt.plot(), plt.scatter(), plt.bar()",
            "📌 **Figure & Axes**: Creating subplots",
            "📌 **Customization**: Colors, markers, line styles",
            "📌 **Labels & Titles**: plt.xlabel(), plt.title()",
            "📌 **Saving Figures**: plt.savefig()"
        ],
        "resources": [
            "📚 Matplotlib Documentation", 
            "🎥 Sentdex Matplotlib Tutorial",
            "📝 Notes: [Matplotlib Visualization Guide](https://example.com/matplotlib_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=UO98lJQ3QGI&list=PL-osiE80TeTvipOqomVEeZ1HRrcEvtZB_"
    },
    "Loops": {
        "notes": [
            "📌 **For Loops**: for item in iterable:",
            "📌 **While Loops**: while condition:",
            "📌 **Loop Control**: break, continue, pass",
            "📌 **Nested Loops**: Loops inside loops",
            "📌 **List Comprehensions**: [expression for item in iterable]"
        ],
        "resources": [
            "📚 Python Documentation", 
            "🎥 Real Python Loops Guide",
            "📝 Notes: [Python Loops Cheat Sheet](https://example.com/loops_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=JkQ0Xeg8LRI"
    },
    
    # Machine Learning Topics
    "Linear Regression": {
        "notes": [
            "📌 **Simple Linear Regression**: y = mx + c",
            "📌 **Multiple Linear Regression**: y = b0 + b1x1 + b2x2 + ...",
            "📌 **Cost Function**: Mean Squared Error (MSE)",
            "📌 **Gradient Descent**: Optimization algorithm",
            "📌 **R-squared**: Goodness of fit measure"
        ],
        "resources": [
            "📚 ISLR Book", 
            "🎥 Andrew Ng ML Course",
            "📝 Notes: [Linear Regression Complete Notes](https://example.com/linear_regression_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=4b5d3muPQmA"
    },
    "Clustering": {
        "notes": [
            "📌 **K-Means**: Partitioning method",
            "📌 **Hierarchical Clustering**: Agglomerative/Divisive",
            "📌 **DBSCAN**: Density-based clustering",
            "📌 **Silhouette Score**: Cluster quality measure",
            "📌 **Elbow Method**: Finding optimal k"
        ],
        "resources": [
            "📚 Pattern Recognition Book", 
            "🎥 StatQuest Clustering",
            "📝 Notes: [Clustering Algorithms PDF](https://example.com/clustering_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=4b5d3muPQmA"
    },
    "Decision Trees": {
        "notes": [
            "📌 **Entropy**: Measure of impurity",
            "📌 **Information Gain**: Splitting criterion",
            "📌 **Gini Index**: Alternative impurity measure",
            "📌 **Pruning**: Reducing overfitting",
            "📌 **Random Forests**: Ensemble of trees"
        ],
        "resources": [
            "📚 ML Book by Mitchell", 
            "🎥 Josh Starmer Decision Trees",
            "📝 Notes: [Decision Trees & Random Forest PDF](https://example.com/decision_trees_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=_L39rN6gz7Y"
    },
    "KNN": {
        "notes": [
            "📌 **K-Nearest Neighbors**: Instance-based learning",
            "📌 **Distance Metrics**: Euclidean, Manhattan, Minkowski",
            "📌 **Choosing K**: Bias-variance tradeoff",
            "📌 **Feature Scaling**: Normalization/Standardization",
            "📌 **Weighted KNN**: Distance-weighted voting"
        ],
        "resources": [
            "📚 ML Textbook", 
            "🎥 Krish Naik KNN Tutorial",
            "📝 Notes: [KNN Algorithm Complete Guide](https://example.com/knn_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=HVXime0nQeI"
    },
    
    # Statistics Topics
    "Regression": {
        "notes": [
            "📌 **Linear Regression**: Modeling relationships",
            "📌 **Logistic Regression**: Binary classification",
            "📌 **Assumptions**: Linearity, independence, homoscedasticity",
            "📌 **Multicollinearity**: Correlation among predictors",
            "📌 **Model Evaluation**: R², Adjusted R², RMSE"
        ],
        "resources": [
            "📚 Statistical Learning Book", 
            "🎥 StatQuest Regression",
            "📝 Notes: [Regression Analysis Notes PDF](https://example.com/regression_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=zITIFTsivN8"
    },
    "Probability": {
        "notes": [
            "📌 **Basic Concepts**: Sample space, events, outcomes",
            "📌 **Bayes Theorem**: P(A|B) = P(B|A)P(A)/P(B)",
            "📌 **Distributions**: Normal, Binomial, Poisson",
            "📌 **Expectation & Variance**: Mean and spread",
            "📌 **Central Limit Theorem**: Sample means distribution"
        ],
        "resources": [
            "📚 Probability & Statistics Book", 
            "🎥 3Blue1Brown Probability",
            "📝 Notes: [Probability Theory Complete PDF](https://example.com/probability_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=ZA4JkHKZM50"
    },
    "Distributions": {
        "notes": [
            "📌 **Normal Distribution**: Bell curve, μ and σ",
            "📌 **Binomial Distribution**: n trials, p probability",
            "📌 **Poisson Distribution**: Events in fixed interval",
            "📌 **Exponential Distribution**: Time between events",
            "📌 **t-Distribution**: Small sample inference"
        ],
        "resources": [
            "📚 Statistics Textbook", 
            "🎥 Khan Academy Distributions",
            "📝 Notes: [Statistical Distributions PDF](https://example.com/distributions_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=ZA4JkHKZM50"
    },
    "Hypothesis Testing": {
        "notes": [
            "📌 **Null & Alternative Hypothesis**: H0 and H1",
            "📌 **p-value**: Probability under H0",
            "📌 **Type I & II Errors**: α and β errors",
            "📌 **t-test**: Comparing means",
            "📌 **Chi-square test**: Categorical data"
        ],
        "resources": [
            "📚 Stats Book", 
            "🎥 StatQuest Hypothesis Testing",
            "📝 Notes: [Hypothesis Testing Guide PDF](https://example.com/hypothesis_testing_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=VK-rnA3-41c"
    },
    
    # Data Science Topics
    "Data Cleaning": {
        "notes": [
            "📌 **Handling Missing Values**: Imputation methods",
            "📌 **Outlier Detection**: IQR, Z-score methods",
            "📌 **Data Normalization**: Min-max, z-score scaling",
            "📌 **Encoding Categorical Variables**: One-hot, label encoding",
            "📌 **Feature Engineering**: Creating new features"
        ],
        "resources": [
            "📚 Data Science Handbook", 
            "🎥 Data School Cleaning",
            "📝 Notes: [Data Cleaning Complete Guide](https://example.com/data_cleaning_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=8gVLRXG9gNo"
    },
    "Feature Engineering": {
        "notes": [
            "📌 **Feature Creation**: Polynomial, interaction terms",
            "📌 **Feature Selection**: Filter, wrapper, embedded methods",
            "📌 **Dimensionality Reduction**: PCA, t-SNE",
            "📌 **Binning**: Converting continuous to categorical",
            "📌 **Text Features**: TF-IDF, word embeddings"
        ],
        "resources": [
            "📚 Feature Engineering Book", 
            "🎥 Kaggle Feature Engineering",
            "📝 Notes: [Feature Engineering PDF](https://example.com/feature_engineering_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=9UwNkfYtyvY"
    },
    "PCA": {
        "notes": [
            "📌 **Principal Components**: Orthogonal directions",
            "📌 **Variance Explained**: Selecting components",
            "📌 **Eigenvectors & Eigenvalues**: Mathematical basis",
            "📌 **Data Standardization**: Before applying PCA",
            "📌 **Applications**: Visualization, noise reduction"
        ],
        "resources": [
            "📚 ML Textbook", 
            "🎥 StatQuest PCA",
            "📝 Notes: [PCA Complete Mathematical Guide](https://example.com/pca_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=FgakZw6K1QQ"
    },
    "EDA": {
        "notes": [
            "📌 **Summary Statistics**: Mean, median, mode, std",
            "📌 **Data Visualization**: Histograms, scatter plots",
            "📌 **Correlation Analysis**: Pearson, Spearman",
            "📌 **Distribution Analysis**: Skewness, kurtosis",
            "📌 **Missing Data Patterns**: Heatmaps, missingno"
        ],
        "resources": [
            "📚 EDA Book", 
            "🎥 Datacamp EDA Course",
            "📝 Notes: [Exploratory Data Analysis Guide](https://example.com/eda_notes.pdf)"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=GPVsHOlRBBI"
    }
}

# ---------- Subject Knowledge Base for Chatbot ----------
SUBJECT_KNOWLEDGE = {
    "Mathematics": {
        "Calculus": {
            "differentiation": {
                "definition": "**Differentiation** is the process of finding the derivative of a function, which represents the rate of change.",
                "rules": [
                    "**Power Rule**: d/dx(xⁿ) = n·xⁿ⁻¹",
                    "**Product Rule**: d/dx(uv) = u'v + uv'",
                    "**Quotient Rule**: d/dx(u/v) = (u'v - uv')/v²",
                    "**Chain Rule**: d/dx(f(g(x))) = f'(g(x))·g'(x)"
                ],
                "examples": [
                    "**Example 1**: Find derivative of f(x) = 3x² + 2x - 5\n**Solution**: f'(x) = 6x + 2",
                    "**Example 2**: Find derivative of f(x) = sin(x²)\n**Solution**: f'(x) = cos(x²)·2x = 2x·cos(x²)",
                    "**Example 3**: Find derivative of f(x) = eˣ·ln(x)\n**Solution**: Using product rule:\nf'(x) = eˣ·ln(x) + eˣ·(1/x) = eˣ(ln(x) + 1/x)"
                ],
                "real_world": "• **Physics**: Velocity is derivative of position\n• **Economics**: Marginal cost is derivative of total cost\n• **Biology**: Growth rate of population"
            },
            "integration": {
                "definition": "**Integration** is the reverse process of differentiation, finding the area under a curve.",
                "methods": [
                    "**Basic Integration**: ∫xⁿ dx = xⁿ⁺¹/(n+1) + C",
                    "**Substitution**: ∫f(g(x))g'(x)dx = ∫f(u)du",
                    "**Integration by Parts**: ∫u dv = uv - ∫v du",
                    "**Partial Fractions**: For rational functions"
                ],
                "examples": [
                    "**Example 1**: ∫(3x² + 2x)dx = x³ + x² + C",
                    "**Example 2**: ∫2x·e^(x²) dx\nLet u = x², du = 2x dx\n= ∫eᵘ du = eᵘ + C = e^(x²) + C",
                    "**Example 3**: ∫x·eˣ dx\nUsing integration by parts:\nLet u = x, dv = eˣ dx\n= x·eˣ - ∫eˣ dx = x·eˣ - eˣ + C"
                ]
            },
            "limits": {
                "definition": "**Limit** describes the behavior of a function as it approaches a certain point.",
                "rules": [
                    "**Direct Substitution**: If f(a) is defined, lim(x→a) f(x) = f(a)",
                    "**L'Hôpital's Rule**: For 0/0 or ∞/∞ forms",
                    "**Squeeze Theorem**: If g(x) ≤ f(x) ≤ h(x) and lim g(x) = lim h(x) = L, then lim f(x) = L"
                ],
                "examples": [
                    "**Example 1**: lim(x→2) (x² - 4)/(x - 2)\n= lim(x→2) (x-2)(x+2)/(x-2) = lim(x→2) (x+2) = 4",
                    "**Example 2**: lim(x→0) sin(x)/x = 1 (Important limit)",
                    "**Example 3**: lim(x→∞) (3x² + 2x)/(5x² - x)\nDivide numerator and denominator by x²:\n= lim(x→∞) (3 + 2/x)/(5 - 1/x) = 3/5"
                ]
            }
        },
        "Algebra": {
            "linear_equations": {
                "definition": "**Linear Equations** are equations of the first degree (no exponents higher than 1).",
                "forms": [
                    "**Slope-intercept**: y = mx + b",
                    "**Standard form**: Ax + By = C",
                    "**Point-slope**: y - y₁ = m(x - x₁)"
                ],
                "solving_methods": [
                    "**Substitution**: Solve one equation for one variable, substitute into other",
                    "**Elimination**: Add/subtract equations to eliminate a variable",
                    "**Graphical**: Find intersection point"
                ],
                "examples": [
                    "**Example**: Solve 2x + 3y = 12 and x - y = 1\nFrom second: x = y + 1\nSubstitute: 2(y+1) + 3y = 12\n2y + 2 + 3y = 12\n5y = 10 ⇒ y = 2, x = 3"
                ]
            },
            "quadratic_equations": {
                "definition": "**Quadratic Equations** are second-degree polynomials: ax² + bx + c = 0",
                "solutions": [
                    "**Quadratic Formula**: x = [-b ± √(b² - 4ac)] / 2a",
                    "**Factoring**: Find factors that multiply to ac and add to b",
                    "**Completing the Square**: Convert to (x - h)² = k form"
                ],
                "discriminant": "D = b² - 4ac\n• D > 0: Two real roots\n• D = 0: One real root\n• D < 0: Two complex roots",
                "examples": [
                    "**Example**: Solve x² - 5x + 6 = 0\nFactoring: (x-2)(x-3) = 0\nx = 2 or x = 3",
                    "**Example**: Solve 2x² - 4x - 6 = 0\nUsing formula: a=2, b=-4, c=-6\nx = [4 ± √(16 + 48)] / 4 = [4 ± √64] / 4\nx = [4 ± 8] / 4 ⇒ x = 3 or x = -1"
                ]
            },
            "matrices": {
                "definition": "**Matrices** are rectangular arrays of numbers arranged in rows and columns.",
                "operations": [
                    "**Addition**: Add corresponding elements",
                    "**Multiplication**: Row × Column dot product",
                    "**Determinant**: Scalar value for square matrices",
                    "**Inverse**: A⁻¹ such that A·A⁻¹ = I"
                ],
                "examples": [
                    "**Addition**: [1 2] + [3 4] = [4 6]\n      [5 6]   [7 8]   [12 14]",
                    "**Multiplication**: [1 2] × [5 6] = [1×5+2×7 1×6+2×8] = [19 22]\n      [3 4]   [7 8]   [3×5+4×7 3×6+4×8]   [43 50]"
                ]
            }
        }
    },
    "Python": {
        "Functions": {
            "definition": "**Functions** are reusable blocks of code that perform specific tasks.",
            "syntax": "```python\ndef function_name(parameters):\n    '''Docstring'''\n    # function body\n    return value\n```",
            "types": [
                "**Built-in**: print(), len(), type()",
                "**User-defined**: Created by programmer",
                "**Lambda**: Anonymous inline functions"
            ],
            "examples": [
                "**Basic Function**:\n```python\ndef greet(name):\n    return f\"Hello, {name}!\"\n\nprint(greet(\"Alice\"))  # Hello, Alice!\n```",
                "**Function with Default Parameters**:\n```python\ndef power(base, exponent=2):\n    return base ** exponent\n\nprint(power(3))     # 9\nprint(power(3, 3))  # 27\n```",
                "**Lambda Function**:\n```python\nsquare = lambda x: x ** 2\nprint(square(5))  # 25\n```"
            ]
        },
        "Pandas": {
            "definition": "**Pandas** is a Python library for data manipulation and analysis.",
            "key_objects": [
                "**Series**: 1D labeled array",
                "**DataFrame**: 2D labeled data structure (like Excel sheet)"
            ],
            "common_operations": [
                "**Reading Data**: pd.read_csv(), pd.read_excel()",
                "**Viewing Data**: df.head(), df.tail(), df.info()",
                "**Selection**: df['column'], df.loc[], df.iloc[]",
                "**Filtering**: df[df['age'] > 30]",
                "**Grouping**: df.groupby('category').mean()"
            ],
            "examples": [
                "**Creating DataFrame**:\n```python\nimport pandas as pd\ndata = {'Name': ['Alice', 'Bob', 'Charlie'],\n        'Age': [25, 30, 35],\n        'City': ['NYC', 'LA', 'Chicago']}\ndf = pd.DataFrame(data)\n```",
                "**Filtering Example**:\n```python\n# Get people older than 28\nadults = df[df['Age'] > 28]\n```",
                "**Grouping Example**:\n```python\n# Average age by city\navg_age = df.groupby('City')['Age'].mean()\n```"
            ]
        },
        "Loops": {
            "definition": "**Loops** are used to execute a block of code multiple times.",
            "types": [
                "**for loop**: Iterates over a sequence",
                "**while loop**: Repeats while condition is true"
            ],
            "control_statements": [
                "**break**: Exit the loop",
                "**continue**: Skip to next iteration",
                "**pass**: Do nothing (placeholder)"
            ],
            "examples": [
                "**for loop**:\n```python\nfor i in range(5):\n    print(i)  # 0 1 2 3 4\n```",
                "**while loop**:\n```python\ncount = 0\nwhile count < 5:\n    print(count)\n    count += 1\n```",
                "**List comprehension**:\n```python\nsquares = [x**2 for x in range(10)]\n```"
            ]
        }
    },
    "Machine Learning": {
        "Linear Regression": {
            "definition": "**Linear Regression** models the relationship between a dependent variable and one or more independent variables using a linear equation.",
            "equation": "y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε",
            "key_concepts": [
                "**Cost Function**: Mean Squared Error (MSE) = (1/n)∑(yᵢ - ŷᵢ)²",
                "**Gradient Descent**: Optimization algorithm to minimize cost",
                "**R-squared**: Measure of how well regression line fits data (0-1)"
            ],
            "assumptions": [
                "Linearity between variables",
                "Independent observations",
                "Homoscedasticity (constant variance)",
                "No multicollinearity",
                "Normal distribution of errors"
            ],
            "examples": [
                "**Simple Linear Regression**:\n```python\nfrom sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)\npredictions = model.predict(X_test)\n```",
                "**Interpretation**: If β₁ = 2.5, for every 1 unit increase in X, Y increases by 2.5 units"
            ]
        },
        "Decision Trees": {
            "definition": "**Decision Trees** are tree-like models used for classification and regression.",
            "key_concepts": [
                "**Root Node**: Starting point",
                "**Decision Nodes**: Make splits",
                "**Leaf Nodes**: Final predictions",
                "**Splitting Criteria**: Gini Index or Entropy"
            ],
            "advantages": [
                "Easy to understand and interpret",
                "Handles both numerical and categorical data",
                "Requires little data preprocessing"
            ],
            "examples": [
                "**Classification Tree**:\n```python\nfrom sklearn.tree import DecisionTreeClassifier\nclf = DecisionTreeClassifier()\nclf.fit(X_train, y_train)\n```",
                "**Visualizing Tree**:\n```python\nfrom sklearn.tree import plot_tree\nimport matplotlib.pyplot as plt\nplt.figure(figsize=(12,8))\nplot_tree(clf, filled=True)\nplt.show()\n```"
            ]
        }
    }
}

# ---------- Persistent Storage Functions ----------
def save_user_progress():
    """Save user progress to JSON file"""
    try:
        progress_data = {
            "user_email": st.session_state.user_email,
            "streak": st.session_state.streak,
            "mood": st.session_state.mood,
            "selected_subject": st.session_state.selected_subject,
            "selected_topic": st.session_state.selected_topic
        }
        
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                all_progress = json.load(f)
        else:
            all_progress = {}
        
        all_progress[st.session_state.user_email] = progress_data
        
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(all_progress, f, indent=4)
    except Exception as e:
        st.error(f"Error saving progress: {e}")

def load_user_progress():
    """Load user progress from JSON file"""
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                all_progress = json.load(f)
            
            if st.session_state.user_email in all_progress:
                progress_data = all_progress[st.session_state.user_email]
                st.session_state.streak = progress_data.get("streak", 0)
                st.session_state.mood = progress_data.get("mood", "")
                st.session_state.selected_subject = progress_data.get("selected_subject", "")
                st.session_state.selected_topic = progress_data.get("selected_topic", "")
                return True
    except Exception as e:
        st.error(f"Error loading progress: {e}")
    return False

def save_study_log():
    """Save study log to JSON file"""
    try:
        if os.path.exists(STUDY_LOG_FILE):
            with open(STUDY_LOG_FILE, 'r') as f:
                all_logs = json.load(f)
        else:
            all_logs = {}
        
        if st.session_state.user_email not in all_logs:
            all_logs[st.session_state.user_email] = []
        
        existing_logs = all_logs[st.session_state.user_email]
        existing_dates = [log.get("date", "") for log in existing_logs]
        
        for log in st.session_state.study_log:
            if log.get("date") not in existing_dates:
                existing_logs.append(log)
        
        all_logs[st.session_state.user_email] = existing_logs[-100:]
        
        with open(STUDY_LOG_FILE, 'w') as f:
            json.dump(all_logs, f, indent=4)
    except Exception as e:
        st.error(f"Error saving study log: {e}")

def load_study_log():
    """Load study log from JSON file"""
    try:
        if os.path.exists(STUDY_LOG_FILE):
            with open(STUDY_LOG_FILE, 'r') as f:
                all_logs = json.load(f)
            
            if st.session_state.user_email in all_logs:
                st.session_state.study_log = all_logs[st.session_state.user_email]
                return True
    except Exception as e:
        st.error(f"Error loading study log: {e}")
    
    st.session_state.study_log = []
    return False

def save_topic_progress():
    """Save topic progress to JSON file"""
    try:
        if os.path.exists(TOPIC_PROGRESS_FILE):
            with open(TOPIC_PROGRESS_FILE, 'r') as f:
                all_topic_progress = json.load(f)
        else:
            all_topic_progress = {}
        
        if st.session_state.user_email not in all_topic_progress:
            all_topic_progress[st.session_state.user_email] = {}
        
        all_topic_progress[st.session_state.user_email] = st.session_state.topic_progress
        
        with open(TOPIC_PROGRESS_FILE, 'w') as f:
            json.dump(all_topic_progress, f, indent=4)
    except Exception as e:
        st.error(f"Error saving topic progress: {e}")

def load_topic_progress():
    """Load topic progress from JSON file"""
    try:
        if os.path.exists(TOPIC_PROGRESS_FILE):
            with open(TOPIC_PROGRESS_FILE, 'r') as f:
                all_topic_progress = json.load(f)
            
            if st.session_state.user_email in all_topic_progress:
                st.session_state.topic_progress = all_topic_progress[st.session_state.user_email]
                return True
    except Exception as e:
        st.error(f"Error loading topic progress: {e}")
    
    st.session_state.topic_progress = {}
    return False

def save_completed_notes():
    """Save completed notes to JSON file"""
    try:
        if os.path.exists(COMPLETED_NOTES_FILE):
            with open(COMPLETED_NOTES_FILE, 'r') as f:
                all_completed_notes = json.load(f)
        else:
            all_completed_notes = {}
        
        if st.session_state.user_email not in all_completed_notes:
            all_completed_notes[st.session_state.user_email] = {}
        
        all_completed_notes[st.session_state.user_email] = st.session_state.completed_notes
        
        with open(COMPLETED_NOTES_FILE, 'w') as f:
            json.dump(all_completed_notes, f, indent=4)
    except Exception as e:
        st.error(f"Error saving completed notes: {e}")

def load_completed_notes():
    """Load completed notes from JSON file"""
    try:
        if os.path.exists(COMPLETED_NOTES_FILE):
            with open(COMPLETED_NOTES_FILE, 'r') as f:
                all_completed_notes = json.load(f)
            
            if st.session_state.user_email in all_completed_notes:
                st.session_state.completed_notes = all_completed_notes[st.session_state.user_email]
                return True
    except Exception as e:
        st.error(f"Error loading completed notes: {e}")
    
    st.session_state.completed_notes = {}
    return False

def save_chat_history(user_id, chat_history):
    """Save chat history to JSON file"""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r') as f:
                all_chats = json.load(f)
        else:
            all_chats = {}
        
        all_chats[user_id] = chat_history[-50:]
        
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump(all_chats, f, indent=4)
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

def load_chat_history(user_id):
    """Load chat history from JSON file"""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r') as f:
                all_chats = json.load(f)
            
            if user_id in all_chats:
                return all_chats[user_id]
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
    
    return []

def save_all_data():
    """Save all user data to files"""
    save_user_progress()
    save_study_log()
    save_topic_progress()
    save_completed_notes()

def load_all_data():
    """Load all user data from files"""
    load_user_progress()
    load_study_log()
    load_topic_progress()
    load_completed_notes()

# ---------- Helper Functions ----------
def create_demo_data():
    """Create demo dataset for testing"""
    subjects = ["Maths", "Python", "Machine Learning", "Statistics", "Data Science"]
    topics = {
        "Maths": ["Calculus", "Algebra", "Vectors", "Matrices"],
        "Python": ["Functions", "Pandas", "Matplotlib", "Loops"],
        "Machine Learning": ["Linear Regression", "Clustering", "Decision Trees", "KNN"],
        "Statistics": ["Regression", "Probability", "Distributions", "Hypothesis Testing"],
        "Data Science": ["Data Cleaning", "Feature Engineering", "PCA", "EDA"]
    }
    
    data = []
    for i in range(100):
        subject = random.choice(subjects)
        topic = random.choice(topics[subject])
        mood = random.choice(["Tired", "Active", "Stressed", "Normal"])
        difficulty = random.choice(["Easy", "Medium", "Hard"])
        
        data.append({
            "Date": f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "Subject": subject,
            "Topic": topic,
            "Difficulty_Level": difficulty,
            "Mood": mood,
            "Planned_Hours": round(random.uniform(1, 4), 1),
            "Actual_Hours": round(random.uniform(0.5, 4), 1),
            "Study_Status": random.choice(["Completed", "Pending"]),
            "Stress_Level": random.choice(["Low", "Medium", "High"]),
            "Doubt_Asked": random.choice(["Yes", "No"])
        })
    
    df = pd.DataFrame(data)
    return df

@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    try:
        if os.path.exists(DATA_FILE):
            return pd.read_csv(DATA_FILE)
        else:
            return create_demo_data()
    except:
        return create_demo_data()

def load_users():
    """Load user credentials"""
    if os.path.exists(USERS_FILE):
        return pd.read_csv(USERS_FILE)
    else:
        demo_users = pd.DataFrame({
            "email": ["student1@example.com", "student2@example.com"],
            "password": ["password123", "password123"]
        })
        demo_users.to_csv(USERS_FILE, index=False)
        return demo_users

def save_user(email, password):
    """Save new user"""
    df = load_users()
    df.loc[len(df)] = [email, password]
    df.to_csv(USERS_FILE, index=False)

def calculate_streak():
    """Calculate study streak"""
    if st.session_state.study_log:
        today = datetime.now().date()
        completed_dates = []
        
        for log in st.session_state.study_log:
            if log.get("status") == "Completed":
                try:
                    date_str = log.get("date", "").split()[0]
                    log_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    completed_dates.append(log_date)
                except:
                    continue
        
        if completed_dates:
            completed_dates.sort()
            last_date = completed_dates[-1]
            
            if (today - last_date).days == 1:
                st.session_state.streak += 1
            elif (today - last_date).days > 1:
                st.session_state.streak = 1
            elif (today - last_date).days == 0:
                if st.session_state.streak == 0:
                    st.session_state.streak = 1
    else:
        st.session_state.streak = 0

def get_reward_badge(streak):
    """Get reward badge based on streak"""
    if streak >= 10:
        return "🏆 Gold"
    elif streak >= 5:
        return "🥈 Silver"
    elif streak >= 3:
        return "🥉 Bronze"
    else:
        return "No Badge"

def get_mood_based_suggestion(mood):
    """Get study suggestion based on mood"""
    mood_lower = str(mood).lower()
    if "tired" in mood_lower:
        return "Take short breaks, revise easy topics 📖"
    elif "active" in mood_lower:
        return "Try difficult problems, focus on new concepts 💪"
    elif "stressed" in mood_lower:
        return "Practice meditation, study in short intervals ☕"
    else:
        return "Maintain consistency, revise important topics 📚"

def get_subjects_by_mood(mood, df):
    """Filter subjects based on user's mood"""
    try:
        mood_lower = str(mood).lower()
        mood_map = {
            "tired": ["Easy"],
            "active": ["Hard", "Medium"],
            "stressed": ["Easy", "Medium"],
            "normal": ["Easy", "Medium", "Hard"]
        }
        
        difficulty_levels = mood_map.get(mood_lower, ["Easy", "Medium", "Hard"])
        filtered_df = df[df["Difficulty_Level"].isin(difficulty_levels)]
        
        if not filtered_df.empty:
            subjects = filtered_df["Subject"].unique()
            return list(subjects)
    except:
        pass
    
    return ["Maths", "Python", "Machine Learning", "Statistics", "Data Science"]

def get_topics_by_mood_subject(mood, subject, df):
    """Filter topics based on mood and subject"""
    try:
        mood_lower = str(mood).lower()
        mood_map = {
            "tired": ["Easy"],
            "active": ["Hard", "Medium"],
            "stressed": ["Easy", "Medium"],
            "normal": ["Easy", "Medium", "Hard"]
        }
        
        difficulty_levels = mood_map.get(mood_lower, ["Easy", "Medium", "Hard"])
        filtered_df = df[
            (df["Subject"] == subject) & 
            (df["Difficulty_Level"].isin(difficulty_levels))
        ]
        
        if not filtered_df.empty:
            topics = filtered_df["Topic"].unique()
            return list(topics)
    except:
        pass
    
    default_topics = {
        "Maths": ["Calculus", "Algebra", "Vectors", "Matrices"],
        "Python": ["Functions", "Pandas", "Matplotlib", "Loops"],
        "Machine Learning": ["Linear Regression", "Clustering", "Decision Trees", "KNN"],
        "Statistics": ["Regression", "Probability", "Distributions", "Hypothesis Testing"],
        "Data Science": ["Data Cleaning", "Feature Engineering", "PCA", "EDA"]
    }
    return default_topics.get(subject, ["General Topic"])

def get_topic_progress(topic):
    """Get progress for a specific topic"""
    if topic not in st.session_state.topic_progress:
        st.session_state.topic_progress[topic] = {
            "notes_completed": 0,
            "total_notes": len(TOPIC_NOTES.get(topic, {}).get("notes", [])),
            "started": False,
            "completed": False,
            "subject": "",
            "last_studied": None
        }
    return st.session_state.topic_progress[topic]

def mark_note_completed(topic, note_index):
    """Mark a specific note as completed"""
    topic_key = f"{topic}_{note_index}"
    st.session_state.completed_notes[topic_key] = True
    
    if topic not in st.session_state.topic_progress:
        st.session_state.topic_progress[topic] = {
            "notes_completed": 0,
            "total_notes": len(TOPIC_NOTES.get(topic, {}).get("notes", [])),
            "started": True,
            "completed": False,
            "subject": st.session_state.selected_subject,
            "last_studied": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    completed_count = 0
    for i in range(st.session_state.topic_progress[topic]["total_notes"]):
        if f"{topic}_{i}" in st.session_state.completed_notes:
            completed_count += 1
    
    st.session_state.topic_progress[topic]["notes_completed"] = completed_count
    st.session_state.topic_progress[topic]["started"] = True
    st.session_state.topic_progress[topic]["subject"] = st.session_state.selected_subject
    st.session_state.topic_progress[topic]["last_studied"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if completed_count == st.session_state.topic_progress[topic]["total_notes"]:
        st.session_state.topic_progress[topic]["completed"] = True
    
    save_completed_notes()
    save_topic_progress()

def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def start_timer(hours, minutes, seconds):
    """Start the study timer"""
    st.session_state.timer_duration = {"hours": hours, "minutes": minutes, "seconds": seconds}
    st.session_state.total_study_seconds = hours * 3600 + minutes * 60 + seconds
    st.session_state.remaining_time = st.session_state.total_study_seconds
    st.session_state.timer_start_time = datetime.now()
    st.session_state.timer_end_time = st.session_state.timer_start_time + timedelta(seconds=st.session_state.total_study_seconds)
    st.session_state.timer_running = True
    st.session_state.timer_paused = False
    st.session_state.study_started = True

def pause_timer():
    """Pause the timer"""
    if st.session_state.timer_running and not st.session_state.timer_paused:
        st.session_state.timer_paused = True
        st.session_state.timer_paused_at = datetime.now()

def resume_timer():
    """Resume the timer"""
    if st.session_state.timer_running and st.session_state.timer_paused:
        pause_duration = (datetime.now() - st.session_state.timer_paused_at).total_seconds()
        st.session_state.timer_end_time += timedelta(seconds=pause_duration)
        st.session_state.timer_paused = False
        st.session_state.timer_paused_at = None

def stop_timer():
    """Stop the timer"""
    st.session_state.timer_running = False
    st.session_state.timer_paused = False
    st.session_state.study_started = False
    st.session_state.remaining_time = 0

def get_remaining_time():
    """Calculate remaining time"""
    if st.session_state.timer_running and not st.session_state.timer_paused:
        now = datetime.now()
        if now < st.session_state.timer_end_time:
            remaining = (st.session_state.timer_end_time - now).total_seconds()
            st.session_state.remaining_time = max(0, remaining)
        else:
            st.session_state.remaining_time = 0
            st.session_state.timer_running = False
    
    return st.session_state.remaining_time

def solve_math_problem(problem, expression):
    """Solve mathematical problems using sympy"""
    try:
        if problem == "differentiate":
            x = sp.symbols('x')
            expr = sp.sympify(expression)
            derivative = sp.diff(expr, x)
            return f"**Derivative of {expression}:**\n```\nd/dx({expression}) = {derivative}\n```"
        
        elif problem == "integrate":
            x = sp.symbols('x')
            expr = sp.sympify(expression)
            integral = sp.integrate(expr, x)
            return f"**Integral of {expression}:**\n```\n∫({expression}) dx = {integral} + C\n```"
        
        elif problem == "solve_equation":
            x = sp.symbols('x')
            equation = sp.sympify(expression)
            solutions = sp.solve(equation, x)
            return f"**Solutions for {expression} = 0:**\n```\nx = {solutions}\n```"
        
        elif problem == "simplify":
            expr = sp.sympify(expression)
            simplified = sp.simplify(expr)
            return f"**Simplified expression:**\n```\n{expression} = {simplified}\n```"
        
    except Exception as e:
        return f"Error solving problem: {str(e)}"

def get_topic_explanation(subject, topic, subtopic=None):
    """Get detailed explanation of a topic"""
    if subject in SUBJECT_KNOWLEDGE:
        if topic in SUBJECT_KNOWLEDGE[subject]:
            topic_info = SUBJECT_KNOWLEDGE[subject][topic]
            
            if subtopic and subtopic in topic_info:
                subtopic_info = topic_info[subtopic]
                response = f"## **{topic} - {subtopic.replace('_', ' ').title()}**\n\n"
                
                for key, value in subtopic_info.items():
                    if isinstance(value, list):
                        response += f"### **{key.replace('_', ' ').title()}:**\n"
                        for item in value:
                            response += f"• {item}\n"
                        response += "\n"
                    else:
                        response += f"### **{key.replace('_', ' ').title()}:**\n{value}\n\n"
                
                return response
            
            else:
                response = f"## **{topic}**\n\n"
                response += "**Available Subtopics:**\n"
                for subtopic_name in topic_info.keys():
                    response += f"• {subtopic_name.replace('_', ' ').title()}\n"
                response += "\n*Ask about any specific subtopic for detailed explanation.*"
                return response
    
    return f"Sorry, I don't have information about {topic} in {subject}."

# ---------- Page Functions ----------
def login_page():
    """Login page"""
    st.title("🔐 Smart Study Planner - Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.info("Demo Credentials:")
        st.write("📧 Email: student1@example.com")
        st.write("🔒 Password: password123")
        
        email = st.text_input("📧 Email")
        password = st.text_input("🔒 Password", type="password")
        
        if st.button("Sign In", key="login_button"):
            if email and password:
                df = load_users()
                user_exists = ((df["email"] == email) & (df["password"] == password)).any()
                
                if user_exists:
                    st.session_state.user_email = email
                    load_all_data()
                    st.session_state.data_loaded = True
                    st.session_state.page = "mood_selection"
                    st.success("Login successful! ✅")
                    st.rerun()
                else:
                    st.error("Invalid email or password ❌")
        
        st.markdown("---")
        if st.button("Create New Account", key="create_account_button"):
            st.session_state.page = "signup"
            st.rerun()

def signup_page():
    """Signup page"""
    st.title("🆕 Create New Account")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        email = st.text_input("📧 Email")
        password = st.text_input("🔒 Password", type="password")
        confirm_password = st.text_input("🔒 Confirm Password", type="password")
        
        if st.button("Sign Up", key="signup_button"):
            if not email or not password:
                st.error("Please fill all fields")
            elif password != confirm_password:
                st.error("Passwords do not match")
            else:
                try:
                    save_user(email, password)
                    st.success("Account created successfully! ✅")
                    st.session_state.page = "login"
                    st.rerun()
                except:
                    st.error("Error creating account")
        
        if st.button("Back to Login", key="back_login_button"):
            st.session_state.page = "login"
            st.rerun()

def mood_selection_page():
    """Mood selection page"""
    st.title("😊 How Are You Feeling Today?")
    st.write("Select your current mood for personalized study suggestions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("😴 Tired", key="mood_tired", use_container_width=True):
            st.session_state.mood = "Tired"
            st.session_state.continue_progress_mode = False
            save_user_progress()
            st.session_state.page = "subject_selection"
            st.rerun()
    
    with col2:
        if st.button("⚡ Active", key="mood_active", use_container_width=True):
            st.session_state.mood = "Active"
            st.session_state.continue_progress_mode = False
            save_user_progress()
            st.session_state.page = "subject_selection"
            st.rerun()
    
    with col3:
        if st.button("😟 Stressed", key="mood_stressed", use_container_width=True):
            st.session_state.mood = "Stressed"
            st.session_state.continue_progress_mode = False
            save_user_progress()
            st.session_state.page = "subject_selection"
            st.rerun()
    
    st.markdown("---")
    if st.button("😊 Normal / Not Sure", key="mood_normal", use_container_width=True):
        st.session_state.mood = "Normal"
        st.session_state.continue_progress_mode = False
        save_user_progress()
        st.session_state.page = "subject_selection"
        st.rerun()
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Dashboard", key="mood_to_dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()
    with col2:
        if st.button("🚪 Logout", key="mood_logout"):
            save_all_data()
            st.session_state.page = "login"
            st.rerun()

def subject_selection_page():
    """Subject selection page"""
    st.title(f"📚 Select Subject")
    st.write(f"**Your Mood:** {st.session_state.mood}")
    
    if st.session_state.continue_progress_mode:
        st.info("🔄 **Continue Progress Mode** - Select a subject to see topics in progress")
        
        subjects_with_progress = set()
        for topic, progress in st.session_state.topic_progress.items():
            if progress.get("started") and not progress.get("completed"):
                if "subject" in progress and progress["subject"]:
                    subjects_with_progress.add(progress["subject"])
        
        subjects_list = list(subjects_with_progress)
        
        if subjects_list:
            cols = st.columns(2)
            for i, subject in enumerate(subjects_list):
                with cols[i % 2]:
                    topic_count = 0
                    for topic, progress in st.session_state.topic_progress.items():
                        if progress.get("started") and not progress.get("completed"):
                            if "subject" in progress and progress["subject"] == subject:
                                topic_count += 1
                    
                    button_text = f"**{subject}** ({topic_count} in progress)"
                    if st.button(button_text, key=f"continue_subject_{subject}", use_container_width=True):
                        st.session_state.selected_subject = subject
                        save_user_progress()
                        st.session_state.page = "topic_selection"
                        st.rerun()
        else:
            st.warning("No topics in progress. Start a new study session instead.")
            if st.button("🎯 Start New Session", key="no_progress_start_new"):
                st.session_state.continue_progress_mode = False
                st.rerun()
    else:
        df = load_data()
        subjects = get_subjects_by_mood(st.session_state.mood, df)
        
        cols = st.columns(2)
        for i, subject in enumerate(subjects):
            with cols[i % 2]:
                if st.button(f"**{subject}**", key=f"subject_{subject}", use_container_width=True):
                    st.session_state.selected_subject = subject
                    save_user_progress()
                    st.session_state.page = "topic_selection"
                    st.rerun()
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Change Mood", key="subject_change_mood"):
            st.session_state.continue_progress_mode = False
            st.session_state.page = "mood_selection"
            st.rerun()
    with col2:
        if st.button("🏠 Dashboard", key="subject_to_dashboard"):
            st.session_state.continue_progress_mode = False
            st.session_state.page = "dashboard"
            st.rerun()
    with col3:
        if st.button("🚪 Logout", key="subject_logout"):
            save_all_data()
            st.session_state.page = "login"
            st.rerun()

def topic_selection_page():
    """Topic selection page"""
    st.title(f"📖 Select Topic")
    st.write(f"**Subject:** {st.session_state.selected_subject}")
    
    if st.session_state.continue_progress_mode:
        st.write("**Mode:** 🔄 Continue Progress")
    else:
        st.write(f"**Mood:** {st.session_state.mood}")
    
    df = load_data()
    
    if st.session_state.continue_progress_mode:
        topics_in_progress = []
        for topic, progress in st.session_state.topic_progress.items():
            if progress.get("started") and not progress.get("completed"):
                if "subject" in progress and progress["subject"] == st.session_state.selected_subject:
                    topics_in_progress.append(topic)
        
        if topics_in_progress:
            st.info(f"Found {len(topics_in_progress)} topic(s) in progress:")
            
            cols = st.columns(2)
            for i, topic in enumerate(topics_in_progress):
                with cols[i % 2]:
                    progress = get_topic_progress(topic)
                    
                    button_text = f"**{topic}**"
                    if progress["started"]:
                        button_text += f" ({progress['notes_completed']}/{progress['total_notes']})"
                    
                    if st.button(button_text, key=f"continue_topic_{topic}", use_container_width=True):
                        st.session_state.selected_topic = topic
                        save_user_progress()
                        st.session_state.page = "study_session"
                        st.rerun()
        else:
            st.warning(f"No topics in progress for {st.session_state.selected_subject}.")
            if st.button("🎯 Start New Topic", key="no_topic_progress_new"):
                st.session_state.continue_progress_mode = False
                topics = get_topics_by_mood_subject(st.session_state.mood, st.session_state.selected_subject, df)
                st.rerun()
    else:
        topics = get_topics_by_mood_subject(st.session_state.mood, st.session_state.selected_subject, df)
        
        cols = st.columns(2)
        for i, topic in enumerate(topics):
            with cols[i % 2]:
                progress = get_topic_progress(topic)
                
                button_text = f"**{topic}**"
                if progress["started"]:
                    if progress["completed"]:
                        button_text += " ✅"
                    else:
                        button_text += f" ({progress['notes_completed']}/{progress['total_notes']})"
                
                if st.button(button_text, key=f"topic_{topic}", use_container_width=True):
                    st.session_state.selected_topic = topic
                    save_user_progress()
                    st.session_state.page = "study_session"
                    st.rerun()
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📚 Change Subject", key="topic_change_subject"):
            st.session_state.continue_progress_mode = False
            st.session_state.page = "subject_selection"
            st.rerun()
    with col2:
        if st.button("🏠 Dashboard", key="topic_to_dashboard"):
            st.session_state.continue_progress_mode = False
            st.session_state.page = "dashboard"
            st.rerun()
    with col3:
        if st.button("🚪 Logout", key="topic_logout"):
            save_all_data()
            st.session_state.page = "login"
            st.rerun()

def study_session_page():
    """Study session page with notes and progress tracking"""
    st.title(f"🎯 Study Session: {st.session_state.selected_topic}")
    
    if st.session_state.study_started and st.session_state.timer_running:
        remaining = get_remaining_time()
        if remaining > 0:
            timer_container = st.container()
            with timer_container:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    hours_remaining = int(remaining // 3600)
                    minutes_remaining = int((remaining % 3600) // 60)
                    seconds_remaining = int(remaining % 60)
                    
                    progress_percent = 100 - (remaining / st.session_state.total_study_seconds * 100)
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        border-radius: 15px;
                        text-align: center;
                        color: white;
                        margin: 10px 0;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    ">
                        <h3 style="margin: 0; color: white;">⏰ Study Timer</h3>
                        <h1 style="margin: 10px 0; font-size: 3em; color: white;">
                            {hours_remaining:02d}:{minutes_remaining:02d}:{seconds_remaining:02d}
                        </h1>
                        <p style="margin: 0; color: white;">
                            {progress_percent:.1f}% completed • Time remaining
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    timer_col1, timer_col2, timer_col3 = st.columns(3)
                    with timer_col1:
                        if st.button("⏸️ Pause", key="pause_timer", use_container_width=True):
                            pause_timer()
                            st.rerun()
                    with timer_col2:
                        if st.button("▶️ Resume", key="resume_timer", use_container_width=True):
                            resume_timer()
                            st.rerun()
                    with timer_col3:
                        if st.button("⏹️ Stop", key="stop_timer", use_container_width=True):
                            stop_timer()
                            st.rerun()
        else:
            st.success("🎉 Study session completed! Time's up!")
            st.session_state.study_started = False
            st.session_state.timer_running = False
    
    st.markdown("---")
    st.subheader("⏰ Set Study Timer")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write("**Hours**")
        hours = st.number_input("Hours", min_value=0, max_value=10, value=1, step=1, key="study_hours", label_visibility="collapsed")
    
    with col2:
        st.write("**Minutes**")
        minutes = st.number_input("Minutes", min_value=0, max_value=59, value=0, step=5, key="study_minutes", label_visibility="collapsed")
    
    with col3:
        st.write("**Seconds**")
        seconds = st.number_input("Seconds", min_value=0, max_value=59, value=0, step=5, key="study_seconds", label_visibility="collapsed")
    
    with col4:
        st.write("**Break Frequency**")
        break_options = ["Every 30 min", "Every 45 min", "Every 60 min", "Every 90 min"]
        default_index = break_options.index("Every 30 min")
        break_frequency = st.selectbox(
            "Select break frequency:",
            break_options,
            index=default_index,
            key="break_frequency_select",
            label_visibility="collapsed"
        )
    
    total_seconds = hours * 3600 + minutes * 60 + seconds
    total_hours = total_seconds / 3600
    time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    st.info(f"**Total Study Time:** {time_formatted} (Hours:Minutes:Seconds)")
    
    if not st.session_state.study_started:
        if st.button("▶️ Start Study Timer", key="start_timer", type="primary", use_container_width=True):
            start_timer(hours, minutes, seconds)
            st.success(f"Timer started for {time_formatted}")
            st.rerun()
    
    if not st.session_state.continue_progress_mode:
        st.info(f"💡 **Model-based Tip:** {get_mood_based_suggestion(st.session_state.mood)}")
    else:
        st.info("💡 **Continue Progress Mode:** You're continuing a previously started topic.")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Subject:** {st.session_state.selected_subject}")
    with col2:
        if st.session_state.continue_progress_mode:
            st.info("**Mode:** 🔄 Continue Progress")
        else:
            st.info(f"**Mood:** {st.session_state.mood}")
    with col3:
        progress = get_topic_progress(st.session_state.selected_topic)
        if progress["total_notes"] > 0:
            completion_percent = (progress["notes_completed"] / progress["total_notes"]) * 100
            st.metric("Notes Progress", f"{progress['notes_completed']}/{progress['total_notes']}", f"{completion_percent:.1f}%")
    
    st.subheader("📝 Study Notes")
    
    topic_data = TOPIC_NOTES.get(st.session_state.selected_topic, {})
    
    if topic_data:
        notes = topic_data.get("notes", [])
        resources = topic_data.get("resources", [])
        youtube_link = topic_data.get("youtube_link", "")
        
        if youtube_link:
            st.markdown(f"""
            <div style="
                background-color: #f0f2f6;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                border-left: 5px solid #FF0000;
            ">
                <h4 style="margin: 0 0 10px 0; color: #FF0000;">
                    📺 Watch YouTube Tutorial for "{st.session_state.selected_topic}"
                </h4>
                <p style="margin: 0;">
                    <a href="{youtube_link}" target="_blank" style="
                        color: #FF0000;
                        text-decoration: none;
                        font-weight: bold;
                        display: inline-block;
                        padding: 8px 15px;
                        background-color: white;
                        border-radius: 5px;
                        border: 2px solid #FF0000;
                    ">
                        🎥 Click here to watch YouTube Tutorial
                    </a>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("**Key Points to Study:**")
        for i, note in enumerate(notes):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(note)
            with col2:
                note_key = f"{st.session_state.selected_topic}_{i}"
                is_completed = st.checkbox("✓", 
                                          value=note_key in st.session_state.completed_notes,
                                          key=f"note_checkbox_{note_key}")
                
                if is_completed and note_key not in st.session_state.completed_notes:
                    mark_note_completed(st.session_state.selected_topic, i)
                    st.rerun()
        
        st.markdown("---")
        st.write("**📚 Recommended Resources:**")
        
        resource_container = st.container()
        with resource_container:
            for resource in resources:
                if "http" in resource or "https" in resource:
                    if "[" in resource and "]" in resource:
                        import re
                        link_pattern = r'\[(.*?)\]\((.*?)\)'
                        match = re.search(link_pattern, resource)
                        if match:
                            link_text = match.group(1)
                            link_url = match.group(2)
                            st.markdown(f"- **📝 {resource.split('[')[0]}** [{link_text}]({link_url})")
                        else:
                            st.write(f"- {resource}")
                    else:
                        st.write(f"- {resource}")
                else:
                    st.write(f"- {resource}")
        
    else:
        st.warning("No notes available for this topic.")
        notes = []
    
    st.markdown("---")
    st.subheader("📊 Actions")
    
    progress = get_topic_progress(st.session_state.selected_topic)
    topic_completed = progress["completed"] or (progress["total_notes"] > 0 and progress["notes_completed"] == progress["total_notes"])
    
    if topic_completed:
        st.success("✅ All notes completed for this topic!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎉 Mark Topic as Completed", key="mark_completed", type="primary", use_container_width=True):
                study_log_entry = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "subject": st.session_state.selected_subject,
                    "topic": st.session_state.selected_topic,
                    "mood": st.session_state.mood,
                    "planned_hours": total_hours,
                    "actual_hours": total_hours,
                    "status": "Completed",
                    "notes_completed": progress["notes_completed"],
                    "total_notes": progress["total_notes"],
                    "break_frequency": break_frequency,
                    "study_time_formatted": time_formatted,
                    "youtube_watched": youtube_link if 'youtube_link' in locals() else ""
                }
                st.session_state.study_log.append(study_log_entry)
                
                calculate_streak()
                
                save_study_log()
                save_user_progress()
                
                st.success(f"🎉 Topic completed! Current streak: {st.session_state.streak} days")
                
                badge = get_reward_badge(st.session_state.streak)
                if badge != "No Badge":
                    st.balloons()
                    st.success(f"🏅 You earned a {badge} badge!")
                
                st.session_state.selected_topic = ""
                st.session_state.study_started = False
                st.session_state.timer_running = False
                st.session_state.continue_progress_mode = False
                st.session_state.page = "subject_selection"
                st.rerun()
        
        with col2:
            if st.button("📚 Study Another Topic", key="another_topic", use_container_width=True):
                st.session_state.selected_topic = ""
                st.session_state.study_started = False
                st.session_state.timer_running = False
                st.session_state.continue_progress_mode = False
                save_user_progress()
                st.session_state.page = "topic_selection"
                st.rerun()
    
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Progress", key="save_progress", use_container_width=True):
                actual_hours = total_hours * 0.5 if st.session_state.study_started else total_hours * 0.2
                study_log_entry = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "subject": st.session_state.selected_subject,
                    "topic": st.session_state.selected_topic,
                    "mood": st.session_state.mood,
                    "planned_hours": total_hours,
                    "actual_hours": actual_hours,
                    "status": "In Progress",
                    "notes_completed": progress["notes_completed"],
                    "total_notes": progress["total_notes"],
                    "break_frequency": break_frequency,
                    "study_time_formatted": time_formatted,
                    "youtube_watched": youtube_link if 'youtube_link' in locals() else ""
                }
                st.session_state.study_log.append(study_log_entry)
                
                save_study_log()
                save_user_progress()
                save_topic_progress()
                save_completed_notes()
                
                st.success("✅ Progress saved successfully! You can continue later.")
                st.session_state.study_started = False
                st.session_state.timer_running = False
                st.session_state.continue_progress_mode = False
                st.session_state.page = "dashboard"
                st.rerun()
        
        with col2:
            if st.button("⏸️ Take a Break", key="take_break", use_container_width=True):
                st.info(f"Taking a break as per: {break_frequency}. Relax! ☕")
    
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📖 Change Topic", key="session_change_topic"):
            save_all_data()
            st.session_state.study_started = False
            st.session_state.timer_running = False
            st.session_state.page = "topic_selection"
            st.rerun()
    
    with col2:
        if st.button("📚 Change Subject", key="session_change_subject"):
            save_all_data()
            st.session_state.study_started = False
            st.session_state.timer_running = False
            st.session_state.continue_progress_mode = False
            st.session_state.page = "subject_selection"
            st.rerun()
    
    with col3:
        if st.button("😊 Change Mood", key="session_change_mood"):
            save_all_data()
            st.session_state.study_started = False
            st.session_state.timer_running = False
            st.session_state.continue_progress_mode = False
            st.session_state.page = "mood_selection"
            st.rerun()
    
    with col4:
        if st.button("🏠 Dashboard", key="session_to_dashboard"):
            save_all_data()
            st.session_state.study_started = False
            st.session_state.timer_running = False
            st.session_state.continue_progress_mode = False
            st.session_state.page = "dashboard"
            st.rerun()
    
    if st.button("🚪 Logout", key="session_logout"):
        save_all_data()
        st.session_state.study_started = False
        st.session_state.timer_running = False
        st.session_state.continue_progress_mode = False
        st.session_state.page = "login"
        st.rerun()
    
    if st.session_state.timer_running and not st.session_state.timer_paused:
        time.sleep(0.1)
        st.rerun()

def dashboard_page():
    """Dashboard page"""
    st.title("📊 Study Dashboard")
    username = st.session_state.user_email.split('@')[0] if '@' in st.session_state.user_email else st.session_state.user_email
    st.write(f"**Welcome, {username}!**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sessions = len(st.session_state.study_log)
        st.metric("Total Sessions", total_sessions)
    
    with col2:
        completed = len([log for log in st.session_state.study_log if log.get("status") == "Completed"])
        st.metric("Completed", completed)
    
    with col3:
        st.metric("Current Streak", f"{st.session_state.streak} days")
    
    with col4:
        badge = get_reward_badge(st.session_state.streak)
        st.metric("Reward Badge", badge.split()[-1] if badge != "No Badge" else "None")
    
    st.subheader("📈 Topics In Progress")
    in_progress_topics = []
    for topic, progress in st.session_state.topic_progress.items():
        if progress.get("started") and not progress.get("completed"):
            progress_percent = (progress["notes_completed"] / progress["total_notes"]) * 100 if progress["total_notes"] > 0 else 0
            subject = progress.get("subject", "Unknown")
            last_studied = progress.get("last_studied", "Never")
            in_progress_topics.append({
                "Topic": topic,
                "Subject": subject,
                "Progress": f"{progress['notes_completed']}/{progress['total_notes']}",
                "Completion": f"{progress_percent:.1f}%",
                "Last Studied": last_studied
            })
    
    if in_progress_topics:
        df_in_progress = pd.DataFrame(in_progress_topics)
        st.dataframe(df_in_progress, use_container_width=True, hide_index=True)
    else:
        st.info("No topics in progress yet. Start studying!")
    
    st.subheader("📊 All Topic Progress")
    if st.session_state.topic_progress:
        progress_data = []
        for topic, progress in st.session_state.topic_progress.items():
            if progress["started"]:
                progress_percent = (progress["notes_completed"] / progress["total_notes"]) * 100 if progress["total_notes"] > 0 else 0
                subject = progress.get("subject", "Unknown")
                progress_data.append({
                    "Topic": topic,
                    "Subject": subject,
                    "Completed Notes": f"{progress['notes_completed']}/{progress['total_notes']}",
                    "Progress": f"{progress_percent:.1f}%",
                    "Status": "✅ Completed" if progress["completed"] else "📚 In Progress"
                })
        
        if progress_data:
            df_progress = pd.DataFrame(progress_data)
            st.dataframe(df_progress, use_container_width=True, hide_index=True)
        else:
            st.info("No topics studied yet. Start your first session!")
    else:
        st.info("No topics studied yet. Start your first session!")
    
    st.subheader("📝 Recent Study Sessions")
    if st.session_state.study_log:
        sorted_logs = sorted(st.session_state.study_log, 
                            key=lambda x: x.get('date', ''), 
                            reverse=True)
        recent_logs = sorted_logs[:10]
        df_logs = pd.DataFrame(recent_logs)
        
        display_cols = ["date", "subject", "topic", "status", "notes_completed", "total_notes"]
        available_cols = [col for col in display_cols if col in df_logs.columns]
        
        if available_cols:
            st.dataframe(df_logs[available_cols], use_container_width=True)
    else:
        st.info("No study sessions yet. Start your first session!")
    
    st.subheader("⚡ Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🎯 Start New Session", key="dashboard_start_session", type="primary", use_container_width=True):
            st.session_state.page = "mood_selection"
            st.rerun()
    
    with col2:
        if st.button("🔄 Continue Progress", key="dashboard_continue", use_container_width=True):
            in_progress_topics_list = []
            for topic, progress in st.session_state.topic_progress.items():
                if progress.get("started") and not progress.get("completed"):
                    in_progress_topics_list.append(topic)
            
            if in_progress_topics_list:
                st.session_state.continue_progress_mode = True
                if len(in_progress_topics_list) == 1:
                    topic = in_progress_topics_list[0]
                    st.session_state.selected_topic = topic
                    if "subject" in st.session_state.topic_progress[topic]:
                        st.session_state.selected_subject = st.session_state.topic_progress[topic]["subject"]
                    else:
                        df = load_data()
                        topic_rows = df[df["Topic"] == topic]
                        if not topic_rows.empty:
                            st.session_state.selected_subject = topic_rows.iloc[0]["Subject"]
                        else:
                            st.session_state.selected_subject = "Unknown"
                    st.session_state.page = "study_session"
                    st.rerun()
                else:
                    st.session_state.page = "subject_selection"
                    st.rerun()
            else:
                st.info("No topics in progress. Start a new session!")
    
    with col3:
        if st.button("🎓 Doubt Solver", key="goto_chatbot", use_container_width=True):
            st.session_state.page = "chatbot"
            st.rerun()
    
    with col4:
        if st.button("📥 Export Data", key="dashboard_export", use_container_width=True):
            if st.session_state.study_log:
                df_export = pd.DataFrame(st.session_state.study_log)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="study_log.csv",
                    mime="text/csv",
                    key="download_csv"
                )
    
    st.markdown("---")
    if st.button("🚪 Logout", key="dashboard_logout", use_container_width=True):
        save_all_data()
        st.session_state.page = "login"
        st.session_state.user_email = ""
        st.rerun()

def chatbot_page():
    """Chatbot page for doubt solving"""
    st.title("🎓 Subject Doubt Solver")
    st.markdown("Ask questions about Mathematics, Python, Machine Learning, and more!")
    
    if 'subject_chat_history' not in st.session_state:
        user_id = st.session_state.get('user_email', 'anonymous')
        st.session_state.subject_chat_history = load_chat_history(user_id)
    
    with st.sidebar:
        st.header("📚 Subjects")
        
        selected_subject = st.selectbox(
            "Choose Subject:",
            ["Mathematics", "Python", "Machine Learning", "All Subjects"]
        )
        
        st.markdown("---")
        st.header("🧮 Math Tools")
        
        math_tool = st.selectbox(
            "Math Problem Solver:",
            ["Differentiate", "Integrate", "Solve Equation"]
        )
        
        math_expression = st.text_input("Enter expression (use x as variable):", 
                                       value="x^2 + 3*x + 2")
        
        if st.button("Solve Math Problem"):
            if math_expression:
                user_msg = f"Math Problem ({math_tool}): {math_expression}"
                st.session_state.subject_chat_history.append({
                    "role": "user",
                    "content": user_msg,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                problem_map = {
                    "Differentiate": "differentiate",
                    "Integrate": "integrate",
                    "Solve Equation": "solve_equation"
                }
                
                solution = solve_math_problem(problem_map[math_tool], math_expression)
                
                st.session_state.subject_chat_history.append({
                    "role": "assistant",
                    "content": solution,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                user_id = st.session_state.get('user_email', 'anonymous')
                save_chat_history(user_id, st.session_state.subject_chat_history)
                st.rerun()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        for message in st.session_state.subject_chat_history[-20:]:
            if message["role"] == "user":
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <div style="color: #1565c0; font-weight: bold;">👤 You ({message['timestamp']})</div>
                    <div>{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f1f8e9; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <div style="color: #558b2f; font-weight: bold;">🤖 Study Assistant ({message['timestamp']})</div>
                    <div>{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📖 Topics Guide")
        
        if selected_subject in SUBJECT_KNOWLEDGE:
            for topic in SUBJECT_KNOWLEDGE[selected_subject].keys():
                if st.button(f"📌 {topic}", key=f"topic_{topic}", use_container_width=True):
                    explanation = get_topic_explanation(selected_subject, topic)
                    
                    st.session_state.subject_chat_history.append({
                        "role": "user",
                        "content": f"Tell me about {topic} in {selected_subject}",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    st.session_state.subject_chat_history.append({
                        "role": "assistant",
                        "content": explanation,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    user_id = st.session_state.get('user_email', 'anonymous')
                    save_chat_history(user_id, st.session_state.subject_chat_history)
                    st.rerun()
    
    st.markdown("---")
    user_input = st.text_input("Ask your question:", 
                              placeholder="E.g., How to solve quadratic equations? Explain derivatives...")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        if st.button("Send Question", type="primary", use_container_width=True):
            if user_input.strip():
                st.session_state.subject_chat_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                if "derivative" in user_input.lower():
                    response = get_topic_explanation("Mathematics", "Calculus", "differentiation")
                elif "integral" in user_input.lower():
                    response = get_topic_explanation("Mathematics", "Calculus", "integration")
                elif "python" in user_input.lower() or "function" in user_input.lower():
                    response = get_topic_explanation("Python", "Functions")
                elif "regression" in user_input.lower():
                    response = get_topic_explanation("Machine Learning", "Linear Regression")
                else:
                    response = "I understand you're asking about study topics. Could you please specify which subject and topic you need help with?"
                
                st.session_state.subject_chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                user_id = st.session_state.get('user_email', 'anonymous')
                save_chat_history(user_id, st.session_state.subject_chat_history)
                st.rerun()
    
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.subject_chat_history = []
            user_id = st.session_state.get('user_email', 'anonymous')
            save_chat_history(user_id, [])
            st.rerun()

def render_sidebar():
    """Render the sidebar"""
    with st.sidebar:
        st.title("📘 Smart Study Planner")
        st.markdown("---")
        
        if st.session_state.page not in ["login", "signup"]:
            if st.session_state.mood:
                st.write(f"**Mood:** {st.session_state.mood}")
            
            if st.session_state.selected_subject:
                st.write(f"**Subject:** {st.session_state.selected_subject}")
            
            if st.session_state.selected_topic:
                st.write(f"**Topic:** {st.session_state.selected_topic}")
            
            if st.session_state.streak > 0:
                st.write(f"**Streak:** {st.session_state.streak} days")
            
            st.markdown("---")
            
            if st.button("🏠 Dashboard", key="sidebar_dashboard", use_container_width=True):
                st.session_state.page = "dashboard"
                st.rerun()
            
            if st.button("🎯 Study Now", key="sidebar_study", use_container_width=True):
                if st.session_state.mood:
                    st.session_state.page = "subject_selection"
                else:
                    st.session_state.page = "mood_selection"
                st.rerun()
            
            if st.button("🎓 Doubt Solver", key="sidebar_chatbot", use_container_width=True):
                st.session_state.page = "chatbot"
                st.rerun()
            
            if st.button("💾 Save Progress", key="sidebar_save", use_container_width=True):
                save_all_data()
                st.success("Progress saved!")
            
            st.markdown("---")
            if st.button("🚪 Logout", key="sidebar_logout", use_container_width=True):
                save_all_data()
                st.session_state.page = "login"
                st.rerun()
        
        st.markdown("---")
        st.caption("Smart Study Planner v1.0")

# ---------- Main App ----------
def main():
    """Main application controller"""
    
    if "page" not in st.session_state:
        st.session_state.page = "login"
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""
    if "mood" not in st.session_state:
        st.session_state.mood = ""
    if "selected_subject" not in st.session_state:
        st.session_state.selected_subject = ""
    if "selected_topic" not in st.session_state:
        st.session_state.selected_topic = ""
    if "study_log" not in st.session_state:
        st.session_state.study_log = []
    if "streak" not in st.session_state:
        st.session_state.streak = 0
    if "topic_progress" not in st.session_state:
        st.session_state.topic_progress = {}
    if "completed_notes" not in st.session_state:
        st.session_state.completed_notes = {}
    if "timer_start_time" not in st.session_state:
        st.session_state.timer_start_time = None
    if "timer_end_time" not in st.session_state:
        st.session_state.timer_end_time = None
    if "timer_running" not in st.session_state:
        st.session_state.timer_running = False
    if "timer_paused" not in st.session_state:
        st.session_state.timer_paused = False
    if "timer_paused_at" not in st.session_state:
        st.session_state.timer_paused_at = None
    if "timer_duration" not in st.session_state:
        st.session_state.timer_duration = {"hours": 1, "minutes": 0, "seconds": 0}
    if "remaining_time" not in st.session_state:
        st.session_state.remaining_time = 0
    if "study_started" not in st.session_state:
        st.session_state.study_started = False
    if "total_study_seconds" not in st.session_state:
        st.session_state.total_study_seconds = 0
    if "continue_progress_mode" not in st.session_state:
        st.session_state.continue_progress_mode = False
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "subject_chat_history" not in st.session_state:
        st.session_state.subject_chat_history = []
    
    render_sidebar()
    
    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "signup":
        signup_page()
    elif st.session_state.page == "mood_selection":
        mood_selection_page()
    elif st.session_state.page == "subject_selection":
        subject_selection_page()
    elif st.session_state.page == "topic_selection":
        topic_selection_page()
    elif st.session_state.page == "study_session":
        study_session_page()
    elif st.session_state.page == "dashboard":
        dashboard_page()
    elif st.session_state.page == "chatbot":
        chatbot_page()

# ---------- Run App ----------
if __name__ == "__main__":
    main()