# Scalable Movie Recommendation System Using PySpark

A scalable, personalised movie recommendation system built using **Apache Spark, PySpark, HDFS, and the Alternating Least Squares (ALS) algorithm**. This project explores distributed collaborative filtering, model optimisation, and large-scale recommendation using the MovieLens dataset, including MovieLens 20M.

## Project Overview

This project implements a distributed movie recommendation system that learns from historical user ratings to predict which movies users are likely to enjoy.

Using Apache Spark's MLlib implementation of Alternating Least Squares (ALS), the system learns relationships between users and movies to generate personalised recommendations.

The project initially evaluates different recommendation models and ALS configurations on a smaller dataset, then extends the experiments to MovieLens 20M to investigate scalability when processing approximately 20 million ratings.

## Objectives

- Build a collaborative filtering recommendation system using ALS.
- Process large-scale user–movie interactions using Apache Spark.
- Load and manage datasets through Hadoop Distributed File System (HDFS).
- Compare personalised recommendations against a non-personalised baseline.
- Experiment with ALS hyperparameters to improve prediction accuracy.
- Generate personalised top-5 movie recommendations for individual users.
- Evaluate model performance using RMSE and Precision@K.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Implementation and data processing |
| Apache Spark | Distributed data processing |
| PySpark | Python API for Apache Spark |
| Spark MLlib | Collaborative filtering and ALS |
| HDFS | Distributed dataset storage |
| MovieLens 20M | Large-scale movie ratings dataset |
| Pandas & Matplotlib | Data analysis and visualisation |

## Dataset

The project uses the MovieLens dataset, which contains user ratings and movie metadata.

The larger-scale experiment uses **MovieLens 20M**, containing approximately:

- 20 million user–movie ratings.
- User identifiers.
- Movie identifiers.
- Explicit movie ratings.
- Movie titles and genres.

The dataset is stored in HDFS and accessed through Spark DataFrames, allowing distributed processing without loading the entire dataset into local memory.

Dataset: [MovieLens 20M – GroupLens Research](https://grouplens.org/datasets/movielens/20m/)

## Project Workflow

### 1. Data Loading and Exploration

- Loaded ratings and movie metadata from HDFS into Spark DataFrames.
- Inspected data schemas, record counts, and missing values.
- Analysed rating distributions, user activity, and movie popularity.
- Investigated dataset sparsity and long-tail distributions.

### 2. Baseline Recommendation Model

Implemented a simple non-personalised recommendation approach using average movie ratings.

This model predicts ratings based on each movie's historical average, providing a benchmark against which the personalised ALS model can be evaluated.

### 3. Collaborative Filtering with ALS

Implemented the Alternating Least Squares algorithm using Spark MLlib.

ALS learns latent representations of users and movies from their historical interactions, allowing the system to predict ratings for individual user–movie combinations.

The model was integrated into a Spark ML Pipeline and evaluated on held-out data.

### 4. Hyperparameter Tuning

Experimented with different ALS configurations to investigate their impact on prediction accuracy.

The parameters explored included:

- **Rank:** Number of latent factors.
- **Regularisation (regParam):** Controls model complexity and helps prevent overfitting.
- **Maximum iterations (maxIter):** Number of optimisation iterations.

The best-performing configuration on the smaller dataset was:

| Parameter | Value |
|---|---:|
| Rank | 10 |
| Regularisation | 0.2 |
| Maximum iterations | 15 |

### 5. Personalised Movie Recommendations

Used the trained ALS model to generate recommendations for individual users.

The recommendation pipeline:

1. Generates candidate user–movie combinations.
2. Removes movies already rated in the training data.
3. Predicts ratings for unseen movies.
4. Ranks movies by predicted rating.
5. Returns the top 5 recommendations for each user.

Movie metadata is joined to the results to display readable movie titles.

### 6. Recommendation Evaluation

Evaluated the system using two complementary metrics:

- **RMSE (Root Mean Squared Error):** Measures how closely predicted ratings match actual user ratings. Lower values indicate better rating prediction accuracy.
- **Precision@5:** Measures the proportion of the five recommended movies that are relevant to the user, where relevance is defined as a rating of 4.0 or higher in the test set.

This provides insight into both rating prediction accuracy and the quality of the generated recommendations.

### 7. Scaling to MovieLens 20M

Extended the implementation to MovieLens 20M using HDFS and Apache Spark.

The ALS approach was applied to approximately 20 million ratings to demonstrate how distributed processing can support larger recommendation workloads.

## Results

### Model Performance Comparison

The following results were obtained on the smaller dataset:

| Model | RMSE |
|---|---:|
| Movie Average Baseline | 0.9764 |
| ALS – Initial Configuration | 0.8870 |
| ALS – Tuned Configuration | **0.8807** |

The tuned ALS model reduced prediction error by approximately **9.8% compared with the baseline model**.

This demonstrates the benefit of personalised collaborative filtering over a simple movie-average prediction approach.

### Large-Scale Experiment

| Experiment | RMSE |
|---|---:|
| Tuned ALS – Smaller Dataset | 0.8807 |
| ALS – MovieLens 20M | **~0.846** |

The large-scale experiment achieved a lower RMSE, demonstrating the potential benefits of learning from a larger volume of user–movie interactions.

*Note: The MovieLens 20M evaluation in the original notebook should be rerun with the model trained on the training split only, before treating this figure as a held-out test result.*

### Key Findings

- Personalised collaborative filtering outperformed the average-rating baseline.
- Regularisation had a stronger impact on performance than increasing the number of latent factors.
- A moderate regularisation value of 0.2 provided the best results in the tuning experiments.
- The project demonstrated how Apache Spark and HDFS can support recommendation workloads at larger data volumes.
- Rating prediction accuracy and recommendation ranking quality measure different aspects of recommender system performance.

## Repository Structure

```text
Scalable-Movie-Recommendation-System-Using-PySpark/
│
├── Project.ipynb
├── Project.html
├── README.md
```

## Running the Project

### Prerequisites

- Python 3.x
- Apache Spark
- PySpark
- Access to an HDFS environment
- Jupyter Notebook

### Setup

1. Clone the repository:

```bash
git clone https://github.com/nour-abuhassira/Scalable-Movie-Recommendation-System-Using-PySpark.git
```
2. Install the required Python packages:

```bash
pip install pyspark pandas matplotlib jupyter
```

3. Download the [MovieLens 20M dataset](https://grouplens.org/datasets/movielens/20m/).

4. Upload the dataset files to your HDFS environment.

5. Update the HDFS paths in the notebook to match your environment.

6. Launch Jupyter Notebook and execute the notebook cells in order.

**Note:** The notebook was developed in a Spark cluster environment. Running it requires a configured Spark session and access to the relevant HDFS paths.

## Limitations and Future Improvements

Although the ALS model provides effective personalised rating predictions, several limitations remain:

- **Data sparsity:** Many user–movie combinations have no recorded interactions, limiting the information available for learning preferences.
- **Cold-start problem:** New users and movies with insufficient interaction history are difficult to recommend accurately.
- **Limited contextual information:** The current model relies on historical user ratings without incorporating movie content or user context.
- **Recommendation ranking:** The relatively low Precision@5 highlights the difficulty of identifying relevant unseen movies, even when rating prediction performance is reasonable.

Potential future improvements include hybrid recommendation approaches combining collaborative and content-based filtering, more advanced ranking techniques, and additional evaluation metrics.

## Author

**Nour AbuHassira**

MSc Data Science & Artificial Intelligence  
Goldsmiths, University of London

