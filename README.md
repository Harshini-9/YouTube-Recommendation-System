# YouTube-Recommendation-System

## Table of Contents
1. [Overview](#overview)
2. [Key Concepts and Technologies](#key-concepts)
3. [Dataset Generation](#dataset-generation)
4. [Recommendation Algorithms](#recommendation-algorithms)
5. [Code Implementation Breakdown](#code-breakdown)
6. [Mathematical Foundations](#mathematical-foundations)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Visualization and Analysis](#visualization)

---

## Overview {#Overview}

This is a comprehensive implementation of a **YouTube Video Recommendation System** that demonstrates multiple recommendation approaches using synthetic data. The system combines collaborative filtering, content-based filtering, and matrix factorization techniques to provide personalized video recommendations.

### Main Components:
- **Synthetic Dataset Generation**: Creates realistic user, video, and interaction data
- **Multiple Recommendation Algorithms**: Implements 4 different recommendation approaches
- **Evaluation Framework**: Measures system performance using standard metrics
- **Data Visualization**: Provides insights into data patterns and system behavior

---

## Key Concepts and Technologies {#key-concepts}

### 1. **Machine Learning Libraries**
```python
import pandas as pd              # Data manipulation and analysis
import numpy as np               # Numerical computing
import matplotlib.pyplot as plt  # Static plotting
import seaborn as sns           # Statistical data visualization
from sklearn.feature_extraction.text import TfidfVectorizer  # Text vectorization
from sklearn.metrics.pairwise import cosine_similarity       # Similarity metrics
from sklearn.decomposition import TruncatedSVD              # Matrix factorization
from sklearn.preprocessing import StandardScaler            # Feature scaling
```

### 2. **Core Recommendation System Concepts**

#### **Collaborative Filtering (CF)**
- **User-Based CF**: "Users who liked similar videos will like similar videos in the future"
- **Item-Based CF**: "Videos similar to what you've liked before"
- **Memory-Based**: Uses user-item interaction history directly

#### **Content-Based Filtering**
- Uses video metadata (title, category, description, tags)
- Creates user profiles based on content preferences
- Recommends videos with similar content features

#### **Matrix Factorization**
- Decomposes user-item interaction matrix into lower-dimensional matrices
- Uses Singular Value Decomposition (SVD)
- Captures latent factors in user preferences and video characteristics

#### **Hybrid Systems**
- Combines multiple recommendation approaches
- Weighted combination of different methods
- Leverages strengths of each individual approach

---

## Dataset Generation {#dataset-generation}

### YouTubeDatasetGenerator Class

This class creates realistic synthetic data that mimics real YouTube platform data:

#### **Video Data Generation**
```python
def generate_videos(self):
    """Generate synthetic video metadata"""
```

**Features Created:**
- `video_id`: Unique identifier
- `title`: Video title
- `category`: One of 10 categories (Technology, Entertainment, Music, etc.)
- `duration`: Normal distribution around 10 minutes
- `views`: Log-normal distribution (realistic for viral content)
- `likes`: Log-normal distribution correlated with views
- `upload_date`: Distributed across 5 years
- `language`: Weighted distribution (60% English)
- `description`: Category-based description
- `tags`: Random combination of relevant tags

**Mathematical Distributions Used:**
- **Normal Distribution**: `np.random.normal(600, 300)` for duration
- **Log-Normal Distribution**: `np.random.lognormal(10, 2)` for views/likes
- **Categorical Distribution**: Weighted choices for languages

#### **User Data Generation**
```python
def generate_users(self):
    """Generate synthetic user profiles"""
```

**User Attributes:**
- `user_id`: Unique identifier
- `age`: Random integer 13-70
- `gender`: Categorical with realistic distribution
- `location`: 8 different countries
- `signup_date`: Distributed over 6 years
- `preferred_categories`: Multiple category preferences

#### **Interaction Data Generation**
```python
def generate_interactions(self, users_df, videos_df):
    """Generate user-video interaction data"""
```

**Interaction Features:**
- `rating`: 1-5 scale with realistic distribution
- `watch_percentage`: Beta distribution (realistic viewing patterns)
- `engagement`: Binary features (liked, commented, shared)
- `timestamp`: Recent interactions weighted higher

**Sparsity Control:**
- Default 95% sparsity (realistic for recommendation systems)
- Only 5% of possible user-video pairs have interactions

---

## Recommendation Algorithms {#recommendation-algorithms}

### 1. **User-Based Collaborative Filtering**

```python
def collaborative_filtering_user_based(self, user_id, n_recommendations=10):
```

**Algorithm Steps:**
1. **Find Similar Users**: Calculate cosine similarity between target user and all other users
2. **Select Top Similar Users**: Choose top 20 most similar users
3. **Aggregate Recommendations**: For each unrated video, collect ratings from similar users
4. **Weight by Similarity**: Weight each rating by user similarity score
5. **Rank and Return**: Sort by weighted average score

**Mathematical Formula:**
```
similarity(u1, u2) = cos(θ) = (u1 · u2) / (||u1|| × ||u2||)

predicted_rating(u, i) = Σ(similarity(u, v) × rating(v, i)) / Σ(similarity(u, v))
```

### 2. **Item-Based Collaborative Filtering**

```python
def collaborative_filtering_item_based(self, user_id, n_recommendations=10):
```

**Algorithm Steps:**
1. **Calculate Item Similarity**: Compute cosine similarity between all video pairs
2. **For Each Unrated Video**: Find similar videos the user has rated
3. **Predict Rating**: Weight similar videos' ratings by similarity scores
4. **Rank Predictions**: Sort by predicted rating

**Key Advantage**: More stable than user-based (items change less than user preferences)

### 3. **Matrix Factorization (SVD)**

```python
def matrix_factorization_svd(self, n_components=50, n_recommendations=10):
```

**Concept**: Decomposes the user-item matrix into two lower-dimensional matrices:
- **User Matrix**: Users × Latent Factors
- **Item Matrix**: Videos × Latent Factors

**Mathematical Representation:**
```
R ≈ U × V^T
where:
- R: User-Item rating matrix (m × n)
- U: User factor matrix (m × k)
- V: Item factor matrix (n × k)
- k: Number of latent factors (50 in this case)
```

**Benefits:**
- Handles sparsity better than memory-based methods
- Captures latent factors (genres, moods, etc.)
- Scalable to large datasets

### 4. **Content-Based Filtering**

```python
def content_based_filtering(self, user_id, n_recommendations=10):
```

**Algorithm Steps:**
1. **Text Feature Extraction**: Use TF-IDF on combined text features
2. **Numerical Feature Scaling**: Standardize duration, views, likes
3. **User Profile Creation**: Average content features of highly-rated videos
4. **Similarity Calculation**: Cosine similarity between user profile and all videos
5. **Recommendation**: Rank unrated videos by similarity

**Feature Engineering:**
```python
# Combine text features
content_text = title + category + description + tags

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
text_features = tfidf.fit_transform(content_text)

# Numerical feature scaling
scaler = StandardScaler()
numerical_features = scaler.fit_transform([duration, views, likes])

# Combined feature vector
content_features = [text_features, numerical_features]
```

### 5. **Hybrid Recommendation System**

```python
def hybrid_recommendations(self, user_id, weights={'cf': 0.4, 'cb': 0.3, 'svd': 0.3}):
```

**Approach**: Linear combination of multiple methods
```
hybrid_score(u, i) = w1 × cf_score(u, i) + w2 × cb_score(u, i) + w3 × svd_score(u, i)
```

**Benefits:**
- Combines strengths of different approaches
- Reduces individual method limitations
- More robust recommendations

---

## Code Implementation Breakdown {#code-breakdown}

### Class Structure

#### **YouTubeDatasetGenerator**
- **Purpose**: Create synthetic but realistic dataset
- **Key Methods**:
  - `generate_videos()`: Creates video metadata
  - `generate_users()`: Creates user profiles  
  - `generate_interactions()`: Creates user-video interactions
- **Parameters**: Control dataset size and sparsity

#### **YouTubeRecommenderSystem**
- **Purpose**: Main recommendation engine
- **Key Attributes**:
  - `user_item_matrix`: Pivot table of user ratings
  - `content_features`: TF-IDF + numerical features
  - `svd_model`: Trained matrix factorization model

### Data Structures

#### **User-Item Matrix**
```python
self.user_item_matrix = interactions_df.pivot_table(
    index='user_id', 
    columns='video_id', 
    values='rating', 
    fill_value=0
)
```
- **Shape**: (n_users, n_videos)
- **Values**: Ratings (1-5) or 0 for unrated
- **Sparsity**: ~95% zeros (realistic for recommendation systems)

#### **Content Features Matrix**
- **Text Features**: TF-IDF vectors (1000 dimensions)
- **Numerical Features**: Scaled duration, views, likes
- **Combined**: Horizontal stack of text and numerical features

### Key Methods Explained

#### **_create_user_item_matrix()**
```python
def _create_user_item_matrix(self):
    self.user_item_matrix = self.interactions_df.pivot_table(
        index='user_id', columns='video_id', values='rating', fill_value=0
    )
```
- Transforms interaction data into matrix format
- Essential for collaborative filtering algorithms
- Handles missing values by filling with 0

#### **_create_content_features()**
```python
def _create_content_features(self):
    # Text preprocessing
    self.videos_df['content_text'] = (title + category + description + tags)
    
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    content_matrix = tfidf.fit_transform(content_text)
    
    # Feature scaling
    scaler = StandardScaler()
    numerical_matrix = scaler.fit_transform(numerical_features)
    
    # Combine features
    self.content_features = np.hstack([content_matrix.toarray(), numerical_matrix])
```

---

## Mathematical Foundations {#mathematical-foundations}

### Similarity Metrics

#### **Cosine Similarity**
```
cos(θ) = (A · B) / (||A|| × ||B||)

where:
- A, B: Feature vectors
- A · B: Dot product
- ||A||, ||B||: Euclidean norms
```

**Range**: [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite

#### **Euclidean Distance**
```
d(A, B) = √(Σ(Ai - Bi)²)
```

### Matrix Factorization Mathematics

#### **Singular Value Decomposition (SVD)**
```
R = U × Σ × V^T

where:
- U: Left singular vectors (users)
- Σ: Singular values (importance weights)
- V^T: Right singular vectors (items)
```

#### **Truncated SVD**
- Keeps only top k singular values
- Reduces dimensionality while preserving most information
- Handles sparsity by focusing on main patterns

### TF-IDF (Term Frequency - Inverse Document Frequency)

```
TF-IDF(t, d) = TF(t, d) × IDF(t)

where:
TF(t, d) = (Number of times term t appears in document d) / (Total terms in d)
IDF(t) = log(Total documents / Documents containing term t)
```

**Purpose**: Weights terms by importance across the corpus

---

## Evaluation Metrics {#evaluation-metrics}

### Implemented Evaluation

```python
def evaluate_recommendations(self, test_size=0.2):
```

#### **Precision**
```
Precision = |Relevant ∩ Recommended| / |Recommended|
```
- Measures accuracy of recommendations
- "Of all recommended videos, how many were actually relevant?"

#### **Recall**
```
Recall = |Relevant ∩ Recommended| / |Relevant|
```
- Measures completeness of recommendations
- "Of all relevant videos, how many were recommended?"

#### **F1-Score**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Harmonic mean of precision and recall
- Balances both metrics

### Evaluation Process
1. **Train-Test Split**: 80% training, 20% testing
2. **Sample Users**: Evaluate on 50 random users
3. **Ground Truth**: Videos rated ≥4 in test set
4. **Recommendations**: Generate top-10 recommendations
5. **Metrics Calculation**: Compare recommended vs. actual

---

## Visualization and Analysis {#visualization}

### Data Analysis Plots

```python
def plot_data_analysis(self):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
```

#### **Six Key Visualizations:**

1. **Rating Distribution**: Histogram of user ratings (1-5 scale)
2. **Video Categories**: Bar chart of video distribution across categories
3. **Watch Percentage**: Distribution of how much users watch videos
4. **User Age Distribution**: Demographics of user base
5. **Views vs Likes**: Scatter plot showing correlation (log scale)
6. **Interactions Over Time**: Time series of user engagement

### Recommendation Display

```python
def display_recommendations(self, user_id, method='hybrid'):
```

**Features:**
- User profile summary
- Top-rated video history
- Detailed recommendations with scores
- Video metadata display

---

## Advanced Implementation Details

### Memory Management
- **Sparse Matrix Handling**: Uses pandas pivot tables efficiently
- **Feature Engineering**: Combines text and numerical features properly
- **Model Storage**: Saves trained models for reuse

### Error Handling
- **User Validation**: Checks if user exists in dataset
- **Empty Recommendations**: Handles cases with no similar users/items
- **Data Consistency**: Ensures proper data types and ranges

### Scalability Considerations
- **Matrix Factorization**: More scalable than memory-based methods
- **Feature Dimensionality**: Limits TF-IDF to 1000 features
- **Batch Processing**: Evaluates on sample of users

### Hyperparameter Tuning
- **SVD Components**: 50 latent factors (adjustable)
- **Similarity Threshold**: Top-20 similar users/items
- **Hybrid Weights**: Configurable combination weights
- **TF-IDF Parameters**: Max features, stop words, etc.

---

## Real-World Applications

### Production Considerations
1. **Cold Start Problem**: How to recommend to new users/videos
2. **Scalability**: Handling millions of users and videos
3. **Real-time Updates**: Incorporating new interactions
4. **Diversity**: Avoiding filter bubbles
5. **Business Metrics**: Click-through rates, watch time

### Extensions Possible
1. **Deep Learning**: Neural collaborative filtering
2. **Sequential Models**: RNNs for temporal patterns
3. **Multi-objective**: Balancing relevance, diversity, novelty
4. **Contextual Factors**: Time, device, location
5. **Implicit Feedback**: Using views, skips, searches

---

## Summary

This implementation demonstrates a comprehensive recommendation system covering:

- **Data Generation**: Realistic synthetic dataset creation
- **Multiple Algorithms**: Four different recommendation approaches
- **Proper Evaluation**: Standard metrics and validation
- **Visualization**: Data insights and recommendation display
- **Production Ready**: Modular, extensible code structure

The system serves as an excellent foundation for understanding recommendation systems and can be extended for real-world applications.
