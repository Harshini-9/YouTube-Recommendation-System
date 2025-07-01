# YouTube Video Recommendation System
# Complete implementation with synthetic dataset generation and multiple recommendation approaches

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class YouTubeDatasetGenerator:
    """Generate synthetic YouTube video dataset for recommendation system"""
    
    def __init__(self, n_users=1000, n_videos=500, sparsity=0.95):
        self.n_users = n_users
        self.n_videos = n_videos
        self.sparsity = sparsity
        
        # Video categories and attributes
        self.categories = ['Technology', 'Entertainment', 'Music', 'Gaming', 'Education', 
                          'Sports', 'News', 'Comedy', 'Lifestyle', 'Travel']
        self.languages = ['English', 'Spanish', 'French', 'German', 'Japanese', 'Korean']
        
    def generate_videos(self):
        """Generate synthetic video metadata"""
        np.random.seed(42)
        
        videos = []
        for i in range(self.n_videos):
            video = {
                'video_id': f'vid_{i:04d}',
                'title': f'Video Title {i}',
                'category': np.random.choice(self.categories),
                'duration': np.random.normal(600, 300),  # seconds
                'views': np.random.lognormal(10, 2),
                'likes': np.random.lognormal(5, 1.5),
                'upload_date': pd.date_range('2020-01-01', '2024-12-31', 
                                           periods=self.n_videos)[i],
                'language': np.random.choice(self.languages, 
                                           p=[0.6, 0.15, 0.1, 0.05, 0.05, 0.05]),
                'description': f'This is a {np.random.choice(self.categories).lower()} video about various topics.',
                'tags': ', '.join(np.random.choice(['tutorial', 'review', 'entertainment', 
                                                   'music', 'funny', 'educational', 'news'], 
                                                  size=np.random.randint(1, 4), replace=False))
            }
            videos.append(video)
        
        return pd.DataFrame(videos)
    
    def generate_users(self):
        """Generate synthetic user profiles"""
        np.random.seed(42)
        
        users = []
        for i in range(self.n_users):
            user = {
                'user_id': f'user_{i:04d}',
                'age': np.random.randint(13, 70),
                'gender': np.random.choice(['M', 'F', 'Other'], p=[0.45, 0.45, 0.1]),
                'location': np.random.choice(['US', 'UK', 'CA', 'AU', 'IN', 'DE', 'FR', 'JP']),
                'signup_date': pd.date_range('2018-01-01', '2024-01-01', 
                                           periods=self.n_users)[i],
                'preferred_categories': ', '.join(np.random.choice(self.categories, 
                                                                   size=np.random.randint(1, 4), 
                                                                   replace=False))
            }
            users.append(user)
        
        return pd.DataFrame(users)
    
    def generate_interactions(self, users_df, videos_df):
        """Generate user-video interaction data (ratings, watch time, etc.)"""
        np.random.seed(42)
        
        interactions = []
        n_interactions = int(self.n_users * self.n_videos * (1 - self.sparsity))
        
        for _ in range(n_interactions):
            user_id = np.random.choice(users_df['user_id'])
            video_id = np.random.choice(videos_df['video_id'])
            
            # Simulate rating (1-5 scale)
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.25, 0.35, 0.25])
            
            # Simulate watch time percentage
            watch_percentage = np.random.beta(2, 2)  # Beta distribution for realistic watch patterns
            
            # Simulate engagement
            liked = np.random.choice([0, 1], p=[0.8, 0.2])
            commented = np.random.choice([0, 1], p=[0.9, 0.1])
            shared = np.random.choice([0, 1], p=[0.95, 0.05])
            
            interaction = {
                'user_id': user_id,
                'video_id': video_id,
                'rating': rating,
                'watch_percentage': watch_percentage,
                'liked': liked,
                'commented': commented,
                'shared': shared,
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
            }
            interactions.append(interaction)
        
        return pd.DataFrame(interactions).drop_duplicates(subset=['user_id', 'video_id'])

class YouTubeRecommenderSystem:
    """Complete YouTube Video Recommendation System with multiple approaches"""
    
    def __init__(self):
        self.users_df = None
        self.videos_df = None
        self.interactions_df = None
        self.user_item_matrix = None
        self.content_features = None
        self.svd_model = None
        
    def load_data(self, users_df, videos_df, interactions_df):
        """Load the datasets"""
        self.users_df = users_df
        self.videos_df = videos_df
        self.interactions_df = interactions_df
        self._create_user_item_matrix()
        self._create_content_features()
    
    def _create_user_item_matrix(self):
        """Create user-item interaction matrix"""
        # Use rating as the primary interaction metric
        self.user_item_matrix = self.interactions_df.pivot_table(
            index='user_id', 
            columns='video_id', 
            values='rating', 
            fill_value=0
        )
        print(f"User-Item Matrix Shape: {self.user_item_matrix.shape}")
        print(f"Sparsity: {(self.user_item_matrix == 0).sum().sum() / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]):.4f}")
    
    def _create_content_features(self):
        """Create content-based features from video metadata"""
        # Combine text features
        self.videos_df['content_text'] = (
            self.videos_df['title'] + ' ' + 
            self.videos_df['category'] + ' ' + 
            self.videos_df['description'] + ' ' + 
            self.videos_df['tags']
        )
        
        # TF-IDF vectorization
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english', lowercase=True)
        content_matrix = tfidf.fit_transform(self.videos_df['content_text'])
        
        # Normalize numerical features
        numerical_features = ['duration', 'views', 'likes']
        scaler = StandardScaler()
        numerical_matrix = scaler.fit_transform(self.videos_df[numerical_features])
        
        # Combine features
        self.content_features = np.hstack([content_matrix.toarray(), numerical_matrix])
        print(f"Content Features Shape: {self.content_features.shape}")
    
    def collaborative_filtering_user_based(self, user_id, n_recommendations=10):
        """User-based Collaborative Filtering"""
        if user_id not in self.user_item_matrix.index:
            return f"User {user_id} not found in the dataset"
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Calculate user similarity using cosine similarity
        user_similarities = cosine_similarity([user_ratings], self.user_item_matrix)[0]
        
        # Find similar users (excluding the user themselves)
        similar_users_indices = np.argsort(user_similarities)[::-1][1:21]  # Top 20 similar users
        
        # Get recommendations based on similar users
        recommendations = {}
        for similar_user_idx in similar_users_indices:
            similar_user_id = self.user_item_matrix.index[similar_user_idx]
            similar_user_ratings = self.user_item_matrix.iloc[similar_user_idx]
            
            # Find videos rated highly by similar user but not watched by target user
            for video_id, rating in similar_user_ratings.items():
                if user_ratings[video_id] == 0 and rating >= 4:  # Unrated by user, highly rated by similar user
                    if video_id not in recommendations:
                        recommendations[video_id] = []
                    recommendations[video_id].append(rating * user_similarities[similar_user_idx])
        
        # Aggregate recommendations
        final_recommendations = {}
        for video_id, scores in recommendations.items():
            final_recommendations[video_id] = np.mean(scores)
        
        # Sort and return top N
        sorted_recommendations = sorted(final_recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:n_recommendations]
    
    def collaborative_filtering_item_based(self, user_id, n_recommendations=10):
        """Item-based Collaborative Filtering"""
        if user_id not in self.user_item_matrix.index:
            return f"User {user_id} not found in the dataset"
        
        # Calculate item similarity
        item_similarity = cosine_similarity(self.user_item_matrix.T)
        
        user_ratings = self.user_item_matrix.loc[user_id]
        recommendations = {}
        
        # For each unrated item, predict rating based on similar items
        for i, video_id in enumerate(self.user_item_matrix.columns):
            if user_ratings[video_id] == 0:  # Unrated video
                # Find similar items that user has rated
                similar_items_scores = []
                for j, other_video_id in enumerate(self.user_item_matrix.columns):
                    if user_ratings[other_video_id] > 0:  # User has rated this item
                        similarity = item_similarity[i][j]
                        similar_items_scores.append(similarity * user_ratings[other_video_id])
                
                if similar_items_scores:
                    recommendations[video_id] = np.mean(similar_items_scores)
        
        # Sort and return top N
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:n_recommendations]
    
    def matrix_factorization_svd(self, n_components=50, n_recommendations=10):
        """Matrix Factorization using SVD"""
        # Apply SVD
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        user_factors = svd.fit_transform(self.user_item_matrix)
        item_factors = svd.components_.T
        
        # Reconstruct the matrix
        reconstructed_matrix = user_factors @ item_factors.T
        
        self.svd_model = {
            'svd': svd,
            'user_factors': user_factors,
            'item_factors': item_factors,
            'reconstructed_matrix': reconstructed_matrix
        }
        
        print(f"SVD Model trained with {n_components} components")
        return self.svd_model
    
    def get_svd_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations using SVD model"""
        if self.svd_model is None:
            self.matrix_factorization_svd()
        
        if user_id not in self.user_item_matrix.index:
            return f"User {user_id} not found in the dataset"
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        predicted_ratings = self.svd_model['reconstructed_matrix'][user_idx]
        
        # Get recommendations for unrated items
        recommendations = []
        for i, video_id in enumerate(self.user_item_matrix.columns):
            if user_ratings.iloc[i] == 0:  # Unrated video
                recommendations.append((video_id, predicted_ratings[i]))
        
        # Sort and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def content_based_filtering(self, user_id, n_recommendations=10):
        """Content-based Filtering using video features"""
        if user_id not in self.user_item_matrix.index:
            return f"User {user_id} not found in the dataset"
        
        # Get user's rating history
        user_ratings = self.user_item_matrix.loc[user_id]
        liked_videos = user_ratings[user_ratings >= 4].index.tolist()
        
        if not liked_videos:
            return "No highly rated videos found for this user"
        
        # Get content features for liked videos
        liked_video_indices = [self.videos_df[self.videos_df['video_id'] == vid].index[0] 
                              for vid in liked_videos if vid in self.videos_df['video_id'].values]
        
        if not liked_video_indices:
            return "No matching videos found in content features"
        
        # Create user profile (average of liked videos' features)
        user_profile = np.mean(self.content_features[liked_video_indices], axis=0)
        
        # Calculate similarity with all videos
        content_similarities = cosine_similarity([user_profile], self.content_features)[0]
        
        # Get recommendations for unrated videos
        recommendations = []
        for i, video_id in enumerate(self.videos_df['video_id']):
            if user_ratings.get(video_id, 0) == 0:  # Unrated video
                recommendations.append((video_id, content_similarities[i]))
        
        # Sort and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def hybrid_recommendations(self, user_id, n_recommendations=10, weights={'cf': 0.4, 'cb': 0.3, 'svd': 0.3}):
        """Hybrid recommendation combining multiple approaches"""
        # Get recommendations from different methods
        cf_recs = dict(self.collaborative_filtering_user_based(user_id, n_recommendations*2))
        cb_recs = dict(self.content_based_filtering(user_id, n_recommendations*2))
        svd_recs = dict(self.get_svd_recommendations(user_id, n_recommendations*2))
        
        # Combine recommendations
        all_videos = set(list(cf_recs.keys()) + list(cb_recs.keys()) + list(svd_recs.keys()))
        
        hybrid_scores = {}
        for video_id in all_videos:
            score = 0
            if video_id in cf_recs:
                score += weights['cf'] * cf_recs[video_id]
            if video_id in cb_recs:
                score += weights['cb'] * cb_recs[video_id]
            if video_id in svd_recs:
                score += weights['svd'] * svd_recs[video_id]
            
            hybrid_scores[video_id] = score
        
        # Sort and return top N
        sorted_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:n_recommendations]
    
    def evaluate_recommendations(self, test_size=0.2):
        """Evaluate recommendation system performance"""
        # Split data into train and test
        train_interactions = self.interactions_df.sample(frac=1-test_size, random_state=42)
        test_interactions = self.interactions_df.drop(train_interactions.index)
        
        # Create train matrix
        train_matrix = train_interactions.pivot_table(
            index='user_id', columns='video_id', values='rating', fill_value=0
        )
        
        # Evaluate for a sample of users
        sample_users = test_interactions['user_id'].unique()[:50]  # Sample 50 users
        
        precision_scores = []
        recall_scores = []
        
        for user_id in sample_users:
            if user_id in train_matrix.index:
                # Get actual high-rated videos in test set
                actual_videos = test_interactions[
                    (test_interactions['user_id'] == user_id) & 
                    (test_interactions['rating'] >= 4)
                ]['video_id'].tolist()
                
                if actual_videos:
                    # Get recommendations (using collaborative filtering for simplicity)
                    recs = self.collaborative_filtering_user_based(user_id, 10)
                    if isinstance(recs, list):
                        recommended_videos = [rec[0] for rec in recs]
                        
                        # Calculate precision and recall
                        relevant_recommended = len(set(actual_videos) & set(recommended_videos))
                        precision = relevant_recommended / len(recommended_videos) if recommended_videos else 0
                        recall = relevant_recommended / len(actual_videos) if actual_videos else 0
                        
                        precision_scores.append(precision)
                        recall_scores.append(recall)
        
        avg_precision = np.mean(precision_scores) if precision_scores else 0
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': f1_score,
            'num_evaluated_users': len(precision_scores)
        }
    
    def plot_data_analysis(self):
        """Create visualizations for data analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Rating distribution
        axes[0, 0].hist(self.interactions_df['rating'], bins=5, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Rating Distribution')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Video categories distribution
        category_counts = self.videos_df['category'].value_counts()
        axes[0, 1].bar(category_counts.index, category_counts.values, color='lightcoral')
        axes[0, 1].set_title('Video Categories Distribution')
        axes[0, 1].set_xlabel('Category')
        axes[0, 1].set_ylabel('Number of Videos')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Watch percentage distribution
        axes[0, 2].hist(self.interactions_df['watch_percentage'], bins=20, alpha=0.7, color='lightgreen')
        axes[0, 2].set_title('Watch Percentage Distribution')
        axes[0, 2].set_xlabel('Watch Percentage')
        axes[0, 2].set_ylabel('Frequency')
        
        # 4. User age distribution
        axes[1, 0].hist(self.users_df['age'], bins=20, alpha=0.7, color='gold')
        axes[1, 0].set_title('User Age Distribution')
        axes[1, 0].set_xlabel('Age')
        axes[1, 0].set_ylabel('Frequency')
        
        # 5. Video views vs likes scatter
        axes[1, 1].scatter(np.log(self.videos_df['views']), np.log(self.videos_df['likes']), 
                          alpha=0.6, color='purple')
        axes[1, 1].set_title('Video Views vs Likes (Log Scale)')
        axes[1, 1].set_xlabel('Log(Views)')
        axes[1, 1].set_ylabel('Log(Likes)')
        
        # 6. Interactions over time
        monthly_interactions = self.interactions_df.set_index('timestamp').resample('M').size()
        axes[1, 2].plot(monthly_interactions.index, monthly_interactions.values, color='darkorange')
        axes[1, 2].set_title('Interactions Over Time')
        axes[1, 2].set_xlabel('Date')
        axes[1, 2].set_ylabel('Number of Interactions')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def display_recommendations(self, user_id, method='hybrid'):
        """Display recommendations in a nice format"""
        print(f"\n=== RECOMMENDATIONS FOR USER: {user_id} ===\n")
        
        # Get user info
        user_info = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
        print(f"User Profile:")
        print(f"  Age: {user_info['age']}")
        print(f"  Gender: {user_info['gender']}")
        print(f"  Location: {user_info['location']}")
        print(f"  Preferred Categories: {user_info['preferred_categories']}")
        
        # Get user's watch history
        user_history = self.interactions_df[self.interactions_df['user_id'] == user_id].sort_values('rating', ascending=False)
        print(f"\nUser's Top Rated Videos:")
        for _, interaction in user_history.head(3).iterrows():
            video_info = self.videos_df[self.videos_df['video_id'] == interaction['video_id']].iloc[0]
            print(f"  {video_info['title']} (Category: {video_info['category']}, Rating: {interaction['rating']})")
        
        # Get recommendations based on method
        if method == 'hybrid':
            recommendations = self.hybrid_recommendations(user_id)
        elif method == 'collaborative':
            recommendations = self.collaborative_filtering_user_based(user_id)
        elif method == 'content':
            recommendations = self.content_based_filtering(user_id)
        elif method == 'svd':
            recommendations = self.get_svd_recommendations(user_id)
        else:
            recommendations = self.hybrid_recommendations(user_id)
        
        print(f"\n{method.upper()} RECOMMENDATIONS:")
        print("-" * 50)
        
        if isinstance(recommendations, str):
            print(recommendations)
        else:
            for i, (video_id, score) in enumerate(recommendations, 1):
                video_info = self.videos_df[self.videos_df['video_id'] == video_id]
                if not video_info.empty:
                    video_info = video_info.iloc[0]
                    print(f"{i:2d}. {video_info['title']}")
                    print(f"     Category: {video_info['category']} | Score: {score:.3f}")
                    print(f"     Views: {video_info['views']:,.0f} | Likes: {video_info['likes']:,.0f}")
                    print(f"     Duration: {video_info['duration']:.0f}s | Language: {video_info['language']}")
                    print()

def main():
    """Main execution function"""
    print("YouTube Video Recommendation System")
    print("=" * 50)
    
    # Generate synthetic dataset
    print("\n1. Generating synthetic dataset...")
    generator = YouTubeDatasetGenerator(n_users=1000, n_videos=500, sparsity=0.95)
    
    users_df = generator.generate_users()
    videos_df = generator.generate_videos()
    interactions_df = generator.generate_interactions(users_df, videos_df)
    
    print(f"   Users: {len(users_df)}")
    print(f"   Videos: {len(videos_df)}")
    print(f"   Interactions: {len(interactions_df)}")
    
    # Initialize recommendation system
    print("\n2. Initializing recommendation system...")
    recommender = YouTubeRecommenderSystem()
    recommender.load_data(users_df, videos_df, interactions_df)
    
    # Train SVD model
    print("\n3. Training Matrix Factorization model...")
    recommender.matrix_factorization_svd(n_components=50)
    
    # Data analysis visualization
    print("\n4. Creating data analysis visualizations...")
    recommender.plot_data_analysis()
    
    # Evaluate system
    print("\n5. Evaluating recommendation system...")
    evaluation_results = recommender.evaluate_recommendations()
    print(f"   Precision: {evaluation_results['precision']:.3f}")
    print(f"   Recall: {evaluation_results['recall']:.3f}")
    print(f"   F1-Score: {evaluation_results['f1_score']:.3f}")
    print(f"   Evaluated Users: {evaluation_results['num_evaluated_users']}")
    
    # Sample recommendations
    print("\n6. Sample recommendations for different users...")
    sample_users = ['user_0001', 'user_0050', 'user_0100']
    methods = ['hybrid', 'collaborative', 'content', 'svd']
    
    for user_id in sample_users:
        for method in methods:
            recommender.display_recommendations(user_id, method)
            input("\nPress Enter to continue to next recommendation...")
    
    print("\nRecommendation System Demo Complete!")
    print("=" * 50)
    
    return recommender, users_df, videos_df, interactions_df

# Run the main function
if __name__ == "__main__":
    recommender, users_df, videos_df, interactions_df = main()
