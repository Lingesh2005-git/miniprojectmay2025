import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class ResearchCollaborationRecommender:
    def __init__(self):
        self.researchers = None
        self.publications = None
        self.coauthor_matrix = None
        self.tfidf_matrix = None
        self.researcher_ids = {}
    
    def load_data(self, researchers_path, publications_path):
        """Load data with explicit path verification"""
        if not os.path.exists(researchers_path):
            raise FileNotFoundError(f"Researchers file not found at: {os.path.abspath(researchers_path)}")
        if not os.path.exists(publications_path):
            raise FileNotFoundError(f"Publications file not found at: {os.path.abspath(publications_path)}")
            
        self.researchers = pd.read_csv(researchers_path)
        self.publications = pd.read_csv(publications_path)
        
        # Create researcher ID mapping
        self.researcher_ids = {name: idx for idx, name in enumerate(self.researchers['name'])}
    
    def build_coauthor_matrix(self):
        """Build co-authorship matrix for collaborative filtering"""
        num_researchers = len(self.researchers)
        self.coauthor_matrix = np.zeros((num_researchers, num_researchers))
        
        for _, pub in self.publications.iterrows():
            authors = [a.strip() for a in pub['authors'].split(',') if a.strip() in self.researcher_ids]
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    idx1 = self.researcher_ids[authors[i]]
                    idx2 = self.researcher_ids[authors[j]]
                    self.coauthor_matrix[idx1, idx2] += 1
                    self.coauthor_matrix[idx2, idx1] += 1
        
        # Convert to sparse matrix
        self.coauthor_matrix = csr_matrix(self.coauthor_matrix)
    
    def build_content_profiles(self):
        """Build TF-IDF profiles from research interests and publications"""
        text_data = []
        for _, researcher in self.researchers.iterrows():
            # Get researcher's publications
            pubs = self.publications[self.publications['authors'].str.contains(researcher['name'])]
            combined_text = researcher['research_interests'] + ' ' + ' '.join(pubs['keywords'])
            text_data.append(combined_text)
        
        vectorizer = TfidfVectorizer()
        self.tfidf_matrix = vectorizer.fit_transform(text_data)
    
    def recommend_collaborators(self, researcher_name, top_n=5):
        """Generate hybrid recommendations"""
        if researcher_name not in self.researcher_ids:
            raise ValueError(f"Researcher {researcher_name} not found in database")
        
        researcher_idx = self.researcher_ids[researcher_name]
        
        # Content-based similarity
        content_sim = cosine_similarity(
            self.tfidf_matrix[researcher_idx],
            self.tfidf_matrix
        ).flatten()
        
        # Collaborative filtering similarity
        collab_sim = self.coauthor_matrix[researcher_idx].toarray().flatten()
        
        # Normalize and combine scores
        content_sim = (content_sim - content_sim.min()) / (content_sim.max() - content_sim.min() + 1e-6)
        collab_sim = (collab_sim - collab_sim.min()) / (collab_sim.max() - collab_sim.min() + 1e-6)
        
        combined_scores = 0.6 * content_sim + 0.4 * collab_sim
        
        # Get top recommendations
        top_indices = np.argsort(combined_scores)[-top_n-1:-1][::-1]
        
        recommendations = []
        for idx in top_indices:
            if idx != researcher_idx:
                rec = self.researchers.iloc[idx].to_dict()
                rec['score'] = combined_scores[idx]
                recommendations.append(rec)
        
        return recommendations

# Example Usage with Enhanced Error Handling
if __name__ == "__main__":
    recommender = ResearchCollaborationRecommender()
    
    # File path configuration
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    data_dir = os.path.join(current_dir, "research_data")
    
    try:
        # Load data
        recommender.load_data(
            os.path.join(data_dir, "researchers.csv"),
            os.path.join(data_dir, "publications.csv")
        )
        
        # Build models
        recommender.build_coauthor_matrix()
        recommender.build_content_profiles()
        
        # Get recommendations
        researcher_name = input("Enter the researcher name provided in researcher.csv file:") # Replace with actual name from your CSV
        recommendations = recommender.recommend_collaborators(researcher_name, top_n=11)
        
        # Print results
        print(f"Recommended collaborators for {researcher_name}:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['name']} ({rec['department']})")
            print(f"   Research Interests: {rec['research_interests']}")
            print(f"   Recommendation Score: {rec['score']:.3f}")
            print(f"   Email: {rec['email']}")
            
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("\n❗ Solution:")
        print(f"1. Create a 'research_data' folder in: {current_dir}")
        print("2. Ensure these files exist:")
        print(f"   - {os.path.join(data_dir, 'researchers.csv')}")
        print(f"   - {os.path.join(data_dir, 'publications.csv')}")
        print("3. Generate data using the data generation script first")
    except Exception as e:
        print(f"An error occurred: {str(e)}")