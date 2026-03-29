from text_preprocessor import TextPreprocessor
from visualization_generator import VisualizationGenerator
from ranknet import RankNetModel
from pairwise import PairwiseDataset
import pandas as pd
import numpy as np
import re
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42) 


class EnhancedCandidateRankingSystem:
    def __init__(self, keywords, model_name='bert-base-uncased'):
        self.keywords = keywords
        self.text_preprocessor = TextPreprocessor()
        self.viz_generator = VisualizationGenerator(output_dir='results/figures')
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load BERT model and tokenizer
        print(f"Loading BERT model: {model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.bert_available = True
        
        # Store data and rankings
        self.candidates_df = None
        self.embeddings = None
        self.rankings = None
        self.starred_candidates = []
        self.embedding_cache = {}
        
        # RankNet model
        self.ranknet_model = None
        self.ranknet_trained = False
        
        print("Preprocessing keywords...")
        self.keywords_processed = self.text_preprocessor.preprocess(keywords)
        
        if self.bert_available:
            self.keyword_embedding = self._get_text_embedding(self.keywords_processed)
        else:
            self.keyword_embedding = None
    
    def _get_text_embedding(self, text):
        """Get BERT embedding with caching"""
        if not self.bert_available:
            return np.zeros(768)
        
        if pd.isna(text) or text == '':
            return np.zeros(768)
        
        # Check cache
        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Tokenize
        try:
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use mean pooling of last hidden state
                embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            
            embedding = embeddings.cpu().numpy().reshape(-1)
            
            # Cache the result
            self.embedding_cache[cache_key] = embedding
            
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return np.zeros(768)
    
    def _extract_role_level(self, job_title):
        title_lower = str(job_title).lower()
        
        level_patterns = {
            'executive': ['chief', 'cfo', 'cto', 'ceo', 'cmo', 'chro', 'svp', 'evp', 'president'],
            'director': ['director', 'head of', 'vp', 'vice president'],
            'senior': ['senior', 'sr', 'sr.', 'lead', 'principal'],
            'mid': ['manager', 'specialist', 'coordinator', 'analyst', 'advisor', 'professional'],
            'junior': ['junior', 'jr', 'jr.', 'associate', 'assistant', 'entry'],
            'intern': ['intern', 'internship', 'trainee', 'student']
        }
        
        for level, patterns in level_patterns.items():
            if any(pattern in title_lower for pattern in patterns):
                return level
        return 'junior'
    
    def _preprocess_connections(self, conn_str):
        if pd.isna(conn_str) or conn_str == '':
            return 0
        
        conn_str = str(conn_str)
        numbers = re.findall(r'\d+', conn_str)
        if numbers:
            num = int(numbers[0])
            if '+' in conn_str:
                return num + 50
            return num
        return 0
    
    def load_and_preprocess_data(self, data_path):
        print(f"\n{'='*60}")
        print(f"LOADING DATA FROM: {data_path}")
        print(f"{'='*60}")
        
        self.candidates_df = pd.read_csv(data_path)
        print(f"Loaded {len(self.candidates_df)} candidates")
        print(f"Columns: {list(self.candidates_df.columns)}")
        
        print("\nMissing data:")
        print(self.candidates_df.isnull().sum())
        
        # Preprocess features
        self._preprocess_features()
        
        return self.candidates_df
    
    def _preprocess_features(self):

        print("\nCleaning job titles and locations...")
        
        # Try lemmatization first, fall back to stemming if it fails
        try:
            self.candidates_df['job_title_cleaned'] = self.candidates_df['job_title'].apply(
                lambda x: self.text_preprocessor.preprocess(x, use_lemmatization=True)
            )
        except Exception as e:
            print(f"Lemmatization failed, using stemming: {e}")
            self.candidates_df['job_title_cleaned'] = self.candidates_df['job_title'].apply(
                lambda x: self.text_preprocessor.preprocess(x, use_lemmatization=False)
            )

        self.candidates_df['role_level'] = self.candidates_df['job_title'].apply(self._extract_role_level)
        
        print("\nSample cleaned titles:")
        for i in range(min(5, len(self.candidates_df))):
            print(f"Original: {self.candidates_df['job_title'].iloc[i]}")
            print(f"Cleaned:  {self.candidates_df['job_title_cleaned'].iloc[i]}")
            print(f"Level:    {self.candidates_df['role_level'].iloc[i]}")
            print()
        
        print("\nCleaning and standardizing locations...")
        self.candidates_df['location_cleaned'] = self.candidates_df['location'].apply(
            lambda x: self.text_preprocessor.clean_location(x)
        )
        
        print("\nSample cleaned locations:")
        sample_size = min(10, len(self.candidates_df))
        for i in range(sample_size):
            print(f"Original: {str(self.candidates_df['location'].iloc[i]):<40} -> Cleaned: {self.candidates_df['location_cleaned'].iloc[i]}")
        
        self.candidates_df['connections_numeric'] = self.candidates_df['connection'].apply(
            self._preprocess_connections
        )
        self.candidates_df['connections_log'] = np.log1p(self.candidates_df['connections_numeric'])
        
        # Calculate BERT similarities
        if self.bert_available:
            print("\nCalculating BERT similarities...")
            bert_similarities = []
            total = len(self.candidates_df)
            for i, title in enumerate(self.candidates_df['job_title']):
                if i % 50 == 0:
                    print(f"  Processing {i}/{total}")
                cleaned = self.text_preprocessor.preprocess(str(title))
                emb = self._get_text_embedding(cleaned)
                sim = cosine_similarity(emb.reshape(1, -1), self.keyword_embedding.reshape(1, -1))[0][0]
                bert_similarities.append(sim)
            
            self.candidates_df['bert_similarity'] = bert_similarities
        else:
            self.candidates_df['bert_similarity'] = 0
        
        self.candidates_df.to_csv('results/summary/candidate_profile.csv', index=False)
        
        # Generate embeddings for re-ranking if BERT is available
        if self.bert_available:
            print("\nGenerating embeddings for re-ranking...")
            self.embeddings = []
            total = len(self.candidates_df)
            for i, title in enumerate(self.candidates_df['job_title_cleaned']):
                if i % 50 == 0:
                    print(f"  Generating embedding {i}/{total}")
                emb = self._get_text_embedding(str(title))
                self.embeddings.append(emb)
            
            self.embeddings = np.array(self.embeddings)
            print(f"Embeddings shape: {self.embeddings.shape}")
        else:
            self.embeddings = np.zeros((len(self.candidates_df), 768))
    
    def initial_ranking(self):
        if self.candidates_df is None:
            raise ValueError("Load data first!")
        
        fitness = self.candidates_df['bert_similarity']
        
        # Add small noise to break ties
        fitness += np.random.normal(0, 0.001, len(fitness))
        fitness = np.clip(fitness, 0, 1)
        
        # Create rankings
        self.candidates_df['fitness_score'] = fitness
        self.rankings = self.candidates_df.sort_values('fitness_score', ascending=False).reset_index(drop=True)
        self.rankings['rank'] = self.rankings.index + 1
        self.rankings['is_starred'] = False
        
        print("\n" + "="*60)
        print("INITIAL RANKING RESULTS")
        print("="*60)
        print(f"Search Query: '{self.keywords}'")
        print(f"Total Candidates: {len(self.rankings)}")
        print("\nTop 10 Candidates:")
        
        display_cols = ['id', 'job_title', 'location_cleaned', 'role_level', 'fitness_score', 'rank']
        display_cols = [c for c in display_cols if c in self.rankings.columns]
        print(self.rankings[display_cols].head(10))
        self.rankings.to_csv('results/summary/initial_rankings.csv', index=False)
        
        return self.rankings
    
    def star_candidate(self, candidate_id):
        """Star a candidate and re-rank with RankNet"""
        if candidate_id not in self.candidates_df['id'].values:
            print(f"Candidate {candidate_id} not found")
            return self.rankings
        
        if candidate_id not in self.starred_candidates:
            self.starred_candidates.append(candidate_id)
        
        print(f"\n{'='*40}")
        print(f"STARRED CANDIDATE: {candidate_id}")
        print(f"{'='*40}")
        candidate_info = self.candidates_df[self.candidates_df['id'] == candidate_id].iloc[0]
        print(f"Title: {candidate_info['job_title']}")
        print(f"Location: {candidate_info.get('location_cleaned', 'N/A')}")
        print(f"Current Fitness: {candidate_info['fitness_score']:.4f}")
        print(f"Total starred: {len(self.starred_candidates)}")
        
        self._rerank()
        return self.rankings
    
    def _train_ranknet(self, features, starred_indices):
        # features = self.embeddings (shape: [num_candidates, 768])

        if len(starred_indices) < 2:
            print("Need at least 2 starred candidates to train RankNet")
            return None
        
        print(f"\nTraining RankNet with {len(starred_indices)} starred candidates...")
        
        pairs = []
        
        for starred_idx in starred_indices:
            for i in range(len(features)):
                if i not in starred_indices:
                    pairs.append((starred_idx, i))
        
        starred_scores = self.candidates_df.iloc[starred_indices]['bert_similarity'].values
        starred_indices_sorted = [starred_indices[i] for i in np.argsort(starred_scores)[::-1]]
        
        # Create pairs among starred candidates based on original scores
        for i in range(len(starred_indices_sorted)):
            for j in range(i + 1, len(starred_indices_sorted)):
                pairs.append((starred_indices_sorted[i], starred_indices_sorted[j]))
        
        if len(pairs) == 0:
            print("No valid pairs generated for training")
            return None
        
        print(f"Generated {len(pairs)} training pairs")
        
        dataset = PairwiseDataset(pairs, features)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        input_dim = features.shape[1]
        self.ranknet_model = RankNetModel(input_dim=input_dim, hidden_dims=[256, 128, 64])
        self.ranknet_model.to(self.device)
        
        optimizer = torch.optim.Adam(self.ranknet_model.parameters(), lr=0.001)
        
        # Training loop
        num_epochs = 100
        self.ranknet_model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (x_i, x_j) in enumerate(dataloader):
                x_i = x_i.to(self.device) 
                x_j = x_j.to(self.device) 
                
                optimizer.zero_grad() # Reset gradients from previous batch
                
                # Forward pass
                score_i = self.ranknet_model(x_i).squeeze()
                score_j = self.ranknet_model(x_j).squeeze()
                
                # RankNet loss: P(i > j) = sigmoid(s_i - s_j)
                # We want i to be ranked higher than j (label = 1)
                pred_probs = torch.sigmoid(score_i - score_j)
                
                # Binary cross entropy loss with label 1 (i should be higher)
                loss = F.binary_cross_entropy(pred_probs, torch.ones_like(pred_probs))
                
                loss.backward() # Compute gradients
                optimizer.step() # Update model weights
                
                total_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        self.ranknet_trained = True
        print("RankNet training complete!")
        return self.ranknet_model
    
    def _rerank(self):
        if not self.starred_candidates or not self.bert_available:
            return
        
        starred_indices = []
        for cand_id in self.starred_candidates:
            idx = self.candidates_df[self.candidates_df['id'] == cand_id].index[0]
            starred_indices.append(idx)
        
        if len(starred_indices) >= 2:
            self._train_ranknet(self.embeddings, starred_indices)
            
            # Use trained model to predict new scores
            if self.ranknet_trained and self.ranknet_model is not None:
                self.ranknet_model.eval()
                with torch.no_grad(): # Disables gradient calculation
                    features_tensor = torch.FloatTensor(self.embeddings).to(self.device) # Convert embeddings to PyTorch tensor
                    new_scores = self.ranknet_model(features_tensor).cpu().numpy().flatten()
                
                # Normalize scores to [0, 1]
                if new_scores.max() > new_scores.min():
                    new_scores = (new_scores - new_scores.min()) / (new_scores.max() - new_scores.min() + 1e-8)
                else:
                    new_scores = np.clip(new_scores, 0, 1)
                
                # Update fitness scores based on RankNet predictions
                new_fitness = new_scores
                
                # Boost starred candidates
                for idx in starred_indices:
                    new_fitness[idx] = min(1.0, new_fitness[idx] * 1.1)
                
                self.candidates_df['fitness_score'] = new_fitness
                
                print("\n" + "="*50)
                print("RANKNET RE-RANKING COMPLETE")
                print("="*50)
                print(f"RankNet Model trained with {len(starred_indices)} starred candidates")
        
        else:
            # Fallback to similarity-based ranking when not enough starred candidates
            print("\n" + "="*50)
            print("INSUFFICIENT STARRED CANDIDATES FOR RANKNET")
            print("="*50)
            print(f"Need at least 2 starred candidates, currently have {len(starred_indices)}")
            print("Using similarity-based fallback...")
            
            # Calculate similarity to starred candidates
            starred_embs = [self.embeddings[idx] for idx in starred_indices]
            avg_starred = np.mean(starred_embs, axis=0).reshape(1, -1)
            starred_sims = cosine_similarity(self.embeddings, avg_starred).flatten()
            
            # Update fitness scores based on similarity to starred candidates
            new_fitness = starred_sims
            
            # Normalize
            if new_fitness.max() > new_fitness.min():
                new_fitness = (new_fitness - new_fitness.min()) / (new_fitness.max() - new_fitness.min() + 1e-8)
            
            # Boost starred candidates
            for idx in starred_indices:
                new_fitness[idx] = min(1.0, new_fitness[idx] * 1.05)
            
            self.candidates_df['fitness_score'] = new_fitness
        
        # Update rankings
        self.rankings = self.candidates_df.sort_values('fitness_score', ascending=False).reset_index(drop=True)
        self.rankings['rank'] = self.rankings.index + 1
        self.rankings['is_starred'] = self.rankings['id'].isin(self.starred_candidates)
        
        # Print results
        print("\nTop 10 after re-ranking:")
        display_cols = ['id', 'job_title', 'role_level', 'fitness_score', 'is_starred']
        display_cols = [c for c in display_cols if c in self.rankings.columns]
        
        display_df = self.rankings[display_cols].head(10).copy()
        if 'fitness_score' in display_df.columns:
            display_df['fitness_score'] = display_df['fitness_score'].round(4)
        
        print(display_df.to_string())
        
        # Show starred candidate positions
        starred_rows = self.rankings[self.rankings['is_starred']]
        if not starred_rows.empty:
            print("\n" + "-"*30)
            print("Starred Candidates New Positions:")
            print("-"*30)
            for _, row in starred_rows.iterrows():
                print(f"ID: {row['id']:<8} | Rank: {row['rank']:<4} | Score: {row['fitness_score']:.4f}")
                print(f"  Title: {row['job_title'][:60]}")
        
        self._calculate_ranking_metrics()
    
    def _calculate_ranking_metrics(self):
        """Calculate metrics to evaluate ranking quality"""
        if not hasattr(self, 'rankings') or self.rankings is None:
            return
        
        starred_mask = self.rankings['is_starred'].values
        starred_positions = np.where(starred_mask)[0] + 1
        
        if len(starred_positions) > 0:
            dcg = np.sum(1 / np.log2(starred_positions + 1)) # NDCG (Normalized Discounted Cumulative Gain)
            idcg = np.sum(1 / np.log2(np.arange(1, len(starred_positions) + 1) + 1)) # Ideal DCG: Best possible score
            ndcg = dcg / idcg if idcg > 0 else 0
            
            print("\n" + "-"*30)
            print("Ranking Quality Metrics:")
            print("-"*30)
            print(f"Number of starred candidates: {len(starred_positions)}")
            print(f"Starred positions: {starred_positions}")
            print(f"Average starred rank: {np.mean(starred_positions):.2f}")
            print(f"Median starred rank: {np.median(starred_positions):.2f}")
            print(f"NDCG (simplified): {ndcg:.4f}")
            
            if len(starred_positions) > 1:
                top_k = min(20, len(self.rankings))
                starred_in_top_k = np.sum(starred_positions <= top_k)
                precision_at_k = starred_in_top_k / min(top_k, len(starred_positions))
                print(f"Precision@{top_k}: {precision_at_k:.4f}")
    
    def get_candidate_details(self, candidate_id):
        if candidate_id not in self.candidates_df['id'].values:
            print(f"Candidate {candidate_id} not found")
            return None
        
        candidate = self.candidates_df[self.candidates_df['id'] == candidate_id].iloc[0]
        
        print(f"\n{'='*40}")
        print(f"CANDIDATE DETAILS: {candidate_id}")
        print(f"{'='*40}")
        print(f"Job Title: {candidate['job_title']}")
        print(f"Location: {candidate.get('location_cleaned', 'N/A')}")
        print(f"Connections: {candidate.get('connection', 'N/A')}")
        print(f"Role Level: {candidate['role_level']}")
        print(f"\nScores:")
        print(f"  BERT Similarity: {candidate['bert_similarity']:.4f}")
        print(f"\n  FINAL FITNESS: {candidate['fitness_score']:.4f}")
        
        return candidate
    
    def export_results(self, filename='enhanced_rankings.csv'):
        if self.rankings is not None:
            export_cols = ['id', 'job_title', 'job_title_cleaned', 'location_cleaned', 'role_level',
                          'bert_similarity', 'fitness_score', 'rank', 'is_starred']
            
            export_cols = [c for c in export_cols if c in self.rankings.columns]
            self.rankings[export_cols].to_csv(filename, index=False)
            print(f"\nResults exported to {filename}")
            print(f"Exported columns: {export_cols}")