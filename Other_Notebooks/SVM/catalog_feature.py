import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Set a global variable to avoid a harmless but noisy warning from the tokenizers library.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CatalogFeatureExtractor:
    """
    A class to handle the feature extraction from the transformed catalog data.
    It uses a pre-trained SentenceTransformer model to create text embeddings
    and adds them as new columns to the input DataFrame.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the feature extractor by loading the pre-trained model.
        
        Args:
            model_name (str): The name of the SentenceTransformer model to use.
        """
        print(f"Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully.")
        
    def extract_features(self, df):
        """
        Takes a preprocessed DataFrame, generates text embeddings for the 'all_text'
        column, and returns a new DataFrame with the embeddings added as columns.

        Args:
            df (pd.DataFrame): The DataFrame produced by the `transform_catalog_data` function.
                               It must contain an 'all_text' column.

        Returns:
            pd.DataFrame: The original DataFrame with new 'text_feat_...' columns appended.
        """
        print("Extracting text features...")
        
        # 1. Generate text embeddings from the 'all_text' column
        print("Generating text embeddings... (This may take a moment)")
        text_embeddings = self.model.encode(
            df['all_text'].tolist(), 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print("Text embeddings generated.")

        # 2. Create a new DataFrame from the embeddings
        text_feature_names = [f'text_feat_{i}' for i in range(text_embeddings.shape[1])]
        text_features_df = pd.DataFrame(text_embeddings, columns=text_feature_names, index=df.index)

        # 3. Concatenate the new feature DataFrame with the original DataFrame
        # This keeps all original columns and adds the new ones.
        final_df = pd.concat([df, text_features_df], axis=1)
        
        print(f"Feature extraction complete. Final DataFrame shape: {final_df.shape}")
        
        return final_df
