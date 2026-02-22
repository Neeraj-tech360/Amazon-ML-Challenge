# file: img_feature.py
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CLIPImageFeatureExtractor:
    """
    Extracts features from images using a pre-trained CLIP model.
    """
    def __init__(self, model_name='clip-ViT-B-32'):
        print(f"Loading CLIP model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("CLIP model loaded successfully.")

    def extract_features(self, df, image_path_col='image_path', sample_id_col='sample_id', batch_size=64):
        """
        Processes a DataFrame to extract image features.

        Returns:
            pd.DataFrame: A new DataFrame with sample_id and image features.
        """
        print("Preparing images for feature extraction...")
        
        image_paths = df[image_path_col].tolist()
        pil_images = []
        placeholder_image = Image.new('RGB', (224, 224))

        for path in tqdm(image_paths, desc="Loading images"):
            if pd.notna(path) and os.path.exists(path):
                try:
                    pil_images.append(Image.open(path))
                except (IOError, Image.UnidentifiedImageError):
                    pil_images.append(placeholder_image)
            else:
                pil_images.append(placeholder_image)

        print("\nEncoding images to feature vectors...")
        feature_vectors = self.model.encode(
            pil_images,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        # Create a new DataFrame for the features, including the sample_id
        features_df = pd.DataFrame(feature_vectors, columns=[f'img_feat_{i}' for i in range(feature_vectors.shape[1])])
        features_df[sample_id_col] = df[sample_id_col].values

        # Reorder columns to have sample_id first for clarity
        final_cols = [sample_id_col] + [col for col in features_df.columns if col != sample_id_col]
        features_df = features_df[final_cols]
        
        print(f"Image feature extraction complete. Shape: {features_df.shape}")
        return features_df