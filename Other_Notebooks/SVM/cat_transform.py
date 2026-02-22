# file: cat_transform.py
import pandas as pd
import numpy as np
import re

class CatalogDataTransformer:
    """
    Transforms catalog data by parsing, cleaning, and feature engineering.
    This class learns parameters (like median for imputation) from the training
    data and applies them consistently to any dataset (train, test, etc.).
    """
    def __init__(self):
        self.median_value = None
        self.expected_unit_cols = [
            'Unit_count', 'Unit_fluid_ounce', 'Unit_foot', 'Unit_gram', 
            'Unit_kilogram', 'Unit_liter', 'Unit_milliliter', 'Unit_other', 
            'Unit_ounce', 'Unit_pound'
        ]

    def fit(self, df):
        """
        Learns the necessary parameters from the training data.
        In this case, it learns the median of the 'Value' column for imputation.

        Args:
            df (pd.DataFrame): The training DataFrame.
        """
        print("Fitting CatalogDataTransformer on training data...")
        # Temporarily process to find the median
        temp_df = self._parse_and_extract(df.copy())
        temp_df['Value'] = pd.to_numeric(temp_df['Value'], errors='coerce')
        self.median_value = temp_df['Value'].median()
        print(f"Learned median 'Value' for imputation: {self.median_value}")
        return self

    def transform(self, df):
        """
        Applies the full transformation pipeline to the data.

        Args:
            df (pd.DataFrame): DataFrame to transform (can be train or test).

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        if self.median_value is None:
            raise RuntimeError("Transformer has not been fitted! Call .fit() on the training data first.")

        print(f"Transforming catalog data... (DataFrame shape: {df.shape})")
        
        # --- 1. Parse, extract, and clean ---
        transformed_df = self._parse_and_extract(df.copy())

        # --- 2. Clean 'Value' and impute using the *learned* median ---
        transformed_df['Value'] = pd.to_numeric(transformed_df['Value'], errors='coerce')
        transformed_df['Value'].fillna(self.median_value, inplace=True)

        # --- 3. One-Hot Encode 'Unit' column ---
        unit_dummies = pd.get_dummies(transformed_df['Unit_cleaned'], prefix='Unit')
        transformed_df = pd.concat([transformed_df, unit_dummies], axis=1)
        
        # Ensure all expected unit columns exist, adding them if they don't
        for col in self.expected_unit_cols:
            if col not in transformed_df.columns:
                transformed_df[col] = False

        # --- 4. Create 'all_text' column for vectorization ---
        transformed_df['all_text'] = (
            transformed_df['Item-name'].fillna('') + ' ' +
            transformed_df['Bullet-Points'].fillna('') + ' ' +
            transformed_df['Product-Descriptions'].fillna('')
        )

        # --- 5. Final Cleanup ---
        # Select and reorder columns to ensure consistency
        final_cols = ['sample_id', 'Value', 'IPQ'] + self.expected_unit_cols + ['all_text']
        final_df = transformed_df[final_cols]
        
        print("Transformation complete.")
        return final_df

    def _parse_and_extract(self, df):
        """Helper method for initial parsing and feature extraction."""
        # This internal method contains the parsing logic from your original function
        parsed_data = [self._parse_row(row) for _, row in df.iterrows()]
        features_df = pd.DataFrame(parsed_data)

        # Consolidate fragmented columns
        def merge_single_value(temp_df, base_name):
            cols = [col for col in temp_df.columns if col.startswith(base_name)]
            return temp_df[cols].bfill(axis=1).iloc[:, 0] if cols else pd.Series(index=temp_df.index)

        def concat_text_features(temp_df, base_name):
            cols = [col for col in temp_df.columns if col.startswith(base_name)]
            if not cols: return pd.Series(index=temp_df.index)
            return temp_df[cols].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

        result_df = pd.DataFrame()
        result_df['sample_id'] = features_df['sample_id']
        result_df['Value'] = features_df.get('Value')
        result_df['Item-name'] = merge_single_value(features_df, 'Item Name')
        result_df['Bullet-Points'] = concat_text_features(features_df, 'Bullet Point')
        result_df['Product-Descriptions'] = concat_text_features(features_df, 'Product Description')
        
        # Extract IPQ
        text_for_ipq = result_df['Item-name'].fillna('') + ' ' + result_df['Bullet-Points'].fillna('')
        result_df['IPQ'] = text_for_ipq.apply(self._extract_ipq)

        # Clean Units
        unit_series = features_df.get('Unit')
        result_df['Unit_cleaned'] = unit_series.apply(self._clean_unit) if unit_series is not None else 'other'

        return result_df

    def _parse_row(self, row):
        features = {'sample_id': row['sample_id']}
        for line in row['catalog_content'].split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                features[parts[0].strip()] = parts[1].strip()
        return features
        
    def _extract_ipq(self, text):
        # Your IPQ extraction logic remains the same
        patterns = [r'pack of (\d+)', r'(\d+)\s*count', r'case of (\d+)', r'(\d+)\s*[pP][kK]', r'(\d+)\s*[cC][tT]', r'pack:\s*(\d+)', r'(\d+)\s*per case', r'\((\d+)\s*tins\)' ]
        match = re.search('|'.join(patterns), text, re.IGNORECASE)
        if match:
            for group in match.groups():
                if group: return int(group)
        return 1
        
    def _clean_unit(self, unit):
        # Your unit cleaning logic remains the same
        if not isinstance(unit, str) or unit.strip() in ['', '-', '---', 'None']: return 'other'
        unit_map = {'ounce': ['ounce', 'ounces', 'oz', '7,2 oz'], 'fluid_ounce': ['fl oz', 'fluid ounce', 'fl. oz', 'fluid ounces', 'fl ounce', 'fl.oz', 'fluid ounce(s)'], 'count': ['count', 'ct', 'each', 'k-cups', 'jar', 'can', 'bottle', 'bottles', 'pack', 'packs', 'bag', 'bags', 'box', 'pouch', 'bucket', 'units', 'capsule', 'tea bags', 'paper cupcake liners', 'ziplock bags', 'box/12', 'per box', 'per package', 'per carton', 'carton'], 'pound': ['pound', 'pounds', 'lb'], 'gram': ['gram', 'grams', 'gr', 'gramm', 'grams(gm)'], 'kilogram': ['kg'], 'liter': ['liters', 'ltr'], 'milliliter': ['millilitre', 'milliliter', 'ml', 'mililitro'], 'foot': ['foot', 'sq ft']}
        value_to_key_map = {v: k for k, values in unit_map.items() for v in values}
        return value_to_key_map.get(unit.lower().strip().replace('.', ''), 'other')