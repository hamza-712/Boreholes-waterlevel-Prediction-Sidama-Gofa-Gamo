import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import warnings
import joblib

warnings.filterwarnings('ignore')

def predict_swl(new_data):
    """
    Transform new input data and predict SWL using a trained model.

    Parameters:
    new_data (pd.DataFrame): DataFrame with raw features matching the training data structure.

    Returns:
    np.ndarray: Predicted SWL values.
    """
    # Load saved artifacts from training
    # Ensure these files exist from the training step
    soil_mapping = joblib.load('soil_mapping.pkl')
    scaler = joblib.load('scaler.pkl')
    selected_features = joblib.load('selected_features.pkl')
    target_transformer = joblib.load('target_transformer.pkl')
    model = joblib.load('model.pkl')

    # Step 1: Engineer features
    features = pd.DataFrame()

    # Basic features
    features['Elevation'] = new_data['Elevation']
    features['latitude'] = new_data['latitude']
    features['longitude'] = new_data['longitude']

    # Soil type encoding using the precomputed mapping
    # Handle potential NaNs if a soil type in new_data is not in the mapping
    features['soil_swl_mean'] = new_data['SOIL_TYPE'].map(soil_mapping)
    # Fill NaNs introduced by mapping with the mean of the known soil_swl_mean values
    mean_soil_swl = pd.Series(soil_mapping.values()).mean()
    features['soil_swl_mean'].fillna(mean_soil_swl, inplace=True)


    # Identify climate-related columns dynamically
    precip_cols = [col for col in new_data.columns if 'Precip' in col]
    temp_cols = [col for col in new_data.columns if 'LSTDay' in col]
    # Correcting humidity column name to match the training code's assumption ('Humidity')
    # If your test data uses 'SpecificHumidity', you'll need to update either the training or prediction side.
    # Based on the error context and the training code, it seems 'Humidity' is expected for aggregation.
    # Let's adjust the prediction code to match the feature engineering logic in the training class.
    humidity_cols = [col for col in new_data.columns if 'Humidity' in col]

    # Aggregated climate features
    # Handle potential NaNs in climate columns by filling with a representative value (e.g., mean)
    # This is a basic approach; a more robust solution might involve imputation.
    for col in precip_cols + temp_cols + humidity_cols:
         if col in new_data.columns:
             # Fill NaNs in the original data before calculating aggregates
             new_data[col].fillna(new_data[col].mean(), inplace=True) # Fill with mean of the column

    features['precip_annual_total'] = new_data[precip_cols].sum(axis=1)
    # Add a check for zero mean to avoid division by zero
    mean_precip = new_data[precip_cols].mean(axis=1)
    features['precip_seasonality'] = new_data[precip_cols].std(axis=1) / (mean_precip + 1e-6) # Add small epsilon
    features['temp_annual_mean'] = new_data[temp_cols].mean(axis=1)
    features['humidity_annual_mean'] = new_data[humidity_cols].mean(axis=1)

    # Interaction features
    features['elevation_x_precip'] = features['Elevation'] * features['precip_annual_total']
    features['temp_x_humidity'] = features['temp_annual_mean'] * features['humidity_annual_mean']

    # Ensure all features used in training are present and in the same order
    # If the number of features in the test set doesn't match the selected_features length after engineering,
    # there's likely a mismatch in feature creation logic.
    # We should ensure the `features` DataFrame has all and only the columns expected by the scaler and model.
    # A robust way is to align columns to the training columns before scaling.
    training_feature_columns = joblib.load('training_feature_columns.pkl') # Need to save this during training

    # Add missing columns (if any) and fill with 0 or mean (depending on feature type)
    for col in training_feature_columns:
        if col not in features.columns:
            features[col] = 0 # Or an appropriate fill value

    # Reindex and select columns to match the order and set from training
    features = features[training_feature_columns]

    # Step 2: Scale features using the trained scaler
    # Ensure there are no NaNs before scaling. Fill any remaining NaNs.
    # Filling with mean is a common approach, but consider the impact on scaled data.
    # It's better to ensure NaNs are handled during feature engineering or imputation.
    # Let's add a final check here just in case, but the goal is to handle NaNs earlier.
    if features.isnull().sum().sum() > 0:
        print("Warning: NaNs found in engineered features before scaling. Filling with mean.")
        features.fillna(features.mean(), inplace=True)


    features_scaled = scaler.transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

    # Step 3: Select the same features as in training
    # The selected_features list should contain column names, so we select from the scaled DataFrame
    X_new = features_scaled_df[selected_features].values

    # Step 4: Predict transformed SWL and inverse-transform to original scale
    y_pred_transformed = model.predict(X_new)
    y_pred = target_transformer.inverse_transform(y_pred_transformed.reshape(-1, 1)).ravel()

    return y_pred

class GroundwaterModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.target_transformer = None
        self.feature_scaler = None
        self.selected_features = None
        self.soil_mapping = None
        self.df = None # Added to make df accessible class-wide
        self.training_feature_columns = None # Added to store feature column names

    def load_and_clean_data(self):
        df = pd.read_csv(self.data_path)
        print(f"Initial dataset: {df.shape[0]} rows, {df.shape[1]} columns")

        # Handle missing values in Elevation
        df['Elevation'].fillna(df['Elevation'].median(), inplace=True)

        # Handle missing values in SWL before outlier removal and soil mapping
        # Option 1: Drop rows with missing SWL (simplest for training)
        df.dropna(subset=['SWL'], inplace=True)
        print(f"Dataset after dropping rows with missing SWL: {df.shape[0]} rows")

        # Handle missing values in SOIL_TYPE before mapping
        # Option 1: Fill with a placeholder or the mode
        if 'SOIL_TYPE' in df.columns:
             df['SOIL_TYPE'].fillna('Unknown', inplace=True) # Fill with 'Unknown' or mode()

        # Handle NaNs in climate columns before aggregation
        climate_cols_all = [col for col in df.columns if any(keyword in col for keyword in ['Precip', 'LSTDay', 'Humidity'])]
        for col in climate_cols_all:
             if col in df.columns:
                 df[col].fillna(df[col].mean(), inplace=True) # Fill with mean for climate data


        # Remove extreme outliers based on cleaned SWL
        swl = df['SWL']
        median_swl = swl.median()
        mad_swl = np.median(np.abs(swl - median_swl))
        # Handle potential zero MAD
        if mad_swl == 0:
             print("Warning: MAD is zero, skipping modified Z-score outlier removal.")
             extreme_outliers = pd.Series([False] * len(df), index=df.index) # No outliers if MAD is 0
        else:
             modified_z_scores = 0.6745 * (swl - median_swl) / mad_swl
             extreme_outliers = np.abs(modified_z_scores) > 3.5

        df = df[~extreme_outliers].reset_index(drop=True)
        print(f"Dataset after outlier removal: {df.shape[0]} rows")

        self.df = df
        return df

    def engineer_features(self):
        features = pd.DataFrame()

        # Basic and derived features
        features['Elevation'] = self.df['Elevation']
        features['latitude'] = self.df['latitude']
        features['longitude'] = self.df['longitude']

        # Compute soil mapping *after* cleaning and outlier removal
        if 'SOIL_TYPE' in self.df.columns and 'SWL' in self.df.columns:
            self.soil_mapping = self.df.groupby('SOIL_TYPE')['SWL'].mean().to_dict()
            features['soil_swl_mean'] = self.df['SOIL_TYPE'].map(self.soil_mapping)
            # Fill NaNs from soil mapping (for types not in training data - though handled by cleaning now)
            # This fillna is less critical if we handle NaNs in new data's map call, but good practice.
            mean_soil_swl_train = pd.Series(self.soil_mapping.values()).mean()
            features['soil_swl_mean'].fillna(mean_soil_swl_train, inplace=True)
        else:
            # Handle case where SOIL_TYPE or SWL is missing
            print("Warning: 'SOIL_TYPE' or 'SWL' column missing. Cannot create 'soil_swl_mean' feature.")
            features['soil_swl_mean'] = 0 # Add a placeholder column


        # Aggregated climate features (adjust column names as needed)
        precip_cols = [col for col in self.df.columns if 'Precip' in col]
        temp_cols = [col for col in self.df.columns if 'LSTDay' in col]
        humidity_cols = [col for col in self.df.columns if 'Humidity' in col]

        # Ensure climate columns exist before aggregating
        valid_precip_cols = [col for col in precip_cols if col in self.df.columns]
        valid_temp_cols = [col for col in temp_cols if col in self.df.columns]
        valid_humidity_cols = [col for col in humidity_cols if col in self.df.columns]

        if valid_precip_cols:
            features['precip_annual_total'] = self.df[valid_precip_cols].sum(axis=1)
            mean_precip = self.df[valid_precip_cols].mean(axis=1)
            features['precip_seasonality'] = self.df[valid_precip_cols].std(axis=1) / (mean_precip + 1e-6) # Avoid division by zero
        else:
             features['precip_annual_total'] = 0
             features['precip_seasonality'] = 0

        if valid_temp_cols:
            features['temp_annual_mean'] = self.df[valid_temp_cols].mean(axis=1)
        else:
            features['temp_annual_mean'] = 0

        if valid_humidity_cols:
            features['humidity_annual_mean'] = self.df[valid_humidity_cols].mean(axis=1)
        else:
            features['humidity_annual_mean'] = 0

        # Interaction features - ensure components exist
        features['elevation_x_precip'] = features['Elevation'] * features['precip_annual_total']
        features['temp_x_humidity'] = features['temp_annual_mean'] * features['humidity_annual_mean']

        # Store the column names of the engineered features for later use in prediction
        self.training_feature_columns = features.columns.tolist()
        joblib.dump(self.training_feature_columns, 'training_feature_columns.pkl')
        print("Saved training_feature_columns.pkl")


        print(f"Created {features.shape[1]} features")
        # Return features and the target variable
        return features, self.df['SWL']


    def transform_target(self, y_train, y_test):
        transformer = PowerTransformer(method='yeo-johnson')
        # Reshape y_train and y_test to be 2D arrays for the transformer
        y_train_transformed = transformer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_transformed = transformer.transform(y_test.values.reshape(-1, 1)).ravel()
        self.target_transformer = transformer

        # Save target_transformer
        joblib.dump(self.target_transformer, 'target_transformer.pkl')
        print("Saved target_transformer.pkl")

        return y_train_transformed, y_test_transformed

    def select_features(self, X_train, y_train):
        selector = RFE(estimator=GradientBoostingRegressor(random_state=42), n_features_to_select=10)
        selector.fit(X_train, y_train)
        self.selected_features = X_train.columns[selector.support_].tolist()

        # Save selected_features
        joblib.dump(self.selected_features, 'selected_features.pkl')
        print("Saved selected_features.pkl")

        return self.selected_features

    def train_and_evaluate(self, X, y):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Save soil_mapping *after* splitting if it was created in engineer_features
        if self.soil_mapping:
             joblib.dump(self.soil_mapping, 'soil_mapping.pkl')
             print("Saved soil_mapping.pkl")

        # Transform target
        y_train_transformed, y_test_transformed = self.transform_target(y_train, y_test)

        # Scale features
        self.feature_scaler = RobustScaler()
        # Ensure X_train is a pandas DataFrame for column name handling
        if not isinstance(X_train, pd.DataFrame):
             X_train = pd.DataFrame(X_train, columns=self.training_feature_columns)

        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test) # Use the original X_test which might be numpy

        # Save feature_scaler
        joblib.dump(self.feature_scaler, 'scaler.pkl')
        print("Saved scaler.pkl")

        # Feature selection
        # RFE needs a DataFrame to select columns by name
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        self.selected_features = self.select_features(X_train_scaled_df, y_train_transformed)
        X_train_selected = X_train_scaled_df[self.selected_features].values

        # Apply feature selection to the test set
        # Need to ensure X_test is also a DataFrame before selecting columns
        if not isinstance(X_test, pd.DataFrame):
            X_test_df = pd.DataFrame(X_test_scaled, columns=X_train.columns) # Use training columns
        else:
            X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns) # Use test columns (assuming same as train)

        X_test_selected = X_test_df[self.selected_features].values


        # Train optimized model
        self.model = GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=5,
            min_samples_split=10, min_samples_leaf=5, subsample=0.8, random_state=42
        )

        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2 = cross_val_score(self.model, X_train_selected, y_train_transformed, cv=cv, scoring='r2')
        print(f"Cross-validated R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

        # Fit and predict
        self.model.fit(X_train_selected, y_train_transformed)

        # Save model
        joblib.dump(self.model, 'model.pkl')
        print("Saved model.pkl")

        y_pred_transformed = self.model.predict(X_test_selected)
        y_pred = self.target_transformer.inverse_transform(y_pred_transformed.reshape(-1, 1)).ravel()

        # Evaluate
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.2f}m")

        return y_pred

    def run_pipeline(self):
        self.load_and_clean_data()
        X, y = self.engineer_features()
        predictions = self.train_and_evaluate(X, y)
        return predictions

# --- Main execution block for training and saving artifacts ---
if __name__ == "__main__":
    # Run the training pipeline first to generate the necessary artifacts
    print("--- Training Model and Saving Artifacts ---")
    model_trainer = GroundwaterModel('Sidama_dataset_for_model.csv')
    training_predictions = model_trainer.run_pipeline()
    print("--- Training Complete ---")

    # Create and save test data (assuming this is independent of training run)
    # You might load the original full dataset again here
    df_full = pd.read_csv("Sidama_dataset_for_model.csv")
    # Take a sample that wasn't necessarily in the training set (if using the same file)
    # Or, if "[Clean]SidamaGridPointsDataForPrediction.csv" is truly new data, just load it.
    # For this example, let's simulate creating test data from the full dataset.
    # A better approach for production is to train on ALL available labeled data
    # and predict on completely new, unlabeled data.
    print("\n--- Preparing Test Data ---")
    # Load the dataset again to ensure we sample from the original structure
    df_original = pd.read_csv("Sidama_dataset_for_model.csv")
    # Sample rows, ideally ensuring they are distinct from training rows
    # A robust way is to split *before* training, or load a completely separate file.
    # Assuming for this example, the sample is representative of prediction data structure.
    df_test = df_original.sample(frac=0.25, random_state=42)
    # Keep all columns needed for feature engineering *except* 'SWL'
    # We need all original columns used in `engineer_features` for prediction
    df_test_prediction_input = df_test.drop(columns=['SWL'], errors='ignore') # Drop 'SWL' if it exists

    # Save the prediction input data
    prediction_input_filename = "[Clean]SidamaGridPointsDataForPrediction.csv"
    df_test_prediction_input.to_csv(prediction_input_filename, index=False)
    print(f"Saved prediction input data to '{prediction_input_filename}'")

    # --- Now, use the prediction function with the saved artifacts ---
    print("\n--- Generating Predictions ---")
    # Load the prediction input data
    test_data_for_prediction = pd.read_csv(prediction_input_filename)

    # Generate predictions using the predict_swl function and saved artifacts
    predicted_swl = predict_swl(test_data_for_prediction)

    # Create a new DataFrame with the original test data and predictions
    output_df = test_data_for_prediction.copy()
    output_df['Predicted_SWL'] = predicted_swl

    # Save the results to a CSV file
    output_filename = "SidamaGridPoints_Predicted.csv"
    output_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to '{output_filename}'")