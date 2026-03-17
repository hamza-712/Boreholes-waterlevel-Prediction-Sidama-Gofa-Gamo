import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import joblib

def predict_swl(new_data):
    """
    Transform new input data and predict SWL using a trained model.
    
    Parameters:
    new_data (pd.DataFrame): DataFrame with raw features matching the training data structure.
    
    Returns:
    np.ndarray: Predicted SWL values.
    """
    # Load saved artifacts from training
    soil_mapping = joblib.load('soil_mapping.pkl')  # Soil type to mean SWL mapping
    scaler = joblib.load('scaler.pkl')              # Fitted RobustScaler
    selected_features = joblib.load('selected_features.pkl')  # List of selected feature names
    target_transformer = joblib.load('target_transformer.pkl')  # Fitted PowerTransformer
    model = joblib.load('model.pkl')                # Trained GradientBoostingRegressor
    
    # Step 1: Engineer features
    features = pd.DataFrame()
    
    # Basic features
    features['Elevation'] = new_data['Elevation']
    features['latitude'] = new_data['latitude']
    features['longitude'] = new_data['longitude']
    
    # Soil type encoding using the precomputed mapping
    features['soil_swl_mean'] = new_data['SOIL_TYPE'].map(soil_mapping)
    
    # Identify climate-related columns dynamically
    precip_cols = [col for col in new_data.columns if 'Precip' in col]
    temp_cols = [col for col in new_data.columns if 'LSTDay' in col]
    humidity_cols = [col for col in new_data.columns if 'SpecificHumidity' in col]
    
    # Aggregated climate features
    features['precip_annual_total'] = new_data[precip_cols].sum(axis=1)
    features['precip_seasonality'] = new_data[precip_cols].std(axis=1) / (new_data[precip_cols].mean(axis=1) + 1)
    features['temp_annual_mean'] = new_data[temp_cols].mean(axis=1)
    features['humidity_annual_mean'] = new_data[humidity_cols].mean(axis=1)
    
    # Interaction features
    features['elevation_x_precip'] = features['Elevation'] * features['precip_annual_total']
    features['temp_x_humidity'] = features['temp_annual_mean'] * features['humidity_annual_mean']
    
    # Step 2: Scale features using the trained scaler
    features_scaled = scaler.transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
    
    # Step 3: Select the same features as in training
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

    def load_and_clean_data(self):
        df = pd.read_csv(self.data_path)
        print(f"Initial dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Handle missing values
        df['Elevation'].fillna(df['Elevation'].median(), inplace=True)
        
        # Remove extreme outliers
        swl = df['SWL']
        median_swl = swl.median()
        mad_swl = np.median(np.abs(swl - median_swl))
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
        features['soil_swl_mean'] = self.df['SOIL_TYPE'].map(self.df.groupby('SOIL_TYPE')['SWL'].mean())
        
        # Aggregated climate features (adjust column names as needed)
        precip_cols = [col for col in self.df.columns if 'Precip' in col]
        temp_cols = [col for col in self.df.columns if 'LSTDay' in col]
        humidity_cols = [col for col in self.df.columns if 'Humidity' in col]
        
        features['precip_annual_total'] = self.df[precip_cols].sum(axis=1)
        features['precip_seasonality'] = self.df[precip_cols].std(axis=1) / (self.df[precip_cols].mean(axis=1) + 1)
        features['temp_annual_mean'] = self.df[temp_cols].mean(axis=1)
        features['humidity_annual_mean'] = self.df[humidity_cols].mean(axis=1)
        
        # Interaction features
        features['elevation_x_precip'] = features['Elevation'] * features['precip_annual_total']
        features['temp_x_humidity'] = features['temp_annual_mean'] * features['humidity_annual_mean']
        
        print(f"Created {features.shape[1]} features")
        return features, self.df['SWL']

    def transform_target(self, y_train, y_test):
        transformer = PowerTransformer(method='yeo-johnson')
        y_train_transformed = transformer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_transformed = transformer.transform(y_test.values.reshape(-1, 1)).ravel()
        self.target_transformer = transformer
        return y_train_transformed, y_test_transformed

    def select_features(self, X_train, y_train):
        selector = RFE(estimator=GradientBoostingRegressor(random_state=42), n_features_to_select=10)
        selector.fit(X_train, y_train)
        self.selected_features = X_train.columns[selector.support_].tolist()
        return self.selected_features

    def train_and_evaluate(self, X, y):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Transform target
        y_train_transformed, y_test_transformed = self.transform_target(y_train, y_test)
        
        # Scale features
        self.feature_scaler = RobustScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Feature selection
        X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
        self.selected_features = self.select_features(X_train_df, y_train_transformed)
        X_train_selected = X_train_df[self.selected_features].values
        X_test_selected = pd.DataFrame(X_test_scaled, columns=X.columns)[self.selected_features].values
        
        # Train optimized model
        self.model = GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=5, 
            min_samples_split=10, min_samples_leaf=5, subsample=0.8, random_state=42
        )
        
        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2 = cross_val_score(self.model, X_train_selected, y_train_transformed, cv=cv, scoring='r2')
        # print(f"Cross-validated R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
        
        # Fit and predict
        self.model.fit(X_train_selected, y_train_transformed)
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

if __name__ == "__main__":
    model = GroundwaterModel('/content/Sidama_dataset_for_model.csv')  # Replace with your file path
    
    predictions = model.run_pipeline()

    #save model
    import pickle
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    # #load model
    # with open('model.pkl', 'rb') as f:
    #     model = pickle.load(f)




    # Helper function to predict SWL for new data
    # helper function converts input data to the same format as training data
    # sample_data = pd.DataFrame({
    #     'fid': [1],
    #     'ID': [101],
    #     'latitude': [40.7128],
    #     'longitude': [-74.0060],
    #     'Elevation': [10.0],
    #     'SOIL_TYPE': ['Sandy'],
    #     'SpecificHumidity_meanCumOctToJan': [0.005],
    #     'SpecificHumidity_meanCumFebToMay': [0.006],
    #     'SpecificHumidity_meanCumJunToSep': [0.007],
    #     'WindSpeedMeanOctToJan24-25': [3.0],
    #     'WindSpeedMeanFebToMay24-25': [2.5],
    #     'WindSpeedMeanJunToSep24-25': [2.8],
    #     'LSTDayMeanOctToJan23-25': [15.0],
    #     'LSTDayMeanFebToMay24-25': [20.0],
    #     'LSTDayMeanJunToSep24-25': [25.0],
    #     'LSTNightMeanOctToJan23-25': [10.0],
    #     'LSTNightMeanFebToMay23-25': [15.0],
    #     'LSTNightMeanJunToSep23-25': [20.0],
    #     'NDVI_Oct.Jan_2023-2025': [0.3],
    #     'NDVI_Feb.May_2023': [0.4],
    #     'NDVI_Jun.Sep_2024': [0.5],
    #     'Precip_meanCumOctToJan': [100.0],
    #     'Precip_meanCumFebToMay': [150.0],
    #     'Precip_meanCumJunToSep': [200.0]
    # })
    
    # # Predict SWL
    # predictions = predict_swl(sample_data)
    # print("Predicted SWL:", predictions)