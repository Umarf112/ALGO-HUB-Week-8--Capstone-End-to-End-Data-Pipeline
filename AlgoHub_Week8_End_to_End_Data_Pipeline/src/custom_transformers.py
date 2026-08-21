"""
Custom Scikit-Learn Transformers for Telco Data Preprocessing Pipeline.
All transformers inherit from BaseEstimator and TransformerMixin.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class IDColumnDropper(BaseEstimator, TransformerMixin):
    """
    Drops non-predictive identifier columns (e.g. customerID) if present.
    """
    def __init__(self, id_cols=None):
        self.id_cols = id_cols if id_cols is not None else ["customerID"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        cols_to_drop = [c for c in self.id_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        return df


class DataTypeCoercer(BaseEstimator, TransformerMixin):
    """
    Coerces target string numeric columns (e.g., TotalCharges with space strings ' ')
    into proper float64 dtype, converting non-numeric values to NaN.
    """
    def __init__(self, target_cols=None):
        self.target_cols = target_cols if target_cols is not None else ["TotalCharges"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        if isinstance(X_out, pd.DataFrame):
            for col in self.target_cols:
                if col in X_out.columns:
                    X_out[col] = pd.to_numeric(X_out[col], errors="coerce")
        elif isinstance(X_out, np.ndarray):
            # If input is ndarray, convert to DF and coerce
            df = pd.DataFrame(X_out)
            for col in self.target_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            X_out = df
        return X_out


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handles missing values (NaNs) in numeric and categorical columns.
    Learns medians for numeric and modes for categorical during fit().
    """
    def __init__(self, numeric_cols=None, categorical_cols=None):
        self.numeric_cols = numeric_cols if numeric_cols is not None else []
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.numeric_medians_ = {}
        self.categorical_modes_ = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col in self.numeric_cols:
            if col in df.columns:
                valid = pd.to_numeric(df[col], errors="coerce").dropna()
                self.numeric_medians_[col] = float(valid.median()) if len(valid) > 0 else 0.0

        for col in self.categorical_cols:
            if col in df.columns:
                valid = df[col].dropna()
                mode_val = valid.mode().iloc[0] if len(valid) > 0 else "Unknown"
                self.categorical_modes_[col] = str(mode_val)
                
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col, median_val in self.numeric_medians_.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median_val)

        for col, mode_val in self.categorical_modes_.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode_val).astype(str)

        return df


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Applies IQR-based Winsorization (capping) to numeric columns.
    Caps values outside [Q1 - factor*IQR, Q3 + factor*IQR].
    """
    def __init__(self, numeric_cols=None, factor=1.5):
        self.numeric_cols = numeric_cols if numeric_cols is not None else []
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col in self.numeric_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(series) > 0:
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - self.factor * iqr
                    upper_bound = q3 + self.factor * iqr
                    self.bounds_[col] = (lower_bound, upper_bound)
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col, (lower, upper) in self.bounds_.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=lower, upper=upper)
        return df


class FeatureEngTransformer(BaseEstimator, TransformerMixin):
    """
    Generates domain-specific features for Customer Churn Analysis:
    - TenureYears: tenure in years
    - AvgMonthlyCostPerTenure: TotalCharges / (tenure + 1)
    - HasAddonsCount: count of active internet add-on services
    - IsLongTermContract: 1 if contract != 'Month-to-month' else 0
    - ElectronicPayment: 1 if payment method is Electronic check else 0
    """
    def __init__(self):
        self.addon_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Tenure in years
        if "tenure" in df.columns:
            df["TenureYears"] = df["tenure"] / 12.0
            
        # Average monthly cost based on total charges and tenure
        if "TotalCharges" in df.columns and "tenure" in df.columns:
            tot_charges_num = pd.to_numeric(df["TotalCharges"], errors="coerce")
            tenure_num = pd.to_numeric(df["tenure"], errors="coerce")
            df["AvgMonthlyCostPerTenure"] = tot_charges_num / (tenure_num + 1.0)
            
        # Addon services count
        existing_addons = [c for c in self.addon_cols if c in df.columns]
        if existing_addons:
            df["HasAddonsCount"] = (df[existing_addons] == "Yes").sum(axis=1)
        else:
            df["HasAddonsCount"] = 0

        # Contract Type binary indicator
        if "Contract" in df.columns:
            df["IsLongTermContract"] = (df["Contract"] != "Month-to-month").astype(int)

        # Payment method risk indicator
        if "PaymentMethod" in df.columns:
            df["ElectronicPayment"] = (df["PaymentMethod"] == "Electronic check").astype(int)

        return df


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    One-Hot encodes categorical columns using scikit-learn's OneHotEncoder.
    Maintains clean feature names and returns a pandas DataFrame.
    """
    def __init__(self, categorical_cols=None):
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.feature_names_ = []

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        existing_cats = [c for c in self.categorical_cols if c in df.columns]
        
        if existing_cats:
            self.encoder_.fit(df[existing_cats].astype(str))
            self.feature_names_ = self.encoder_.get_feature_names_out(existing_cats).tolist()
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        existing_cats = [c for c in self.categorical_cols if c in df.columns]
        non_cat_cols = [c for c in df.columns if c not in existing_cats]
        
        df_non_cat = df[non_cat_cols].reset_index(drop=True)
        
        if existing_cats and hasattr(self.encoder_, "categories_"):
            encoded_arr = self.encoder_.transform(df[existing_cats].astype(str))
            df_encoded = pd.DataFrame(encoded_arr, columns=self.feature_names_)
            return pd.concat([df_non_cat, df_encoded], axis=1)
        
        return df_non_cat


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Scales specified numerical features using StandardScaler while preserving column names.
    """
    def __init__(self, numeric_cols=None):
        self.numeric_cols = numeric_cols if numeric_cols is not None else []
        self.scaler_ = StandardScaler()
        self.scaled_cols_ = []

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        self.scaled_cols_ = [c for c in self.numeric_cols if c in df.columns]
        
        if self.scaled_cols_:
            self.scaler_.fit(df[self.scaled_cols_])
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        if self.scaled_cols_:
            scaled_arr = self.scaler_.transform(df[self.scaled_cols_])
            df[self.scaled_cols_] = scaled_arr
        return df
