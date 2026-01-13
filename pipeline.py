"""
B2B Sales ML Pipeline
=====================
Production-ready ML pipeline for B2B Sales use cases:
1. Lead Scoring Model
2. Churn Prediction Model

Includes preprocessing, training, evaluation, and model persistence.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


class SalesPipeline:
    """
    End-to-end ML pipeline for B2B sales predictions.
    Handles preprocessing, training, and inference for both lead scoring and churn prediction.
    """

    def __init__(self, model_type='gradient_boosting', random_state=42):
        """
        Initialize the sales pipeline.

        Args:
            model_type (str): 'gradient_boosting' or 'random_forest'
            random_state (int): Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.lead_model = None
        self.churn_model = None
        self.lead_preprocessor = None
        self.churn_preprocessor = None

    def _get_classifier(self):
        """
        Get the classifier based on model type.

        Returns:
            sklearn classifier: Configured classifier
        """
        if self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
                subsample=0.8
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                min_samples_split=10,
                min_samples_leaf=5
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _create_lead_preprocessor(self, X):
        """
        Create preprocessing pipeline for lead scoring data.

        Args:
            X (pd.DataFrame): Feature dataframe

        Returns:
            ColumnTransformer: Configured preprocessor
        """
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Remove boolean columns from numerical (will be treated as is)
        boolean_cols = [col for col in X.columns if X[col].dtype == 'bool']
        numerical_cols = [col for col in numerical_cols if col not in boolean_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
            ],
            remainder='passthrough'  # Keep boolean columns as-is
        )

        return preprocessor

    def _create_churn_preprocessor(self, X):
        """
        Create preprocessing pipeline for churn prediction data.

        Args:
            X (pd.DataFrame): Feature dataframe

        Returns:
            ColumnTransformer: Configured preprocessor
        """
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Remove boolean columns from numerical
        boolean_cols = [col for col in X.columns if X[col].dtype == 'bool']
        numerical_cols = [col for col in numerical_cols if col not in boolean_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
            ],
            remainder='passthrough'
        )

        return preprocessor

    def train_lead_scoring_model(self, data_path='data/leads.csv', test_size=0.2):
        """
        Train the lead scoring model.

        Args:
            data_path (str): Path to leads CSV file
            test_size (float): Proportion of data for testing

        Returns:
            dict: Training metrics and results
        """
        print("\n" + "=" * 60)
        print("TRAINING LEAD SCORING MODEL")
        print("=" * 60)

        # Load data
        df = pd.read_csv(data_path)
        print(f"\n📊 Loaded {len(df)} leads")
        print(f"   Conversion rate: {df['converted'].mean():.1%}")

        # Prepare features and target
        X = df.drop('converted', axis=1)
        y = df['converted']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        print(f"\n📦 Train set: {len(X_train)} | Test set: {len(X_test)}")

        # Create preprocessor
        self.lead_preprocessor = self._create_lead_preprocessor(X_train)

        # Create full pipeline
        self.lead_model = Pipeline([
            ('preprocessor', self.lead_preprocessor),
            ('classifier', self._get_classifier())
        ])

        # Train model
        print(f"\n🔧 Training {self.model_type} model...")
        self.lead_model.fit(X_train, y_train)

        # Evaluate
        print("\n📈 Model Performance:")
        print("-" * 60)

        # Training scores
        train_score = self.lead_model.score(X_train, y_train)
        print(f"Training Accuracy: {train_score:.3f}")

        # Test scores
        test_score = self.lead_model.score(X_test, y_test)
        y_pred = self.lead_model.predict(X_test)
        y_pred_proba = self.lead_model.predict_proba(X_test)[:, 1]

        print(f"Test Accuracy: {test_score:.3f}")
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")

        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Conversion', 'Converted']))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n🎯 Confusion Matrix:")
        print(f"   True Negatives: {cm[0][0]:4d} | False Positives: {cm[0][1]:4d}")
        print(f"   False Negatives: {cm[1][0]:4d} | True Positives: {cm[1][1]:4d}")

        # Feature importance
        if hasattr(self.lead_model.named_steps['classifier'], 'feature_importances_'):
            self._print_feature_importance(
                self.lead_model.named_steps['classifier'],
                self.lead_preprocessor,
                X_train
            )

        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': cm
        }

        print("\n✅ Lead Scoring Model Training Complete!")
        return results

    def train_churn_prediction_model(self, data_path='data/customers.csv', test_size=0.2):
        """
        Train the churn prediction model.

        Args:
            data_path (str): Path to customers CSV file
            test_size (float): Proportion of data for testing

        Returns:
            dict: Training metrics and results
        """
        print("\n" + "=" * 60)
        print("TRAINING CHURN PREDICTION MODEL")
        print("=" * 60)

        # Load data
        df = pd.read_csv(data_path)
        print(f"\n📊 Loaded {len(df)} customers")
        print(f"   Churn rate: {df['churned'].mean():.1%}")

        # Prepare features and target (drop customer_id)
        X = df.drop(['churned', 'customer_id'], axis=1)
        y = df['churned']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        print(f"\n📦 Train set: {len(X_train)} | Test set: {len(X_test)}")

        # Create preprocessor
        self.churn_preprocessor = self._create_churn_preprocessor(X_train)

        # Create full pipeline
        self.churn_model = Pipeline([
            ('preprocessor', self.churn_preprocessor),
            ('classifier', self._get_classifier())
        ])

        # Train model
        print(f"\n🔧 Training {self.model_type} model...")
        self.churn_model.fit(X_train, y_train)

        # Evaluate
        print("\n📈 Model Performance:")
        print("-" * 60)

        # Training scores
        train_score = self.churn_model.score(X_train, y_train)
        print(f"Training Accuracy: {train_score:.3f}")

        # Test scores
        test_score = self.churn_model.score(X_test, y_test)
        y_pred = self.churn_model.predict(X_test)
        y_pred_proba = self.churn_model.predict_proba(X_test)[:, 1]

        print(f"Test Accuracy: {test_score:.3f}")
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")

        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n🎯 Confusion Matrix:")
        print(f"   True Negatives: {cm[0][0]:4d} | False Positives: {cm[0][1]:4d}")
        print(f"   False Negatives: {cm[1][0]:4d} | True Positives: {cm[1][1]:4d}")

        # Feature importance
        if hasattr(self.churn_model.named_steps['classifier'], 'feature_importances_'):
            self._print_feature_importance(
                self.churn_model.named_steps['classifier'],
                self.churn_preprocessor,
                X_train
            )

        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': cm
        }

        print("\n✅ Churn Prediction Model Training Complete!")
        return results

    def _print_feature_importance(self, classifier, preprocessor, X_sample):
        """
        Print feature importances for tree-based models.

        Args:
            classifier: Trained classifier
            preprocessor: Fitted preprocessor
            X_sample: Sample of training data for feature names
        """
        try:
            # Get feature names after preprocessing
            feature_names = []

            for name, transformer, columns in preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend(columns)
                elif name == 'cat':
                    # Get one-hot encoded feature names
                    if hasattr(transformer, 'get_feature_names_out'):
                        cat_features = transformer.get_feature_names_out(columns)
                        feature_names.extend(cat_features)
                    else:
                        feature_names.extend(columns)
                elif name == 'remainder':
                    # Boolean columns
                    bool_cols = [col for col in X_sample.columns
                               if col not in columns and X_sample[col].dtype == 'bool']
                    feature_names.extend(bool_cols)

            # Get importances
            importances = classifier.feature_importances_

            # Sort by importance
            indices = np.argsort(importances)[::-1][:10]

            print("\n🔍 Top 10 Most Important Features:")
            print("-" * 60)
            for i, idx in enumerate(indices, 1):
                if idx < len(feature_names):
                    print(f"   {i:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")

        except Exception as e:
            print(f"\n⚠️  Could not extract feature importance: {e}")

    def save_models(self, output_dir='models'):
        """
        Save trained models to disk.

        Args:
            output_dir (str): Directory to save model files
        """
        if self.lead_model is None or self.churn_model is None:
            raise ValueError("Models must be trained before saving!")

        lead_path = f"{output_dir}/lead_scoring_model.joblib"
        churn_path = f"{output_dir}/churn_prediction_model.joblib"

        joblib.dump(self.lead_model, lead_path)
        joblib.dump(self.churn_model, churn_path)

        print("\n" + "=" * 60)
        print("💾 Models saved successfully:")
        print(f"   • Lead Scoring: {lead_path}")
        print(f"   • Churn Prediction: {churn_path}")
        print("=" * 60)

    def load_models(self, output_dir='models'):
        """
        Load trained models from disk.

        Args:
            output_dir (str): Directory containing model files
        """
        lead_path = f"{output_dir}/lead_scoring_model.joblib"
        churn_path = f"{output_dir}/churn_prediction_model.joblib"

        self.lead_model = joblib.load(lead_path)
        self.churn_model = joblib.load(churn_path)

        print("\n✅ Models loaded successfully!")


def main():
    """
    Main training pipeline execution.
    """
    print("\n" + "=" * 60)
    print("B2B SALES ML PIPELINE - TRAINING")
    print("=" * 60)

    # Initialize pipeline
    pipeline = SalesPipeline(model_type='gradient_boosting', random_state=42)

    # Train lead scoring model
    lead_results = pipeline.train_lead_scoring_model(
        data_path='data/leads.csv',
        test_size=0.2
    )

    # Train churn prediction model
    churn_results = pipeline.train_churn_prediction_model(
        data_path='data/customers.csv',
        test_size=0.2
    )

    # Save models
    pipeline.save_models(output_dir='models')

    # Summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING SUMMARY")
    print("=" * 60)
    print(f"\n📊 Lead Scoring Model:")
    print(f"   Test Accuracy: {lead_results['test_accuracy']:.3f}")
    print(f"   ROC-AUC: {lead_results['roc_auc']:.3f}")

    print(f"\n📊 Churn Prediction Model:")
    print(f"   Test Accuracy: {churn_results['test_accuracy']:.3f}")
    print(f"   ROC-AUC: {churn_results['roc_auc']:.3f}")

    print("\n" + "=" * 60)
    print("✅ All models trained and saved successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
