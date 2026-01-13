"""
B2B Sales Data Generator
========================
Generates synthetic but realistic B2B sales data for:
1. Lead Scoring Model
2. Churn Prediction Model

The data generation follows realistic B2B sales patterns and includes
domain-specific knowledge about lead quality and customer health indicators.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class B2BSalesDataGenerator:
    """
    Generates synthetic B2B sales data with realistic distributions
    and correlations based on domain knowledge.
    """

    def __init__(self, random_state=42):
        """
        Initialize the data generator.

        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def generate_lead_data(self, n_samples=1000):
        """
        Generate synthetic lead scoring data.

        Domain Knowledge Applied:
        - Referral leads convert better than cold calls
        - Reaching decision makers significantly improves conversion
        - Enterprise companies (larger) are more valuable but harder to close
        - Email engagement is a strong indicator

        Args:
            n_samples (int): Number of lead records to generate

        Returns:
            pd.DataFrame: Lead data with features and target variable
        """
        data = {}

        # Company size: Small (1-50), Medium (51-500), Large (501-5000), Enterprise (5000+)
        size_categories = ['Small', 'Medium', 'Large', 'Enterprise']
        size_weights = [0.4, 0.35, 0.2, 0.05]  # Most leads are SMB
        data['company_size'] = np.random.choice(size_categories, n_samples, p=size_weights)

        # Lead source with realistic distribution
        sources = ['Cold Call', 'Webinar', 'Referral', 'Content Download', 'Conference']
        source_weights = [0.25, 0.20, 0.15, 0.30, 0.10]
        data['lead_source'] = np.random.choice(sources, n_samples, p=source_weights)

        # Decision maker reached (critical in B2B)
        data['decision_maker_reached'] = np.random.choice([True, False], n_samples, p=[0.3, 0.7])

        # Email interaction score (0-10): how engaged are they with emails?
        data['email_interaction_score'] = np.random.randint(0, 11, n_samples)

        # Days since first contact
        data['days_since_first_contact'] = np.random.randint(1, 180, n_samples)

        # Budget indicated (Boolean)
        data['budget_confirmed'] = np.random.choice([True, False], n_samples, p=[0.4, 0.6])

        # Number of touchpoints
        data['touchpoints_count'] = np.random.randint(1, 20, n_samples)

        df = pd.DataFrame(data)

        # Generate target variable with realistic correlations
        conversion_probability = self._calculate_lead_conversion_probability(df)
        df['converted'] = (np.random.random(n_samples) < conversion_probability).astype(int)

        return df

    def _calculate_lead_conversion_probability(self, df):
        """
        Calculate conversion probability based on feature values.
        This encodes domain knowledge about what makes leads convert.

        Args:
            df (pd.DataFrame): Lead features

        Returns:
            np.array: Conversion probabilities for each lead
        """
        # Base probability
        prob = np.full(len(df), 0.15)

        # Lead source impact (Referrals are gold in B2B!)
        prob += (df['lead_source'] == 'Referral').values * 0.35
        prob += (df['lead_source'] == 'Webinar').values * 0.20
        prob += (df['lead_source'] == 'Conference').values * 0.15
        prob -= (df['lead_source'] == 'Cold Call').values * 0.05

        # Decision maker reached is crucial
        prob += df['decision_maker_reached'].values * 0.30

        # Email engagement
        prob += (df['email_interaction_score'] / 10) * 0.20

        # Budget confirmed
        prob += df['budget_confirmed'].values * 0.25

        # Touchpoints (but diminishing returns after too many)
        touchpoint_factor = np.minimum(df['touchpoints_count'] / 15, 1.0) * 0.15
        prob += touchpoint_factor

        # Company size effect (Enterprise harder to close but we still want some)
        prob -= (df['company_size'] == 'Enterprise').values * 0.10
        prob += (df['company_size'] == 'Medium').values * 0.05

        # Time decay: leads get stale
        prob -= (df['days_since_first_contact'] > 90).values * 0.15

        # Clip to valid probability range
        return np.clip(prob, 0, 1)

    def generate_customer_data(self, n_samples=800):
        """
        Generate synthetic customer churn data.

        Domain Knowledge Applied:
        - Low NPS scores strongly correlate with churn
        - Support ticket volume indicates problems
        - Product usage is the leading indicator of success
        - Time since last login shows engagement

        Args:
            n_samples (int): Number of customer records to generate

        Returns:
            pd.DataFrame: Customer data with features and churn target
        """
        data = {}

        # Customer ID
        data['customer_id'] = [f'CUST_{i:05d}' for i in range(n_samples)]

        # Net Promoter Score (0-10): Critical health metric
        data['nps_score'] = np.random.randint(0, 11, n_samples)

        # Support tickets in last month
        data['support_tickets_last_month'] = np.random.poisson(2, n_samples)

        # Contract value (annual, in thousands)
        data['contract_value'] = np.random.lognormal(10, 1.5, n_samples).astype(int)

        # Days since last login
        data['last_login_days_ago'] = np.random.gamma(2, 5, n_samples).astype(int)

        # Feature usage index (0-100): percentage of available features used
        data['feature_usage_index'] = np.random.randint(0, 101, n_samples)

        # Months as customer
        data['tenure_months'] = np.random.randint(1, 60, n_samples)

        # Number of active users at the account
        data['active_users_count'] = np.random.randint(1, 50, n_samples)

        # Contract type
        data['contract_type'] = np.random.choice(
            ['Monthly', 'Annual', 'Multi-Year'],
            n_samples,
            p=[0.4, 0.45, 0.15]
        )

        # Has dedicated CSM (Customer Success Manager)
        data['has_csm'] = np.random.choice([True, False], n_samples, p=[0.3, 0.7])

        df = pd.DataFrame(data)

        # Generate churn target with realistic correlations
        churn_probability = self._calculate_churn_probability(df)
        df['churned'] = (np.random.random(n_samples) < churn_probability).astype(int)

        return df

    def _calculate_churn_probability(self, df):
        """
        Calculate churn probability based on customer health indicators.
        This encodes domain knowledge about customer success patterns.

        Args:
            df (pd.DataFrame): Customer features

        Returns:
            np.array: Churn probabilities for each customer
        """
        # Base churn rate
        prob = np.full(len(df), 0.15)

        # NPS Score (strongest predictor of churn)
        # Detractors (0-6): high churn, Passives (7-8): medium, Promoters (9-10): low
        prob += (df['nps_score'] <= 6).values * 0.40
        prob += ((df['nps_score'] == 7) | (df['nps_score'] == 8)).values * 0.10
        prob -= (df['nps_score'] >= 9).values * 0.25

        # Support tickets (too many = problems)
        prob += (df['support_tickets_last_month'] > 5).values * 0.25
        prob += ((df['support_tickets_last_month'] >= 3) &
                (df['support_tickets_last_month'] <= 5)).values * 0.10

        # Last login (engagement metric)
        prob += (df['last_login_days_ago'] > 30).values * 0.35
        prob += ((df['last_login_days_ago'] > 14) &
                (df['last_login_days_ago'] <= 30)).values * 0.15

        # Feature usage (low usage = no value realization)
        prob += (df['feature_usage_index'] < 20).values * 0.30
        prob -= (df['feature_usage_index'] > 70).values * 0.20

        # Contract value (higher value customers churn less due to more attention)
        prob -= (df['contract_value'] > 50).values * 0.10

        # Tenure (early customers more likely to churn)
        prob += (df['tenure_months'] < 6).values * 0.20
        prob -= (df['tenure_months'] > 24).values * 0.15

        # Contract type (monthly easier to churn)
        prob += (df['contract_type'] == 'Monthly').values * 0.20
        prob -= (df['contract_type'] == 'Multi-Year').values * 0.15

        # CSM assigned (reduces churn)
        prob -= df['has_csm'].values * 0.15

        # Active users (more users = more stickiness)
        prob -= (df['active_users_count'] > 10).values * 0.10

        return np.clip(prob, 0, 1)

    def save_data(self, output_dir='data'):
        """
        Generate and save both datasets to CSV files.

        Args:
            output_dir (str): Directory to save the CSV files
        """
        print("Generating B2B Sales Data...")
        print("=" * 50)

        # Generate lead data
        print("\n1. Generating Lead Scoring Data...")
        lead_df = self.generate_lead_data(n_samples=1000)
        lead_path = f"{output_dir}/leads.csv"
        lead_df.to_csv(lead_path, index=False)
        print(f"   ✓ Saved {len(lead_df)} leads to {lead_path}")
        print(f"   Conversion Rate: {lead_df['converted'].mean():.1%}")
        print(f"   Features: {', '.join(lead_df.columns.drop('converted'))}")

        # Generate customer data
        print("\n2. Generating Customer Churn Data...")
        customer_df = self.generate_customer_data(n_samples=800)
        customer_path = f"{output_dir}/customers.csv"
        customer_df.to_csv(customer_path, index=False)
        print(f"   ✓ Saved {len(customer_df)} customers to {customer_path}")
        print(f"   Churn Rate: {customer_df['churned'].mean():.1%}")
        print(f"   Features: {', '.join(customer_df.columns.drop(['customer_id', 'churned']))}")

        print("\n" + "=" * 50)
        print("Data generation complete!")

        return lead_df, customer_df


if __name__ == "__main__":
    # Generate the data
    generator = B2BSalesDataGenerator(random_state=42)
    generator.save_data(output_dir='data')
