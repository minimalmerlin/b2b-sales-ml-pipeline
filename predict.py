"""
B2B Sales Prediction & Inference
=================================
Load trained models and make predictions on new data.
Generates prioritized action lists for sales and customer success teams.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime


class SalesPredictor:
    """
    Production inference system for B2B sales predictions.
    Provides actionable insights for sales and customer success teams.
    """

    def __init__(self, models_dir='models'):
        """
        Initialize the predictor with trained models.

        Args:
            models_dir (str): Directory containing saved model files
        """
        self.models_dir = models_dir
        self.lead_model = None
        self.churn_model = None
        self._load_models()

    def _load_models(self):
        """Load the trained models from disk."""
        lead_path = f"{self.models_dir}/lead_scoring_model.joblib"
        churn_path = f"{self.models_dir}/churn_prediction_model.joblib"

        try:
            self.lead_model = joblib.load(lead_path)
            self.churn_model = joblib.load(churn_path)
            print("✅ Models loaded successfully!")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model files not found. Please train models first using pipeline.py\n{e}"
            )

    def predict_lead_scores(self, leads_df):
        """
        Score leads and return prioritized list.

        Args:
            leads_df (pd.DataFrame): DataFrame with lead features

        Returns:
            pd.DataFrame: Leads with conversion probabilities and priorities
        """
        # Make predictions
        conversion_proba = self.lead_model.predict_proba(leads_df)[:, 1]

        # Create results dataframe
        results = leads_df.copy()
        results['conversion_probability'] = conversion_proba
        results['conversion_probability_pct'] = (conversion_proba * 100).round(1)

        # Assign priority tiers
        results['priority'] = pd.cut(
            conversion_proba,
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )

        # Sort by probability
        results = results.sort_values('conversion_probability', ascending=False)

        return results

    def predict_churn_risk(self, customers_df):
        """
        Predict customer churn risk and return prioritized list.

        Args:
            customers_df (pd.DataFrame): DataFrame with customer features

        Returns:
            pd.DataFrame: Customers with churn probabilities and risk levels
        """
        # Separate customer_id if present
        customer_ids = None
        if 'customer_id' in customers_df.columns:
            customer_ids = customers_df['customer_id']
            features_df = customers_df.drop('customer_id', axis=1)
        else:
            features_df = customers_df

        # Make predictions
        churn_proba = self.churn_model.predict_proba(features_df)[:, 1]

        # Create results dataframe
        results = features_df.copy()
        if customer_ids is not None:
            results.insert(0, 'customer_id', customer_ids.values)

        results['churn_probability'] = churn_proba
        results['churn_probability_pct'] = (churn_proba * 100).round(1)

        # Assign risk levels
        results['risk_level'] = pd.cut(
            churn_proba,
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        )

        # Sort by probability
        results = results.sort_values('churn_probability', ascending=False)

        return results

    def generate_high_priority_leads(self, leads_df, threshold=0.80):
        """
        Generate a focused list of high-priority leads.

        Args:
            leads_df (pd.DataFrame): Lead features
            threshold (float): Minimum probability threshold

        Returns:
            pd.DataFrame: High-priority leads with recommended actions
        """
        scored_leads = self.predict_lead_scores(leads_df)
        high_priority = scored_leads[scored_leads['conversion_probability'] >= threshold].copy()

        # Add recommended actions based on features
        high_priority['recommended_action'] = high_priority.apply(
            self._recommend_lead_action, axis=1
        )

        return high_priority

    def generate_at_risk_customers(self, customers_df, threshold=0.70):
        """
        Generate a list of at-risk customers requiring intervention.

        Args:
            customers_df (pd.DataFrame): Customer features
            threshold (float): Minimum churn probability threshold

        Returns:
            pd.DataFrame: At-risk customers with recommended interventions
        """
        scored_customers = self.predict_churn_risk(customers_df)
        at_risk = scored_customers[scored_customers['churn_probability'] >= threshold].copy()

        # Add recommended interventions based on features
        at_risk['recommended_intervention'] = at_risk.apply(
            self._recommend_churn_intervention, axis=1
        )

        return at_risk

    def _recommend_lead_action(self, row):
        """
        Recommend specific action for a lead based on their profile.

        Args:
            row (pd.Series): Lead data

        Returns:
            str: Recommended action
        """
        actions = []

        if not row.get('decision_maker_reached', False):
            actions.append("Schedule executive meeting")

        if not row.get('budget_confirmed', False):
            actions.append("Discuss budget & ROI")

        if row.get('email_interaction_score', 0) < 5:
            actions.append("Increase engagement (webinar/demo)")

        if row.get('days_since_first_contact', 0) > 60:
            actions.append("Fast-track - lead aging")

        if row.get('lead_source') == 'Referral':
            actions.append("Leverage referral relationship")

        if not actions:
            actions.append("Move to proposal stage")

        return " | ".join(actions)

    def _recommend_churn_intervention(self, row):
        """
        Recommend intervention strategy for at-risk customer.

        Args:
            row (pd.Series): Customer data

        Returns:
            str: Recommended intervention
        """
        interventions = []

        if row.get('nps_score', 10) <= 6:
            interventions.append("NPS follow-up call - address concerns")

        if row.get('support_tickets_last_month', 0) > 5:
            interventions.append("Escalate to senior support")

        if row.get('last_login_days_ago', 0) > 30:
            interventions.append("Re-engagement campaign + training")

        if row.get('feature_usage_index', 100) < 30:
            interventions.append("Value realization workshop")

        if row.get('contract_type') == 'Monthly':
            interventions.append("Offer annual contract incentive")

        if not row.get('has_csm', False):
            interventions.append("Assign dedicated CSM")

        if not interventions:
            interventions.append("Executive business review")

        return " | ".join(interventions)

    def export_predictions(self, leads_df=None, customers_df=None, output_dir='outputs'):
        """
        Generate prediction reports and save to CSV.

        Args:
            leads_df (pd.DataFrame): Lead data for scoring
            customers_df (pd.DataFrame): Customer data for churn prediction
            output_dir (str): Directory to save output files

        Returns:
            dict: Paths to generated reports
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        reports = {}

        print("\n" + "=" * 60)
        print("GENERATING PREDICTION REPORTS")
        print("=" * 60)

        # Lead Scoring Report
        if leads_df is not None:
            print("\n📊 Lead Scoring Analysis...")
            scored_leads = self.predict_lead_scores(leads_df)
            high_priority = self.generate_high_priority_leads(leads_df, threshold=0.80)

            # Save all scored leads
            all_leads_path = f"{output_dir}/all_leads_scored_{timestamp}.csv"
            scored_leads.to_csv(all_leads_path, index=False)
            reports['all_leads'] = all_leads_path

            # Save high-priority leads
            priority_leads_path = f"{output_dir}/high_priority_leads_{timestamp}.csv"
            high_priority.to_csv(priority_leads_path, index=False)
            reports['high_priority_leads'] = priority_leads_path

            print(f"   ✓ Scored {len(scored_leads)} leads")
            print(f"   ✓ {len(high_priority)} high-priority leads (>80% probability)")
            print(f"   Saved to: {all_leads_path}")
            print(f"   Saved to: {priority_leads_path}")

        # Churn Prediction Report
        if customers_df is not None:
            print("\n📊 Churn Risk Analysis...")
            scored_customers = self.predict_churn_risk(customers_df)
            at_risk = self.generate_at_risk_customers(customers_df, threshold=0.70)

            # Save all scored customers
            all_customers_path = f"{output_dir}/all_customers_scored_{timestamp}.csv"
            scored_customers.to_csv(all_customers_path, index=False)
            reports['all_customers'] = all_customers_path

            # Save at-risk customers
            at_risk_path = f"{output_dir}/at_risk_customers_{timestamp}.csv"
            at_risk.to_csv(at_risk_path, index=False)
            reports['at_risk_customers'] = at_risk_path

            print(f"   ✓ Scored {len(scored_customers)} customers")
            print(f"   ✓ {len(at_risk)} at-risk customers (>70% churn probability)")
            print(f"   Saved to: {all_customers_path}")
            print(f"   Saved to: {at_risk_path}")

        print("\n" + "=" * 60)
        print("✅ Reports generated successfully!")
        print("=" * 60 + "\n")

        return reports

    def print_summary(self, leads_df=None, customers_df=None):
        """
        Print executive summary of predictions.

        Args:
            leads_df (pd.DataFrame): Lead data
            customers_df (pd.DataFrame): Customer data
        """
        print("\n" + "=" * 60)
        print("EXECUTIVE SUMMARY")
        print("=" * 60)

        if leads_df is not None:
            scored_leads = self.predict_lead_scores(leads_df)
            high_priority = scored_leads[scored_leads['conversion_probability'] >= 0.80]
            medium_priority = scored_leads[
                (scored_leads['conversion_probability'] >= 0.60) &
                (scored_leads['conversion_probability'] < 0.80)
            ]

            print("\n🎯 LEAD SCORING INSIGHTS:")
            print(f"   Total Leads Analyzed: {len(scored_leads)}")
            print(f"   High Priority (>80%): {len(high_priority)} leads")
            print(f"   Medium Priority (60-80%): {len(medium_priority)} leads")
            print(f"   Average Conversion Probability: {scored_leads['conversion_probability'].mean():.1%}")

            if len(high_priority) > 0:
                print(f"\n   🔥 Top 3 Hottest Leads:")
                for idx, (_, lead) in enumerate(high_priority.head(3).iterrows(), 1):
                    print(f"      {idx}. {lead.get('company_size', 'N/A')} company "
                          f"from {lead.get('lead_source', 'N/A')} - "
                          f"{lead['conversion_probability_pct']}% probability")

        if customers_df is not None:
            scored_customers = self.predict_churn_risk(customers_df)
            critical_risk = scored_customers[scored_customers['churn_probability'] >= 0.80]
            high_risk = scored_customers[
                (scored_customers['churn_probability'] >= 0.70) &
                (scored_customers['churn_probability'] < 0.80)
            ]

            print("\n⚠️  CHURN RISK INSIGHTS:")
            print(f"   Total Customers Analyzed: {len(scored_customers)}")
            print(f"   Critical Risk (>80%): {len(critical_risk)} customers")
            print(f"   High Risk (70-80%): {len(high_risk)} customers")
            print(f"   Average Churn Probability: {scored_customers['churn_probability'].mean():.1%}")

            if len(critical_risk) > 0:
                print(f"\n   🚨 Top 3 Critical Risk Customers:")
                for idx, (_, customer) in enumerate(critical_risk.head(3).iterrows(), 1):
                    cust_id = customer.get('customer_id', 'N/A')
                    nps = customer.get('nps_score', 'N/A')
                    print(f"      {idx}. {cust_id} - NPS: {nps} - "
                          f"{customer['churn_probability_pct']}% churn risk")

        print("\n" + "=" * 60 + "\n")


def main():
    """
    Main prediction pipeline - demonstrates inference on new data.
    """
    print("\n" + "=" * 60)
    print("B2B SALES PREDICTION SYSTEM")
    print("=" * 60)

    # Initialize predictor
    predictor = SalesPredictor(models_dir='models')

    # Load test data (in production, this would be new leads/customers)
    print("\n📂 Loading data for predictions...")

    try:
        leads_df = pd.read_csv('data/leads.csv').drop('converted', axis=1)
        customers_df = pd.read_csv('data/customers.csv').drop('churned', axis=1)
        print("   ✓ Data loaded successfully")
    except FileNotFoundError:
        print("   ⚠️  Data files not found. Please run data_gen.py first.")
        return

    # Generate predictions and reports
    reports = predictor.export_predictions(
        leads_df=leads_df,
        customers_df=customers_df,
        output_dir='outputs'
    )

    # Print executive summary
    predictor.print_summary(leads_df=leads_df, customers_df=customers_df)

    # Show sample high-priority leads
    high_priority_leads = predictor.generate_high_priority_leads(leads_df, threshold=0.80)
    if len(high_priority_leads) > 0:
        print("=" * 60)
        print("SAMPLE: HIGH-PRIORITY LEADS (Top 5)")
        print("=" * 60)
        display_cols = ['company_size', 'lead_source', 'decision_maker_reached',
                       'conversion_probability_pct', 'recommended_action']
        available_cols = [col for col in display_cols if col in high_priority_leads.columns]
        print(high_priority_leads[available_cols].head().to_string(index=False))

    # Show sample at-risk customers
    at_risk_customers = predictor.generate_at_risk_customers(customers_df, threshold=0.70)
    if len(at_risk_customers) > 0:
        print("\n" + "=" * 60)
        print("SAMPLE: AT-RISK CUSTOMERS (Top 5)")
        print("=" * 60)
        display_cols = ['customer_id', 'nps_score', 'support_tickets_last_month',
                       'churn_probability_pct', 'recommended_intervention']
        available_cols = [col for col in display_cols if col in at_risk_customers.columns]
        print(at_risk_customers[available_cols].head().to_string(index=False))

    print("\n" + "=" * 60)
    print("✅ Prediction pipeline complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
