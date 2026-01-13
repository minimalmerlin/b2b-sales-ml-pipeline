"""
Example Analysis Script
========================
Demonstrates how to use the trained models for various business analyses.
This script shows practical use cases beyond the basic prediction pipeline.
"""

import pandas as pd
import numpy as np
from predict import SalesPredictor


def analyze_lead_segments():
    """Analyze lead conversion by different segments."""
    print("\n" + "=" * 70)
    print("LEAD SEGMENTATION ANALYSIS")
    print("=" * 70)

    # Load data
    leads = pd.read_csv('data/leads.csv')
    predictor = SalesPredictor(models_dir='models')

    # Score all leads
    leads_features = leads.drop('converted', axis=1)
    scored_leads = predictor.predict_lead_scores(leads_features)
    scored_leads['actual_converted'] = leads['converted'].values

    # Analysis by Lead Source
    print("\n📊 Conversion Analysis by Lead Source:")
    print("-" * 70)

    source_analysis = scored_leads.groupby('lead_source').agg({
        'conversion_probability': 'mean',
        'actual_converted': 'mean',
        'lead_source': 'count'
    }).rename(columns={'lead_source': 'count'})

    source_analysis['predicted_conv_rate'] = (source_analysis['conversion_probability'] * 100).round(1)
    source_analysis['actual_conv_rate'] = (source_analysis['actual_converted'] * 100).round(1)

    print(source_analysis[['count', 'predicted_conv_rate', 'actual_conv_rate']].to_string())

    # Analysis by Company Size
    print("\n📊 Conversion Analysis by Company Size:")
    print("-" * 70)

    size_analysis = scored_leads.groupby('company_size').agg({
        'conversion_probability': 'mean',
        'actual_converted': 'mean',
        'company_size': 'count'
    }).rename(columns={'company_size': 'count'})

    size_analysis['predicted_conv_rate'] = (size_analysis['conversion_probability'] * 100).round(1)
    size_analysis['actual_conv_rate'] = (size_analysis['actual_converted'] * 100).round(1)

    print(size_analysis[['count', 'predicted_conv_rate', 'actual_conv_rate']].to_string())

    # Decision Maker Impact
    print("\n📊 Impact of Reaching Decision Maker:")
    print("-" * 70)

    dm_analysis = scored_leads.groupby('decision_maker_reached').agg({
        'conversion_probability': 'mean',
        'actual_converted': 'mean',
        'decision_maker_reached': 'count'
    }).rename(columns={'decision_maker_reached': 'count'})

    dm_analysis['predicted_conv_rate'] = (dm_analysis['conversion_probability'] * 100).round(1)
    dm_analysis['actual_conv_rate'] = (dm_analysis['actual_converted'] * 100).round(1)
    dm_analysis.index = ['Not Reached', 'Reached']

    print(dm_analysis[['count', 'predicted_conv_rate', 'actual_conv_rate']].to_string())


def analyze_churn_segments():
    """Analyze churn risk by different customer segments."""
    print("\n" + "=" * 70)
    print("CUSTOMER CHURN SEGMENTATION ANALYSIS")
    print("=" * 70)

    # Load data
    customers = pd.read_csv('data/customers.csv')
    predictor = SalesPredictor(models_dir='models')

    # Score all customers
    customers_features = customers.drop('churned', axis=1)
    scored_customers = predictor.predict_churn_risk(customers_features)
    scored_customers['actual_churned'] = customers['churned'].values

    # Analysis by NPS Score
    print("\n📊 Churn Analysis by NPS Score:")
    print("-" * 70)

    # Group NPS into Detractors (0-6), Passives (7-8), Promoters (9-10)
    scored_customers['nps_category'] = pd.cut(
        scored_customers['nps_score'],
        bins=[-1, 6, 8, 10],
        labels=['Detractors (0-6)', 'Passives (7-8)', 'Promoters (9-10)']
    )

    nps_analysis = scored_customers.groupby('nps_category').agg({
        'churn_probability': 'mean',
        'actual_churned': 'mean',
        'nps_category': 'count'
    }).rename(columns={'nps_category': 'count'})

    nps_analysis['predicted_churn_rate'] = (nps_analysis['churn_probability'] * 100).round(1)
    nps_analysis['actual_churn_rate'] = (nps_analysis['actual_churned'] * 100).round(1)

    print(nps_analysis[['count', 'predicted_churn_rate', 'actual_churn_rate']].to_string())

    # Analysis by Contract Type
    print("\n📊 Churn Analysis by Contract Type:")
    print("-" * 70)

    contract_analysis = scored_customers.groupby('contract_type').agg({
        'churn_probability': 'mean',
        'actual_churned': 'mean',
        'contract_type': 'count'
    }).rename(columns={'contract_type': 'count'})

    contract_analysis['predicted_churn_rate'] = (contract_analysis['churn_probability'] * 100).round(1)
    contract_analysis['actual_churn_rate'] = (contract_analysis['actual_churned'] * 100).round(1)

    print(contract_analysis[['count', 'predicted_churn_rate', 'actual_churn_rate']].to_string())

    # Analysis by Feature Usage
    print("\n📊 Churn Analysis by Feature Usage:")
    print("-" * 70)

    scored_customers['usage_category'] = pd.cut(
        scored_customers['feature_usage_index'],
        bins=[0, 30, 60, 100],
        labels=['Low (<30%)', 'Medium (30-60%)', 'High (>60%)']
    )

    usage_analysis = scored_customers.groupby('usage_category').agg({
        'churn_probability': 'mean',
        'actual_churned': 'mean',
        'usage_category': 'count'
    }).rename(columns={'usage_category': 'count'})

    usage_analysis['predicted_churn_rate'] = (usage_analysis['churn_probability'] * 100).round(1)
    usage_analysis['actual_churn_rate'] = (usage_analysis['actual_churned'] * 100).round(1)

    print(usage_analysis[['count', 'predicted_churn_rate', 'actual_churn_rate']].to_string())

    # CSM Impact
    print("\n📊 Impact of CSM Assignment:")
    print("-" * 70)

    csm_analysis = scored_customers.groupby('has_csm').agg({
        'churn_probability': 'mean',
        'actual_churned': 'mean',
        'has_csm': 'count'
    }).rename(columns={'has_csm': 'count'})

    csm_analysis['predicted_churn_rate'] = (csm_analysis['churn_probability'] * 100).round(1)
    csm_analysis['actual_churn_rate'] = (csm_analysis['actual_churned'] * 100).round(1)
    csm_analysis.index = ['No CSM', 'Has CSM']

    print(csm_analysis[['count', 'predicted_churn_rate', 'actual_churn_rate']].to_string())


def calculate_roi_metrics():
    """Calculate business ROI metrics."""
    print("\n" + "=" * 70)
    print("BUSINESS ROI CALCULATION")
    print("=" * 70)

    # Load predictions
    leads = pd.read_csv('data/leads.csv')
    predictor = SalesPredictor(models_dir='models')

    leads_features = leads.drop('converted', axis=1)
    scored_leads = predictor.predict_lead_scores(leads_features)

    # Assumptions (adjust for your business)
    avg_deal_value = 50000  # Average deal value
    sales_time_per_lead = 10  # Hours per lead
    sales_cost_per_hour = 100  # Cost per sales hour

    # Without ML: Sales team works all leads equally
    total_leads = len(scored_leads)
    baseline_conversion_rate = leads['converted'].mean()
    baseline_revenue = total_leads * baseline_conversion_rate * avg_deal_value
    baseline_cost = total_leads * sales_time_per_lead * sales_cost_per_hour

    # With ML: Focus on top 30% high-probability leads
    top_30_pct = int(total_leads * 0.3)
    high_prob_leads = scored_leads.nlargest(top_30_pct, 'conversion_probability')

    # Add actual conversion data
    high_prob_leads['actual_converted'] = leads.loc[high_prob_leads.index, 'converted']

    ml_conversion_rate = high_prob_leads['actual_converted'].mean()
    ml_revenue = top_30_pct * ml_conversion_rate * avg_deal_value
    ml_cost = top_30_pct * sales_time_per_lead * sales_cost_per_hour

    print("\n📊 Lead Scoring ROI:")
    print("-" * 70)
    print(f"\nBaseline (No ML):")
    print(f"   Leads worked: {total_leads}")
    print(f"   Conversion rate: {baseline_conversion_rate:.1%}")
    print(f"   Revenue: ${baseline_revenue:,.0f}")
    print(f"   Cost: ${baseline_cost:,.0f}")
    print(f"   Profit: ${baseline_revenue - baseline_cost:,.0f}")

    print(f"\nWith ML (Top 30% Focus):")
    print(f"   Leads worked: {top_30_pct}")
    print(f"   Conversion rate: {ml_conversion_rate:.1%}")
    print(f"   Revenue: ${ml_revenue:,.0f}")
    print(f"   Cost: ${ml_cost:,.0f}")
    print(f"   Profit: ${ml_revenue - ml_cost:,.0f}")

    profit_improvement = (ml_revenue - ml_cost) - (baseline_revenue - baseline_cost)
    efficiency_gain = (baseline_cost - ml_cost) / baseline_cost

    print(f"\n💰 Business Impact:")
    print(f"   Additional profit: ${profit_improvement:,.0f}")
    print(f"   Cost reduction: {efficiency_gain:.1%}")
    print(f"   ROI: {(profit_improvement / ml_cost) * 100:.1f}%")


def main():
    """Run all example analyses."""
    print("\n" + "=" * 70)
    print("B2B SALES ML - ADVANCED ANALYTICS")
    print("=" * 70)

    try:
        # Run analyses
        analyze_lead_segments()
        analyze_churn_segments()
        calculate_roi_metrics()

        print("\n" + "=" * 70)
        print("✅ Analysis Complete!")
        print("=" * 70 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run 'python run_pipeline.py' first to generate data and train models.\n")


if __name__ == "__main__":
    main()
