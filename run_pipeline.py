#!/usr/bin/env python3
"""
B2B Sales ML Pipeline - Complete Workflow Runner
================================================
Runs the entire pipeline from data generation to predictions.
Use this script for a complete end-to-end demonstration.
"""

import sys
import os
from datetime import datetime


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def run_complete_pipeline():
    """Execute the complete ML pipeline."""

    print_header("B2B SALES ML PIPELINE - COMPLETE WORKFLOW")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Step 1: Generate Data
    print_header("STEP 1/3: DATA GENERATION")
    try:
        from data_gen import B2BSalesDataGenerator
        generator = B2BSalesDataGenerator(random_state=42)
        generator.save_data(output_dir='data')
        print("\n✅ Data generation completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in data generation: {e}")
        sys.exit(1)

    # Step 2: Train Models
    print_header("STEP 2/3: MODEL TRAINING")
    try:
        from pipeline import SalesPipeline
        pipeline = SalesPipeline(model_type='gradient_boosting', random_state=42)

        # Train lead scoring
        lead_results = pipeline.train_lead_scoring_model(
            data_path='data/leads.csv',
            test_size=0.2
        )

        # Train churn prediction
        churn_results = pipeline.train_churn_prediction_model(
            data_path='data/customers.csv',
            test_size=0.2
        )

        # Save models
        pipeline.save_models(output_dir='models')

        print("\n✅ Model training completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in model training: {e}")
        sys.exit(1)

    # Step 3: Generate Predictions
    print_header("STEP 3/3: PREDICTIONS & REPORTING")
    try:
        from predict import SalesPredictor
        import pandas as pd

        predictor = SalesPredictor(models_dir='models')

        # Load data
        leads_df = pd.read_csv('data/leads.csv').drop('converted', axis=1)
        customers_df = pd.read_csv('data/customers.csv').drop('churned', axis=1)

        # Generate reports
        reports = predictor.export_predictions(
            leads_df=leads_df,
            customers_df=customers_df,
            output_dir='outputs'
        )

        # Print summary
        predictor.print_summary(leads_df=leads_df, customers_df=customers_df)

        print("✅ Predictions & reporting completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in predictions: {e}")
        sys.exit(1)

    # Final Summary
    print_header("PIPELINE EXECUTION SUMMARY")

    print("📁 Generated Files:")
    print("\n   Data:")
    print("   ├── data/leads.csv")
    print("   └── data/customers.csv")

    print("\n   Models:")
    print("   ├── models/lead_scoring_model.joblib")
    print("   └── models/churn_prediction_model.joblib")

    print("\n   Reports:")
    for report_name, report_path in reports.items():
        print(f"   └── {report_path}")

    print("\n" + "=" * 70)
    print("🎉 COMPLETE PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 70)

    print("\n📊 Next Steps:")
    print("   1. Review the generated reports in outputs/")
    print("   2. Check model performance metrics above")
    print("   3. Customize features in data_gen.py for your use case")
    print("   4. Integrate with your CRM/data warehouse")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    try:
        run_complete_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
