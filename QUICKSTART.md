# Quick Start Guide

## Installation (30 Sekunden)

```bash
pip install -r requirements.txt
```

## Komplette Pipeline ausführen (1 Befehl)

```bash
python run_pipeline.py
```

Das war's! Dieser eine Befehl:
1. Generiert synthetische B2B Daten (1000 Leads, 800 Kunden)
2. Trainiert beide ML-Modelle (Lead Scoring + Churn Prediction)
3. Erstellt Vorhersage-Reports mit priorisierten Listen

## Oder: Schritt für Schritt

### Schritt 1: Daten generieren
```bash
python data_gen.py
```
Erstellt `data/leads.csv` und `data/customers.csv`

### Schritt 2: Modelle trainieren
```bash
python pipeline.py
```
Trainiert und speichert beide Modelle in `models/`

### Schritt 3: Vorhersagen generieren
```bash
python predict.py
```
Erstellt Reports in `outputs/`:
- High-Priority Leads (>80% Conversion)
- At-Risk Customers (>70% Churn Risk)

## Output-Dateien

Nach dem Ausführen findest du:

```
b2b_sales_ml_pipeline/
├── data/
│   ├── leads.csv                    # 1000 synthetische Leads
│   └── customers.csv                # 800 synthetische Kunden
├── models/
│   ├── lead_scoring_model.joblib    # Trainiertes Lead Scoring Model
│   └── churn_prediction_model.joblib # Trainiertes Churn Model
└── outputs/
    ├── all_leads_scored_*.csv       # Alle Leads mit Scores
    ├── high_priority_leads_*.csv    # Top Leads (>80%)
    ├── all_customers_scored_*.csv   # Alle Kunden mit Churn Risk
    └── at_risk_customers_*.csv      # At-Risk Kunden (>70%)
```

## Verwendung im Code

```python
from predict import SalesPredictor
import pandas as pd

# Modelle laden
predictor = SalesPredictor(models_dir='models')

# Leads scoren
leads = pd.read_csv('my_leads.csv')
scored = predictor.predict_lead_scores(leads)

# High-Priority Leads identifizieren
hot_leads = predictor.generate_high_priority_leads(leads, threshold=0.80)
print(hot_leads[['company_size', 'conversion_probability_pct', 'recommended_action']])
```

## Wichtigste Features

### Lead Scoring
- `lead_source` - **Referrals konvertieren 2-3x besser!**
- `decision_maker_reached` - Im B2B entscheidend
- `email_interaction_score` - Zeigt echtes Interesse
- `budget_confirmed` - Ohne Budget kein Deal

### Churn Prediction
- `nps_score` - **Stärkster Churn-Indikator** (0-6 = Detractors)
- `feature_usage_index` - Low Usage = kein Value
- `last_login_days_ago` - >30 Tage = hohes Risiko
- `support_tickets_last_month` - Viele Tickets = Probleme

## Next Steps

1. Schau dir die generierten Reports in `outputs/` an
2. Lies die vollständige Dokumentation in [README.md](README.md)
3. Passe die Features in `data_gen.py` an deine Bedürfnisse an
4. Integriere mit deinem CRM/Data Warehouse

## Support

Fragen? Schau in [README.md](README.md) für detaillierte Erklärungen zu:
- Feature Engineering & Business Rationale
- Model Performance Metriken
- Code-Beispiele
- Erweiterungsmöglichkeiten
