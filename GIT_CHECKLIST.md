# Git Push Checklist ✅

## Vor dem ersten Push

### 1. Git Repository initialisieren

```bash
cd b2b_sales_ml_pipeline
git init
```

### 2. Sensitive Daten prüfen

✅ **Folgende Dateien sind NICHT im Git (siehe .gitignore):**
- `data/*.csv` - Generierte Daten (lokal neu generieren)
- `models/*.joblib` - Trainierte Modelle (lokal neu trainieren)
- `outputs/*.csv` - Vorhersage-Reports
- `__pycache__/` - Python Cache

✅ **Diese Dateien SIND im Git:**
- Alle `.py` Skripte
- `README.md`, `QUICKSTART.md`, `PROJECT_SUMMARY.md`
- `requirements.txt`
- `LICENSE`, `CONTRIBUTING.md`
- `.gitignore`
- `.gitkeep` Dateien (für Ordner-Struktur)

### 3. Projekt-Struktur verifizieren

```bash
# Prüfen was committed wird
git status

# Sollte zeigen:
# - Alle .py Dateien
# - Alle .md Dateien
# - requirements.txt
# - .gitignore
# - LICENSE
```

### 4. Erste Commits

```bash
# Alle Dateien zum Staging hinzufügen
git add .

# Ersten Commit erstellen
git commit -m "Initial commit: B2B Sales ML Pipeline

- Lead Scoring Model (Gradient Boosting)
- Churn Prediction Model (Gradient Boosting)
- Complete end-to-end pipeline with data generation
- Comprehensive documentation (README, QUICKSTART)
- Production-ready code with docstrings
- Business-focused feature engineering"

# Branch umbenennen (falls nötig)
git branch -M main
```

### 5. GitHub Repository erstellen

1. Gehe zu https://github.com/new
2. Repository Name: `b2b-sales-ml-pipeline`
3. Description: "End-to-End ML Pipeline for B2B Sales Lead Scoring & Churn Prediction"
4. Public oder Private wählen
5. **WICHTIG:** Keine README/License/gitignore hinzufügen (haben wir schon lokal!)
6. Repository erstellen

### 6. Remote hinzufügen & pushen

```bash
# Remote hinzufügen (ersetze USERNAME mit deinem GitHub Username)
git remote add origin https://github.com/USERNAME/b2b-sales-ml-pipeline.git

# Pushen
git push -u origin main
```

## Nach dem Push

### Repository-Einstellungen (optional)

**GitHub Topics hinzufügen:**
- `machine-learning`
- `sales-automation`
- `churn-prediction`
- `lead-scoring`
- `gradient-boosting`
- `scikit-learn`
- `b2b-saas`
- `customer-success`

**About Section:**
```
End-to-End ML Pipeline for B2B Sales: Lead Scoring & Churn Prediction
with Gradient Boosting. Production-ready code with business-focused
feature engineering.
```

## Wichtige Hinweise für andere User

### Was neue User tun müssen:

```bash
# 1. Repository clonen
git clone https://github.com/USERNAME/b2b-sales-ml-pipeline.git
cd b2b-sales-ml-pipeline

# 2. Dependencies installieren
pip install -r requirements.txt

# 3. Komplette Pipeline ausführen (generiert Daten + trainiert Modelle)
python run_pipeline.py
```

**Das war's!** Nach `run_pipeline.py` haben sie:
- ✅ Synthetische Daten in `data/`
- ✅ Trainierte Modelle in `models/`
- ✅ Vorhersage-Reports in `outputs/`

## Häufige Fragen

### "Warum sind keine Modelle im Git?"

Trainierte ML-Modelle sind:
- Groß (mehrere MB)
- Lokal schnell zu regenerieren (`python pipeline.py`)
- Plattform-abhängig (joblib-Versionen)

→ Best Practice: Nur Code committen, Modelle lokal trainieren

### "Sollte ich sample Daten committen?"

**NEIN**, weil:
- Nutzer sollen `data_gen.py` verstehen
- Zeigt, dass das System funktioniert
- Daten sind nur 5 Sekunden zum generieren

Aber: Du kannst optional eine `data/sample_leads.csv` mit 10 Zeilen als Beispiel committen.

## Git Workflow für Updates

```bash
# Feature Branch erstellen
git checkout -b feature/xgboost-support

# Änderungen machen...
# Code schreiben, testen

# Committen
git add .
git commit -m "Add XGBoost model support"

# Pushen
git push origin feature/xgboost-support

# Auf GitHub: Pull Request öffnen
# Nach Review: In main mergen
```

## Status Check

Bevor du pushst, prüfe:

- [ ] `python run_pipeline.py` läuft ohne Fehler
- [ ] README ist aktuell
- [ ] Keine `.csv` oder `.joblib` Dateien im Staging
- [ ] `requirements.txt` enthält alle Dependencies
- [ ] Commit Message ist aussagekräftig
- [ ] Keine Secrets im Code (API Keys, etc.)

---

**Ready to push?** 🚀

```bash
git push -u origin main
```
