# Contributing to B2B Sales ML Pipeline

Danke für dein Interesse an diesem Projekt!

## Wie du beitragen kannst

### 1. Issues melden

Wenn du einen Bug findest oder eine Feature-Idee hast:
- Öffne ein Issue auf GitHub
- Beschreibe das Problem oder die Idee klar
- Füge Code-Beispiele oder Screenshots hinzu, falls relevant

### 2. Code beitragen

#### Setup für lokale Entwicklung

```bash
# Repo clonen
git clone https://github.com/DEIN-USERNAME/b2b-sales-ml-pipeline.git
cd b2b-sales-ml-pipeline

# Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # Auf Windows: venv\Scripts\activate

# Dependencies installieren
pip install -r requirements.txt

# Projekt testen
python run_pipeline.py
```

#### Pull Request Guidelines

1. **Fork das Repository**
2. **Erstelle einen Feature Branch** (`git checkout -b feature/AmazingFeature`)
3. **Schreibe sauberen Code:**
   - Docstrings für alle Funktionen/Klassen
   - Type Hints wo sinnvoll
   - Kommentare für komplexe Logik
4. **Teste deine Änderungen** (alle 3 Skripte müssen funktionieren)
5. **Commit mit klarer Message** (`git commit -m 'Add XGBoost model support'`)
6. **Push zum Branch** (`git push origin feature/AmazingFeature`)
7. **Öffne einen Pull Request**

### 3. Interessante Contributions

Hier sind Bereiche, wo wir Hilfe gebrauchen können:

#### High Priority
- [ ] XGBoost Integration (wie in README beschrieben)
- [ ] Unit Tests (pytest)
- [ ] CI/CD Pipeline (GitHub Actions)
- [ ] Hyperparameter Tuning (GridSearchCV)

#### Medium Priority
- [ ] Streamlit Dashboard für Visualisierung
- [ ] REST API Wrapper (FastAPI)
- [ ] Docker Container
- [ ] CRM-Integration (Salesforce/HubSpot Connector)

#### Nice to Have
- [ ] Jupyter Notebook Tutorial
- [ ] Model Explainability (SHAP Values)
- [ ] A/B Testing Framework
- [ ] Multi-Language Support (EN/DE)

## Code Style

- **Python:** PEP 8 Konventionen
- **Docstrings:** Google Style
- **Variablen:** Sprechende Namen (`conversion_probability` nicht `cp`)
- **Funktionen:** Kleinbuchstaben mit Underscores (`calculate_roi()` nicht `calculateROI()`)

## Testing

Bevor du einen PR öffnest, stelle sicher dass:

```bash
# 1. Daten-Generierung funktioniert
python data_gen.py

# 2. Training läuft ohne Fehler
python pipeline.py

# 3. Predictions funktionieren
python predict.py

# 4. Komplette Pipeline läuft
python run_pipeline.py

# 5. Analyse-Skript funktioniert
python example_analysis.py
```

## Fragen?

Öffne ein Issue mit dem Label `question` oder schreibe eine Email.

## Lizenz

Durch deinen Beitrag stimmst du zu, dass deine Änderungen unter der MIT Lizenz veröffentlicht werden.

---

**Danke, dass du dieses Projekt besser machst!** 🚀
