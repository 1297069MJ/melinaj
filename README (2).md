# Hispanic Heritage Historical Data Explorer

A Streamlit classroom app for exploring World Bank historical indicators, editing a working data table, fitting polynomial models, calculating estimates, comparing countries, and studying functions/calculus.

## Country coverage

The country selector includes all **20 Spanish-speaking sovereign countries**:

Argentina, Bolivia, Chile, Colombia, Costa Rica, Cuba, Dominican Republic, Ecuador, El Salvador, Equatorial Guinea, Guatemala, Honduras, Mexico, Nicaragua, Panama, Paraguay, Peru, Spain, Uruguay, and Venezuela.

It also lists **Puerto Rico separately as a territory** and **Brazil separately as a Latin American comparison** (Brazil is not a Spanish-speaking/Hispanic country).

## Personalize the look

Use the sidebar's **Personalize your website** controls to switch between:
- Fonts: Nunito, Poppins, Quicksand, Playfair Display, and Lora.
- Backgrounds: Cotton candy, Sage garden, Peach lemonade, Lavender sky, Ocean breeze, and Clean paper.

The selected font is requested from Google Fonts when the browser has internet access; system sans-serif fallbacks are provided. Background themes are built into the app and do not require image files.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Render

Create a **Python Web Service** connected to your GitHub repository. Put `app.py`, `requirements.txt`, `README.md`, and `sources.md` in the repository root.

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`

If the service fails to start, open Render's deployment/runtime logs and check the first Python traceback. Confirm that the repository root contains `app.py` and `requirements.txt`, and that the start command uses `$PORT`.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `README.md`, and `sources.md`.
2. In Streamlit Community Cloud, create an app from that repository.
3. Select `app.py` as the main file and deploy.

## Data and model notes

- The app requests the 70 completed calendar years from current year minus 70 through current year minus 1. Actual coverage depends on the country and indicator.
- Missing World Bank values are omitted; the app does not invent replacement data.
- The polynomial degree must have at least degree + 1 distinct valid years.
- R-squared and MAE are in-sample metrics. A chronological holdout MAE is shown separately only when enough rows exist.
- Polynomial extrapolations can be unstable and are not validated forecasts or prediction intervals.
- GDP per capita is not average wealth or household income; crude birth rate is a rate, not a count; net migration is not an emigration count.
- Editing the table creates a user-modified scenario. Reset restores source observations.
- The function lab accepts a restricted mathematical grammar and does not execute arbitrary Python code. Numerical calculus analysis for complicated expressions can be approximate or incomplete; do not treat this as a validated scientific or medical model.

## Verification performed for this revision

- Python syntax compilation: passed.
- Helper checks: polynomial fitting on a known quadratic, numeric/year validation, duplicate-year detection, and restricted mathematical expression parsing: passed.
- Full Streamlit browser execution and live World Bank API retrieval were not run in this environment. Test the deployed app and inspect Render logs before submitting it.

## Deployment checklist

- Confirm all 20 Spanish-speaking sovereign countries appear in the country selector.
- Confirm World Bank data loads and reports actual year coverage and observation count.
- Test an unavailable API response or offline connection; the app should show an error/no-data message and must not fabricate values.
- Edit a value, then use **Reset to source**.
- Try duplicate years, non-integer years, nonnumeric values, and invalid negative values for nonnegative indicators.
- Try a polynomial degree that exceeds available distinct years.
- Compare a single-year estimate with the same year in batch results.
- Change the font, background, degree, sample increment, display horizon, comparison countries, course mode, and function-visibility controls.
- Download and open the CSV, XLSX, and PDF files.
- Review Render logs if deployment fails.
