import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pandas.tseries.offsets import BDay
import datetime

# --- Streamlit config ---
st.set_page_config(page_title="🤖 AI Aksje Trader", page_icon="💹", layout="centered")
st.title("🤖 AI Trader – 10 års historikk + maskinlæring")
st.write("Denne appen trener en Random Forest på tekniske indikatorer og gir en AI-anbefaling for neste børsdag.")

# --- Velg aksjer manuelt (raske og gyldige) ---
tickere = ["AAPL", "TSLA", "MSFT", "GOOG", "AMZN", "NVDA", "META", "NFLX", "EQNR.OL", "DNB.OL"]

# --- Input ---
ticker = st.selectbox("Velg ticker:", tickere, index=0)

if st.button("Analyser"):
    # --- 1. Hent data ---
    data = yf.download(ticker, period="10y", interval="1d")

    if data.empty:
        st.error("Kunne ikke hente data for denne tickeren.")
    else:
        # --- 2. Beregn indikatorer ---
        data["SMA_20"] = data["Close"].rolling(window=20).mean()
        data["SMA_50"] = data["Close"].rolling(window=50).mean()

        # RSI
        delta = data["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))

        # EMA
        ema12 = data["Close"].ewm(span=12, adjust=False).mean()
        ema26 = data["Close"].ewm(span=26, adjust=False).mean()
        data["EMA_12"] = ema12
        data["EMA_26"] = ema26
        data["EMA_Cross"] = (ema12 - ema26)  # positiv = bullish, negativ = bearish

        # MACD
        data["MACD"] = ema12 - ema26

        # Bollinger Bands
        data["BB_mid"] = data["Close"].rolling(window=20).mean()
        data["BB_std"] = data["Close"].rolling(window=20).std()
        data["BB_upper"] = data["BB_mid"] + 2 * data["BB_std"]
        data["BB_lower"] = data["BB_mid"] - 2 * data["BB_std"]

        # Volatilitet & momentum
        data["Volatility"] = data["Close"].pct_change().rolling(20).std()
        data["Momentum"] = data["Close"].pct_change(10)
        data["Volume_trend"] = data["Volume"].pct_change().rolling(20).mean()

        # --- 3. Lag target (opp/ned neste dag) ---
        fremtid = 5  # antall dager frem
        data["Target"] = (data["Close"].shift(-fremtid) > data["Close"]).astype(int)

        data = data.dropna()

        # --- 4. Sett opp features og labels ---
        X = data[[
            "SMA_20", "SMA_50", "RSI", "MACD", "Volatility",
            "Momentum", "Volume_trend", "EMA_Cross", "BB_mid", "BB_upper", "BB_lower"
        ]]
        y = data["Target"]

        # --- 5. Split data ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # --- 6. Tren modell (med class_weight balansert) ---
        model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)

        # --- 7. Evaluer ---
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.subheader("📊 Modell-evaluering")
        st.write(f"Treffsikkerhet: **{acc:.2%}**")
        st.text(classification_report(y_test, y_pred))

        # --- 8. Feature importance ---
        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.subheader("🔎 Viktigste indikatorer")
        st.bar_chart(importance.set_index("Feature"))
        st.table(importance)

        # --- 9. AI-anbefaling for neste børsdag ---
        siste = X.iloc[[-1]]
        sannsynlighet = model.predict_proba(siste)[0, 1]

        neste_borsdag = (datetime.date.today() + BDay(1)).date()

        st.subheader("🤖 AI-anbefaling")
        st.write(
            f"Sannsynlighet for oppgang i {ticker} neste børsdag (**{neste_borsdag}**): "
            f"**{sannsynlighet:.2%}**"
        )

        if sannsynlighet > 0.6:
            st.success("💚 AI sier: **KJØP** (modellen ser høy sannsynlighet for oppgang)")
        elif sannsynlighet < 0.4:
            st.error("❤️ AI sier: **SELG** (modellen ser høy sannsynlighet for nedgang)")
        else:
            st.warning("⚖️ AI sier: **VENT** (usikkert signal)")

