import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- Streamlit config ---
st.set_page_config(page_title="🤖 AI Aksje Trader", page_icon="💹", layout="centered")
st.title("🤖 AI Trader – 10 års historikk + maskinlæring")
st.write("Appen trener en Random Forest på tekniske indikatorer og gir en AI-anbefaling for neste dag.")

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

        delta = data["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))

        ema12 = data["Close"].ewm(span=12, adjust=False).mean()
        ema26 = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = ema12 - ema26

        data["Volatility"] = data["Close"].pct_change().rolling(20).std()
        data["Momentum"] = data["Close"].pct_change(10)
        data["Volume_trend"] = data["Volume"].pct_change().rolling(20).mean()

        # --- 3. Lag target (opp/ned neste dag) ---
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
        data = data.dropna()

        # --- 4. Sett opp features og labels ---
        X = data[["SMA_20", "SMA_50", "RSI", "MACD", "Volatility", "Momentum", "Volume_trend"]]
        y = data["Target"]

        # --- 5. Split data ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # --- 6. Tren modell ---
        model = RandomForestClassifier(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)

        # --- 7. Evaluer ---
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.subheader("📊 Modell-evaluering")
        st.write(f"Treffsikkerhet: **{acc:.2%}**")
        st.text(classification_report(y_test, y_pred))

        # --- 8. Feature importance ---
        importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.subheader("🔎 Viktigste indikatorer")
        st.bar_chart(importance)

        # --- 9. AI-anbefaling for neste dag ---
        siste = X.iloc[[-1]]
        sannsynlighet = model.predict_proba(siste)[0, 1]

        st.subheader("🤖 AI-anbefaling")
        st.write(f"Sannsynlighet for oppgang i {ticker} i morgen: **{sannsynlighet:.2%}**")

        if sannsynlighet > 0.6:
            st.success("💚 AI sier: **KJØP**")
        elif sannsynlighet < 0.4:
            st.error("❤️ AI sier: **SELG**")
        else:
            st.warning("⚖️ AI sier: **VENT**")

        # --- 10. Ekstra indikator-sjekker ---
        rsi_verdi = float(siste["RSI"].iloc[0])
        macd_verdi = float(siste["MACD"].iloc[0])
        momentum_verdi = float(siste["Momentum"].iloc[0])
        vol_verdi = float(siste["Volatility"].iloc[0])

        # RSI
        if rsi_verdi < 30:
            st.info("📉 RSI: Oversolgt → mulig KJØP-signal")
        elif rsi_verdi > 70:
            st.info("📈 RSI: Overkjøpt → mulig SELG-signal")
        else:
            st.info("RSI: Nøytral sone")

        # MACD
        if macd_verdi > 0:
            st.info("📊 MACD: Positiv trend")
        else:
            st.info("📊 MACD: Negativ trend")

        # Momentum
        if momentum_verdi > 0:
            st.info("⚡ Momentum: Positiv siste 10 dager")
        else:
            st.info("⚡ Momentum: Negativ siste 10 dager")

        # Volatilitet
        if vol_verdi > 0.02:
            st.info("🌪️ Høy volatilitet: Risikoen er større")
