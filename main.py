import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- App-tittel ---
st.title("🤖 AI Trader – Historisk analyse + AI-anbefaling")
st.write("""
Appen henter 10+ års data, trener en XGBoost-modell på tekniske indikatorer, 
og gir en AI-basert anbefaling (kjøp, selg eller vent).
""")

# --- Brukervalg ---
ticker = st.selectbox(
    "Velg ticker:",
    [
        "AAPL", "MSFT", "TSLA", "AMZN", "META", "GOOG", "NVDA", "NFLX", "NOVO-B.CO"
    ],
    index=0
)


st.write("Velg en aksje og trykk **Analyser** for å starte AI-vurderingen.")

# --- Hovedlogikk ---
if st.button("Analyser"):
    st.info(f"Henter historiske data for {ticker} ...")
    data = yf.download(ticker, start="2010-01-01", progress=False)

    if data is None or data.empty:
        st.error("Kunne ikke hente data for denne tickeren.")
        st.stop()

    # --- Bruk justert sluttkurs hvis tilgjengelig ---
    data["Close"] = data.get("Adj Close", data["Close"])

    # --- RSI ---
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    # --- EMA ---
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["EMA_200"] = data["Close"].ewm(span=200, adjust=False).mean()

    # --- Bollinger Bands ---
    bb_mid = data["Close"].rolling(window=20, min_periods=1).mean()
    bb_std = data["Close"].rolling(window=20, min_periods=1).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    data["BB_%B"] = (data["Close"] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # --- ADX ---
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    close = data["Close"].astype(float)

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    tr_components = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1)
    tr = tr_components.max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean().replace(0, np.nan)

    plus_di = 100 * (plus_dm.rolling(14, min_periods=1).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(14, min_periods=1).sum() / atr)
    adx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    if isinstance(adx, pd.DataFrame):
        adx = adx.iloc[:, 0]
    adx = pd.Series(adx.values, index=data.index, name="ADX")
    data["ADX"] = adx.fillna(method="bfill").fillna(method="ffill")

    # --- Rens data ---
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(method="bfill").fillna(method="ffill")

    if len(data) < 100:
        st.error("For lite data igjen etter indikatorberegninger.")
        st.stop()

    # --- Tren og evaluer for flere horisonter ---
    X = data[["RSI", "EMA_20", "EMA_50", "EMA_200", "BB_%B", "ADX"]]

    results = []
    best_model = None
    best_horizon = None
    best_acc = 0.0

    for horizon in [1, 5, 10, 20]:
        target_col = f"Target_{horizon}d"
        data[target_col] = np.where(data["Close"].shift(-horizon) > data["Close"], 1, 0)

        y = data[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        results.append({
            "Horisont": f"{horizon} dager",
            "Nøyaktighet": f"{acc:.2%}"
        })

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_horizon = horizon

    # --- Vis resultat-tabell ---
    st.subheader("📊 Sammenlikning av ulike tidshorisonter")
    st.table(pd.DataFrame(results))

    # --- AI-anbefaling basert på beste horisont ---
    sannsynlighet = best_model.predict_proba(X.iloc[[-1]])[0, 1]

    st.subheader("🤖 AI-anbefaling")
    if sannsynlighet > 0.6:
        st.success(
            f"AI-en vurderer {ticker} som **KJØP** "
            f"(oppgangssannsynlighet: {sannsynlighet:.1%}, horisont: {best_horizon} dager, "
            f"modellens treffsikkerhet: {best_acc:.2%}) 📈"
        )
    elif sannsynlighet < 0.4:
        st.error(
            f"AI-en vurderer {ticker} som **SELGE** "
            f"(oppgangssannsynlighet: {sannsynlighet:.1%}, horisont: {best_horizon} dager, "
            f"modellens treffsikkerhet: {best_acc:.2%}) 📉"
        )
    else:
        st.warning(
            f"AI-en anbefaler **VENT** "
            f"(oppgangssannsynlighet: {sannsynlighet:.1%}, horisont: {best_horizon} dager, "
            f"modellens treffsikkerhet: {best_acc:.2%}) ⚖️"
        )

    st.caption(f"""
📅 Merk: AI-en testet flere tidshorisonter (1, 5, 10, 20 dager).  
Denne anbefalingen er basert på **{best_horizon} dager**, som historisk ga høyest treffsikkerhet ({best_acc:.2%}).  
""")
