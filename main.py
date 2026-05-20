# Практическая работа: ML-продукт для анализа отзывов на датасете Rotten Tomatoes.
# Запуск: pip install -r requirements.txt  →  python main.py
# После нужно открыть http://127.0.0.1:5000 в браузере
from __future__ import annotations

import base64
import io
import os
import re
import sqlite3
import uuid
import warnings
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROTTEN_CSV_PATH = os.path.join(BASE_DIR, "rotten_tomatoes_movie_reviews.csv")
DB_PATH = os.path.join(BASE_DIR, "reviews.db")
REPORT_PATH = os.path.join(BASE_DIR, "tomato_report.png")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-pr10-mpt-secret-Jjfhdb8i29fh9snbiskjeo2fiod")

# Глобальные артефакты после init_app()
vectorizer: Optional[TfidfVectorizer] = None
best_clf: Any = None
BEST_MODEL_NAME: str = ""
METRICS_ROWS: List[Dict[str, str]] = []
PRODUCT_LIST: List[str] = []
TRAIN_NOTE: str = ""


def mask_phone(phone: str) -> str:
    if phone and len(phone) >= 10:
        return phone[:4] + "***" + phone[-4:]
    return phone or ""


def mask_name(name: str) -> str:
    if not name:
        return ""
    if len(name) <= 3:
        return name[0] + "*" * (len(name) - 1)
    return name[:2] + "*" * max(0, len(name) - 4) + name[-2:]


def _parse_original_score_to_5(score: Any) -> Optional[int]:
    if score is None:
        return None
    s = str(score).strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    num = float(m.group(1))
    den = float(m.group(2))
    if den <= 0:
        return None
    val = int(round((num / den) * 5))
    return max(1, min(5, val))


def load_catalog_dataframe(max_rows: int = 20000) -> pd.DataFrame:
    # Основной рабочий набор из Rotten Tomatoes: текст, оценка, товар, дата, пользователь.
    use_cols = ["id", "reviewId", "creationDate", "criticName", "originalScore", "reviewText", "scoreSentiment"]
    df = pd.read_csv(ROTTEN_CSV_PATH, usecols=use_cols)
    df = df.dropna(subset=["reviewText", "scoreSentiment"]).copy()
    df["scoreSentiment"] = df["scoreSentiment"].astype(str).str.upper().str.strip()
    df = df[df["scoreSentiment"].isin(["POSITIVE", "NEGATIVE"])]
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    df["text"] = df["reviewText"].astype(str)
    df["product"] = (
        df["id"]
        .fillna("unknown_movie")
        .astype(str)
        .str.replace("_", " ", regex=False)
        .str.strip()
        .str.title()
    )
    df["user_name"] = df["criticName"].fillna("Unknown Critic").astype(str)
    df["user_id"] = (
        "critic_"
        + df["user_name"]
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    df.loc[df["user_id"] == "critic_", "user_id"] = "critic_unknown"
    df["date"] = pd.to_datetime(df["creationDate"], errors="coerce")
    bad_dates = df["date"].isna()
    if bad_dates.any():
        rng = np.random.default_rng(42)
        days_ago = rng.integers(0, 540, size=int(bad_dates.sum()))
        repl_dates = [datetime.now() - timedelta(days=int(d)) for d in days_ago]
        df.loc[bad_dates, "date"] = repl_dates

    rating_parsed = df["originalScore"].apply(_parse_original_score_to_5)
    rating_fallback = pd.Series(np.where(df["scoreSentiment"] == "POSITIVE", 4, 2), index=df.index)
    df["rating"] = rating_parsed.where(rating_parsed.notna(), rating_fallback).astype(int).clip(1, 5)
    df["sentiment"] = (df["scoreSentiment"] == "POSITIVE").astype(int)

    rng = np.random.default_rng(42)
    df["phone"] = [f"+1{rng.integers(200,999)}{rng.integers(1000000,9999999)}" for _ in range(len(df))]
    return df[["text", "rating", "product", "date", "user_id", "user_name", "phone", "sentiment"]].reset_index(
        drop=True
    )


def load_sentiment_training_dataframe(max_rows: int = 120000) -> pd.DataFrame:
    # Датасет для обучения тональности: reviewText + scoreSentiment.
    if os.path.exists(ROTTEN_CSV_PATH):
        src = pd.read_csv(ROTTEN_CSV_PATH, usecols=["reviewText", "scoreSentiment"])
        src = src.dropna(subset=["reviewText", "scoreSentiment"]).copy()
        src["scoreSentiment"] = src["scoreSentiment"].astype(str).str.upper().str.strip()
        src = src[src["scoreSentiment"].isin(["POSITIVE", "NEGATIVE"])]
        if len(src) > max_rows:
            src = src.sample(n=max_rows, random_state=42)
        out = pd.DataFrame(
            {
                "text": src["reviewText"].astype(str),
                "sentiment": (src["scoreSentiment"] == "POSITIVE").astype(int),
            }
        )
        return out

    fallback = load_catalog_dataframe(max_rows=5000)[["text", "sentiment"]].copy()
    return fallback


def tokenize_clean(text: str) -> List[str]:
    text = text.lower()
    words = re.findall(r"[a-zA-Z']+", text)
    return [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]


def predict_sentiment_with_confidence(text: str) -> Tuple[str, float]:
    # Возвращает метку (позитивный или негативный) и уверенность модели от 0 до 1.
    if not text or not text.strip():
        return "негативный", 0.5
    assert vectorizer is not None and best_clf is not None
    Xv = vectorizer.transform([text])
    proba = best_clf.predict_proba(Xv)[0]
    cls = int(np.argmax(proba))
    conf = float(proba[cls])
    label = "позитивный" if cls == 1 else "негативный"
    return label, conf


def top_products_by_positive_share(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    # Топ товаров по доле позитива (sentiment_pos == 1).
    if df.empty:
        return pd.DataFrame(columns=["product", "pos_share", "n"])
    g = df.groupby("product").agg(pos=("sentiment_pos", "mean"), n=("sentiment_pos", "size")).reset_index()
    g = g.rename(columns={"pos": "pos_share"})
    return g.sort_values("pos_share", ascending=False).head(top_n)


def five_most_frequent_words_negative(df: pd.DataFrame) -> List[Tuple[str, int]]:
    neg_texts = df.loc[df["sentiment_pos"] == 0, "text"].astype(str)
    cnt: Counter[str] = Counter()
    for t in neg_texts:
        cnt.update(tokenize_clean(t))
    return cnt.most_common(5)


def init_db() -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            user_name TEXT,
            phone TEXT,
            product TEXT NOT NULL,
            text TEXT NOT NULL,
            rating INTEGER NOT NULL,
            review_date TEXT NOT NULL,
            sentiment_pos INTEGER NOT NULL
        )
        """
    )
    con.commit()
    con.close()


def db_review_count() -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM reviews")
    n = int(cur.fetchone()[0])
    con.close()
    return n


def seed_db_from_df(df: pd.DataFrame) -> None:
    con = sqlite3.connect(DB_PATH)
    pred_list = df["sentiment"].astype(int).tolist()
    rows = []
    for j, (_, r) in enumerate(df.iterrows()):
        sp = int(pred_list[j])
        rows.append(
            (
                r["user_id"],
                mask_name(str(r["user_name"])),
                mask_phone(str(r["phone"])),
                r["product"],
                r["text"],
                int(r["rating"]),
                pd.Timestamp(r["date"]).isoformat(),
                sp,
            )
        )
    cur = con.cursor()
    cur.executemany(
        "INSERT INTO reviews (user_id, user_name, phone, product, text, rating, review_date, sentiment_pos) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()


def load_reviews_from_db() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM reviews", con)
    con.close()
    if df.empty:
        return df
    # В БД могут быть смешанные форматы времени (с/без микросекунд),
    # поэтому включаем "mixed" и мягкий fallback.
    df["date"] = pd.to_datetime(df["review_date"], errors="coerce", format="mixed")
    if df["date"].isna().any():
        df.loc[df["date"].isna(), "date"] = pd.Timestamp.now()
    return df


def train_classifiers(df: pd.DataFrame) -> Tuple[TfidfVectorizer, Any, str, List[Dict[str, str]]]:
    X_text = df["text"].astype(str).values
    y = df["sentiment"].astype(int).values
    vec = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), min_df=2, stop_words="english")
    X = vec.fit_transform(X_text)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=800, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=40, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=80,
            max_depth=5,
            learning_rate=0.12,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        ),
    }
    rows: List[Dict[str, str]] = []
    best_name = ""
    best_f1 = -1.0
    best_model = None
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        rows.append(
            {
                "name": name,
                "acc": f"{acc:.3f}",
                "prec": f"{prec:.3f}",
                "rec": f"{rec:.3f}",
                "f1": f"{f1:.3f}",
                "_f1": f1,
            }
        )
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = clf
    for r in rows:
        r.pop("_f1", None)
    return vec, best_model, best_name, rows


def user_based_recommendations(df: pd.DataFrame, user_id: str, k: int = 5) -> List[Dict[str, Any]]:
    # User-based CF: косинус между пользователями по вектору оценок по фильмам.
    if df.empty or "rating" not in df.columns:
        return []
    sub = df[df["user_id"] == user_id]
    if len(sub) < 2:
        gmean = df.groupby("product")["rating"].mean().sort_values(ascending=False)
        seen = set(sub["product"].tolist()) if len(sub) else set()
        out = []
        for movie, val in gmean.items():
            if movie not in seen:
                out.append({"title": movie, "score": float(val)})
            if len(out) >= k:
                break
        return out
    pivot = df.pivot_table(index="user_id", columns="product", values="rating", aggfunc="mean")
    pivot = pivot.fillna(0.0)
    if user_id not in pivot.index:
        return []
    urow = pivot.loc[user_id]
    if isinstance(urow, pd.DataFrame):
        urow = urow.mean(axis=0)
    Pu = urow.values.reshape(1, -1)
    sim = cosine_similarity(Pu, pivot.values)[0]
    idx = int(np.where(pivot.index == user_id)[0][0])
    sim[idx] = 0.0
    neigh = np.argsort(-sim)[: min(30, len(sim))]
    scores: Dict[str, float] = {}
    rated = set(sub["product"].tolist())
    for movie in pivot.columns:
        if movie in rated:
            continue
        num, den = 0.0, 0.0
        for ni in neigh:
            if sim[ni] <= 0:
                continue
            r = pivot.iloc[ni][movie]
            if r > 0:
                num += sim[ni] * r
                den += abs(sim[ni])
        if den > 0:
            scores[movie] = num / den
    if not scores:
        gmean = df.groupby("product")["rating"].mean().sort_values(ascending=False)
        for movie, val in gmean.items():
            if movie not in rated:
                scores[movie] = float(val)
            if len(scores) >= k * 3:
                break
    top = sorted(scores.items(), key=lambda x: -x[1])[:k]
    return [{"title": t, "score": float(s)} for t, s in top]


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def build_charts_b64(df: pd.DataFrame) -> Tuple[str, str, str]:
    if df.empty:
        empty = base64.b64encode(b"").decode("ascii")
        return empty, empty, empty
    fig1, ax1 = plt.subplots(figsize=(5, 3.2))
    ax1.hist(df["rating"].clip(1, 5), bins=np.arange(0.5, 6.5, 1), edgecolor="black", color="#6c5ce7", alpha=0.85)
    ax1.set_title("Распределение оценок")
    ax1.set_xticks([1, 2, 3, 4, 5])
    b1 = fig_to_b64(fig1)

    daily = df.groupby(df["date"].dt.date).size().reset_index(name="c")
    daily["date"] = pd.to_datetime(daily["date"])
    fig2, ax2 = plt.subplots(figsize=(5, 3.2))
    ax2.plot(daily["date"], daily["c"], color="#0984e3", marker="o", linewidth=2)
    ax2.set_title("Динамика отзывов по дням")
    fig2.autofmt_xdate()
    b2 = fig_to_b64(fig2)

    text_blob = " ".join(df["text"].astype(str).tolist()[:2000])
    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text_blob or " ")
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.imshow(wc, interpolation="bilinear")
    ax3.axis("off")
    ax3.set_title("Облако слов")
    b3 = fig_to_b64(fig3)
    return b1, b2, b3


def build_student_report(
    df: pd.DataFrame,
    metrics_rows: List[Dict[str, str]],
    best_name: str,
    top_pos: pd.DataFrame,
    neg_words: List[Tuple[str, int]],
) -> None:
    # Сохраняет в tomato_report.png сводную статистику и графики (базовый отчёт + расширение).
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Отчёт студента: анализ отзывов (IMDB Top 1000)", fontsize=14, fontweight="bold")

    daily = pd.DataFrame()
    if not df.empty:
        axes[0, 0].hist(df["rating"].clip(1, 5), bins=np.arange(0.5, 6.5, 1), edgecolor="black", color="#6c5ce7", alpha=0.8)
        axes[0, 0].set_title("Распределение оценок")
        axes[0, 0].set_xticks([1, 2, 3, 4, 5])

        daily = df.groupby(df["date"].dt.date).size().reset_index(name="reviews_count")
        daily["date"] = pd.to_datetime(daily["date"])
        axes[0, 1].plot(daily["date"], daily["reviews_count"], color="#0984e3", linewidth=2)
        axes[0, 1].set_title("Динамика отзывов по дням")
        fig.autofmt_xdate()

        text_blob = " ".join(df["text"].astype(str).tolist()[:3000])
        wc = WordCloud(width=600, height=300, background_color="white").generate(text_blob or " ")
        axes[0, 2].imshow(wc, interpolation="bilinear")
        axes[0, 2].axis("off")
        axes[0, 2].set_title("Облако слов")

    names = [r["name"] for r in metrics_rows]
    f1s = [float(r["f1"]) for r in metrics_rows]
    axes[1, 0].bar(names, f1s, color=["#00b894", "#fdcb6e", "#e17055"])
    axes[1, 0].set_title("F1-score моделей")
    axes[1, 0].set_ylim(0, 1)

    if not top_pos.empty:
        axes[1, 1].barh(top_pos["product"].head(8)[::-1], top_pos["pos_share"].head(8)[::-1], color="#00cec9")
        axes[1, 1].set_title("Топ товаров по доле позитива")
        axes[1, 1].set_xlabel("Доля позитива")

    if neg_words:
        ws, cs = zip(*neg_words)
        axes[1, 2].bar(ws, cs, color="#d63031", alpha=0.8)
        axes[1, 2].set_title("5 частых слов (негатив)")
        axes[1, 2].tick_params(axis="x", rotation=25)
    else:
        axes[1, 2].text(0.5, 0.5, "Нет негативных отзывов", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(REPORT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Прогноз числа отзывов (как в лекции — линейная регрессия по лагам)
    if not daily.empty and len(daily) >= 12:
        dc = daily.copy()
        dc["dow"] = dc["date"].dt.dayofweek
        for lag in [1, 2, 3]:
            dc[f"lag_{lag}"] = dc["reviews_count"].shift(lag)
        dc = dc.dropna()
        if len(dc) >= 10:
            feat = ["dow", "lag_1", "lag_2", "lag_3"]
            Xtr = dc[feat].values[:-3]
            ytr = dc["reviews_count"].values[:-3]
            Xte = dc[feat].values[-3:]
            yte = dc["reviews_count"].values[-3:]
            lr = LinearRegression().fit(Xtr, ytr)
            pred = lr.predict(Xte)
            fig2, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["День 1", "День 2", "День 3"], yte, alpha=0.6, label="Факт", color="steelblue")
            ax.bar(["День 1", "День 2", "День 3"], pred, alpha=0.6, label="Прогноз", color="darkorange")
            ax.legend()
            ax.set_title("Прогноз числа отзывов (3 дня)")
            fig2.tight_layout()
            fig2.savefig(os.path.join(BASE_DIR, "forecast_reviews.png"), dpi=120, bbox_inches="tight")
            plt.close(fig2)


def init_app() -> None:
    global vectorizer, best_clf, BEST_MODEL_NAME, METRICS_ROWS, PRODUCT_LIST, TRAIN_NOTE
    df_cat = load_catalog_dataframe()
    assert len(df_cat) >= 200

    train_df = load_sentiment_training_dataframe()
    vectorizer, best_clf, BEST_MODEL_NAME, METRICS_ROWS = train_classifiers(train_df)

    ordered = sorted(METRICS_ROWS, key=lambda r: float(r["f1"]), reverse=True)
    TRAIN_NOTE = (
        f"Сравнение на отложенной выборке: лучший F1 у {ordered[0]['name']} ({ordered[0]['f1']}), "
        f"затем {ordered[1]['name']} ({ordered[1]['f1']}) и {ordered[2]['name']} ({ordered[2]['f1']}). "
        f"Для корпуса Rotten Tomatoes (reviewText + scoreSentiment) используем модель с наибольшим F1."
    )

    init_db()
    if db_review_count() == 0:
        seed_db_from_df(df_cat)

    df_db = load_reviews_from_db()
    top_pos = top_products_by_positive_share(df_db, top_n=12)
    neg_words = five_most_frequent_words_negative(df_db)
    build_student_report(df_db, METRICS_ROWS, BEST_MODEL_NAME, top_pos, neg_words)

    PRODUCT_LIST = df_cat["product"].value_counts().head(300).index.tolist()


init_app()


@app.before_request
def ensure_user() -> None:
    if "user_id" not in session:
        session["user_id"] = uuid.uuid4().hex
        session.permanent = True


@app.route("/")
def index():
    df = load_reviews_from_db()
    uid = session["user_id"]
    recs = user_based_recommendations(df, uid, k=5)
    top_pos = top_products_by_positive_share(df, top_n=8)
    top_pos_summary = ", ".join(f"{r['product']} ({r['pos_share']:.0%})" for _, r in top_pos.iterrows()) or "—"
    neg_words = five_most_frequent_words_negative(df)
    neg_words_summary = ", ".join(f"{w} ({c})" for w, c in neg_words) or "—"
    avg_rating = f"{df['rating'].mean():.2f}" if not df.empty else "—"
    c1, c2, c3 = build_charts_b64(df)
    return render_template(
        "index.html",
        user_id=uid,
        recs=recs,
        metrics_rows=METRICS_ROWS,
        best_model_name=BEST_MODEL_NAME,
        best_model_note=TRAIN_NOTE,
        product_list=PRODUCT_LIST,
        top_pos_summary=top_pos_summary,
        neg_words_summary=neg_words_summary,
        avg_rating=avg_rating,
        chart_ratings=c1,
        chart_daily=c2,
        chart_wc=c3,
    )


@app.route("/submit_review", methods=["POST"])
def submit_review():
    product = request.form.get("product", "").strip()
    text = request.form.get("text", "").strip()
    rating = int(request.form.get("rating", "5"))
    rating = max(1, min(5, rating))
    uid = session["user_id"]
    label, _ = predict_sentiment_with_confidence(text)
    sp = 1 if label == "позитивный" else 0
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO reviews (user_id, user_name, phone, product, text, rating, review_date, sentiment_pos) VALUES (?,?,?,?,?,?,?,?)",
        (uid, mask_name("User"), "", product, text, rating, datetime.now().isoformat(), sp),
    )
    con.commit()
    con.close()
    return redirect(url_for("index"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    text = str(data.get("text", ""))
    label, conf = predict_sentiment_with_confidence(text)
    return jsonify({"label": label, "confidence": conf})


@app.route("/stats/products")
def products_stats():
    df = load_reviews_from_db()
    rows = []
    if not df.empty:
        for prod, g in df.groupby("product"):
            avg_r = float(g["rating"].mean())
            n = int(len(g))
            pos_share = float(g["sentiment_pos"].mean()) if n else 0.0
            rows.append({"product": prod, "avg_rating": avg_r, "n_reviews": n, "pos_share": pos_share})
        rows.sort(key=lambda x: -x["n_reviews"])
    uid = session.get("user_id", "")
    return render_template("products.html", product_rows=rows[:200], user_id=uid)


if __name__ == "__main__":
    print(f"База: {DB_PATH}, отчёт: {REPORT_PATH}")
    print(f"Лучшая модель тональности: {BEST_MODEL_NAME}")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
