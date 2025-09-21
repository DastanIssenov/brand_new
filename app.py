# app.py
import os
import io
import pickle
import time
import json
import requests
import pandas as pd
import numpy as np
import faiss
import streamlit as st
from urllib.parse import urlparse, parse_qs
from sentence_transformers import SentenceTransformer

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="Comment Classifier → CSV", page_icon="🧩", layout="wide")
st.title("🧩 Comment Classifier → CSV")
st.caption("Paste a YouTube (or optional Instagram) link, classify + answer, and export to CSV.")

# ----------------------------
# Secrets / Environment
# ----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

if not OPENAI_API_KEY:
    st.warning("Add your OpenAI key to `Secrets` as `OPENAI_API_KEY` (or set the env var).")
if not GOOGLE_API_KEY:
    st.info("YouTube comments require a Google API key in `GOOGLE_API_KEY`.")

# ----------------------------
# Helper: Load model (cached)
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_encoder(model_name: str = "intfloat/multilingual-e5-large"):
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=True)
def read_faiss_index(path: str):
    return faiss.read_index(path)

@st.cache_resource(show_spinner=True)
def read_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

# ----------------------------
# Sidebar: Paths / Uploads
# ----------------------------
st.sidebar.header("🔧 Model & Index Settings")

model_name = st.sidebar.text_input(
    "SentenceTransformer model",
    value="intfloat/multilingual-e5-large",
)

# Paths for sentiment / language / class / ban
INDEX_PATH_TEMP = st.sidebar.text_input("Index path (TEMP)", value="index_temp.faiss")
METADATA_PATH_TEMP = st.sidebar.text_input("Metadata path (TEMP)", value="metadata_temp.pkl")

INDEX_PATH_BAN = st.sidebar.text_input("Index path (BAN)", value="index_ban.faiss")
METADATA_PATH_BAN = st.sidebar.text_input("Metadata path (BAN)", value="metadata_ban.pkl")

INDEX_PATH_CLAS = st.sidebar.text_input("Index path (CLAS)", value="index_clas.faiss")
METADATA_PATH_CLAS = st.sidebar.text_input("Metadata path (CLAS)", value="metadata_clas.pkl")

INDEX_PATH_LANG = st.sidebar.text_input("Index path (LANG)", value="index_lang.faiss")
METADATA_PATH_LANG = st.sidebar.text_input("Metadata path (LANG)", value="metadata_lang.pkl")

# Q/A embeddings
INDEX_PATH_Q = st.sidebar.text_input("Index path (Q)", value="index.faiss")
METADATA_PATH_A = st.sidebar.text_input("Metadata path (A)", value="metadata_a.pkl")

# Optional Instagram (Selenium) toggle
enable_instagram = st.sidebar.toggle("Enable experimental Instagram scraping (requires local Selenium/ChromeDriver)", value=False)

# ----------------------------
# Input URL
# ----------------------------
url_input = st.text_input("Paste a YouTube (or Instagram) URL:", placeholder="https://www.youtube.com/watch?v=...")

# ----------------------------
# Utilities
# ----------------------------
def get_video_id(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com", "m.youtube.com"]:
        if parsed_url.path == "/watch":
            qs = parse_qs(parsed_url.query)
            if "v" in qs and qs["v"]:
                return qs["v"][0]
    if parsed_url.hostname in ["youtu.be", "www.youtu.be"]:
        return parsed_url.path.lstrip("/")
    raise ValueError("Not a valid YouTube watch URL")

def build_openai_headers():
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

def openai_chat(model: str, system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers=build_openai_headers(), json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def get_query_embedding(encoder: SentenceTransformer, query: str) -> np.ndarray:
    return encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)

def search_document(encoder: SentenceTransformer, query: str, index, metadata, k: int = 1, threshold: float = 0.3):
    q_vec = get_query_embedding(encoder, query)
    distances, indices = index.search(q_vec, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if distances[0][i] >= threshold:
            results.append(metadata[idx])
    return results

def fetch_youtube_comments(youtube_key: str, video_url: str, max_results: int = 200) -> pd.DataFrame:
    video_id = get_video_id(video_url)
    # Simple one-page fetch (maxResults up to 100); loop if needed
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": min(max_results, 100),
        "key": youtube_key,
        "textFormat": "plainText",
        "order": "relevance",
    }
    comments = []
    page_count = 0
    while True:
        page_count += 1
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("items", []):
            c = item["snippet"]["topLevelComment"]["snippet"]
            comments.append([
                c.get("authorDisplayName", ""),
                c.get("publishedAt", ""),
                c.get("textDisplay", ""),
            ])
        if "nextPageToken" in data and len(comments) < max_results:
            params["pageToken"] = data["nextPageToken"]
        else:
            break
        if page_count > 10:  # sanity bound
            break
    df = pd.DataFrame(comments, columns=["author", "posted_at", "text"])
    return df

def fetch_instagram_comments(url: str) -> pd.DataFrame:
    """
    Very experimental: requires Selenium + ChromeDriver available in the runtime.
    This will likely NOT work on Streamlit Cloud. Use locally.
    """
    from selenium import webdriver
    from bs4 import BeautifulSoup

    df = pd.DataFrame(columns=["author", "posted_at", "text"])
    driver = webdriver.Chrome()
    try:
        driver.get(url)
        time.sleep(5)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        # NOTE: Instagram’s DOM changes frequently; this selector may break.
        divs = soup.find_all("div", class_="html-div xdj266r x14z9mp xat24cr x1lziwak xyri2b x1c1uobl x9f619 xjbqb8w x78zum5 x15mokao x1ga7v0g x16uus16 xbiv7yw xsag5q8 xz9dl7a x1uhb9sk x1plvlek xryxfnj x1c4vz4f x2lah0s x1q0g3np xqjyukv x1qjc9v5 x1oa3qoh x1nhvcw1")
        comments = []
        for div in divs[1:]:
            spans = div.find_all("span")
            if len(spans) >= 4:
                comments.append([
                    spans[1].get_text(strip=True),
                    spans[2].get_text(strip=True),
                    spans[3].get_text(strip=True)
                ])
        if comments:
            df = pd.DataFrame(comments, columns=["author", "posted_at", "text"])
        return df
    finally:
        driver.quit()

# ----------------------------
# Main Action
# ----------------------------
run = st.button("Run Classification")

if run:
    if not url_input:
        st.error("Please paste a URL first.")
        st.stop()

    # Load encoder and indices
    with st.spinner("Loading encoder and indices..."):
        encoder = load_encoder(model_name)

        index_clas = read_faiss_index(INDEX_PATH_CLAS)
        metadata_clas = read_pickle(METADATA_PATH_CLAS)

        index_temp = read_faiss_index(INDEX_PATH_TEMP)
        metadata_temp = read_pickle(METADATA_PATH_TEMP)

        index_lang = read_faiss_index(INDEX_PATH_LANG)
        metadata_lang = read_pickle(METADATA_PATH_LANG)

        index_ban = read_faiss_index(INDEX_PATH_BAN)
        metadata_ban = read_pickle(METADATA_PATH_BAN)

        index_q = read_faiss_index(INDEX_PATH_Q)
        metadata_a = read_pickle(METADATA_PATH_A)

    # Fetch comments
    with st.spinner("Fetching comments..."):
        df = pd.DataFrame(columns=["author", "posted_at", "text"])
        parsed = urlparse(url_input)
        is_youtube = any(h in (parsed.hostname or "") for h in ["youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"])
        if is_youtube:
            if not GOOGLE_API_KEY:
                st.error("YouTube comment fetching requires GOOGLE_API_KEY.")
                st.stop()
            try:
                df = fetch_youtube_comments(GOOGLE_API_KEY, url_input, max_results=200)
            except Exception as e:
                st.exception(e)
                st.stop()
        else:
            if enable_instagram:
                try:
                    df = fetch_instagram_comments(url_input)
                except Exception as e:
                    st.warning("Instagram scraping failed (DOM changes / login walls are common).")
                    st.exception(e)
                    st.stop()
            else:
                st.error("Only YouTube is supported by default. Enable Instagram (experimental) in the sidebar to try scraping.")
                st.stop()

        if df.empty:
            st.warning("No comments found.")
            st.stop()

    st.write(f"Fetched **{len(df)}** comments.")

    # Process rows
    progress = st.progress(0)
    lang_list, temp_list, ban_list, clas_list, answers_list = [], [], [], [], []

    for idx, row in df.iterrows():
        txt = row["text"]

        # 1) Language
        try:
            lang_resp = openai_chat(
                model="gpt-4.1-mini",
                system_prompt="определи язык комментария: русский или казахский. Ответ должен состоять просто из одного слова",
                user_prompt=f"Комментарий: {txt}"
            )
        except Exception as e:
            lang_resp = "русский"  # fallback
        lang_match = search_document(encoder, lang_resp, index_lang, metadata_lang, k=1)
        lang_val = lang_match[0] if lang_match else lang_resp
        lang_list.append(lang_val)

        # 2) Classification
        try:
            clas_resp = openai_chat(
                model="gpt-4o-mini",
                system_prompt="Классифицируй комментарий как одну из этих категорий: вопрос, отзыв, жалоба, благодарность или неизвестный. Ответ должен быть одим из этих слов на русском. Если есть знак вопроса то это скорее всего вопрос",
                user_prompt=f"Комментарий: {txt}"
            )
        except Exception as e:
            clas_resp = "неизвестный"
        clas_match = search_document(encoder, clas_resp, index_clas, metadata_clas, k=1)
        clas_val = clas_match[0] if clas_match else clas_resp
        clas_list.append(clas_val)

        # 3) Sentiment (temp)
        try:
            temp_resp = openai_chat(
                model="gpt-4o-mini",
                system_prompt="определи тон комментария: позитивный, негативный, нейтральный. Ответ должен быть одим из этих слов на русском",
                user_prompt=f"Комментарий: {txt}"
            )
        except Exception as e:
            temp_resp = "нейтральный"
        temp_match = search_document(encoder, temp_resp, index_temp, metadata_temp, k=1)
        temp_val = temp_match[0] if temp_match else temp_resp
        temp_list.append(temp_val)

        # 4) Moderation (ban)
        try:
            ban_resp = openai_chat(
                model="gpt-4o-mini",
                system_prompt="относиться ли комментарий к одной из этих групп: спам, оскорбления, нецензурная лексика. Если да то ответ должен быть одим из этих слов на русском если нет то ответ нет",
                user_prompt=f"Комментарий: {txt}"
            )
        except Exception as e:
            ban_resp = "нет"
        ban_match = search_document(encoder, ban_resp, index_ban, metadata_ban, k=1)
        ban_val = ban_match[0] if ban_match else ban_resp
        ban_list.append(ban_val)

        # 5) Q/A for questions
        if clas_val.strip().lower() == "вопрос":
            chunks = search_document(encoder, txt, index_q, metadata_a, k=3)
            context = " ".join(chunks) if chunks else ""
            try:
                ans = openai_chat(
                    model="gpt-4.1-mini",
                    system_prompt=f"Определи вопрос и ответь на него основываясь на этих данных: {context}. Ответь на этом языке {lang_val}",
                    user_prompt=f"Вопрос: {txt}"
                )
            except Exception as e:
                ans = ""
            answers_list.append(ans)
        else:
            answers_list.append("")

        progress.progress(int(((idx + 1) / len(df)) * 100))

    # Assemble output
    out_df = df.copy()
    out_df["lang"] = lang_list
    out_df["clas"] = clas_list
    out_df["temp"] = temp_list
    out_df["to_ban"] = ban_list
    out_df["answer"] = answers_list

    st.subheader("Preview")
    st.dataframe(out_df.head(50), use_container_width=True)

    # Download
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name="comments_classified.csv",
        mime="text/csv",
    )

    st.success("Done! CSV is ready to download.")
