# ============================================
# SISTEM REKOMENDASI PONSEL - VERSI REFACTORED
# Dengan TF-IDF Brand Filtering & Statistical Aggregation
# ============================================

# 1. Install Dependencies
print("Installing dependencies...")
!pip install gradio kagglehub plotly scikit-learn -q
print("Dependencies installed!\n")

# 2. Import Libraries
print("Importing libraries...")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
from scipy import stats

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

print("Libraries imported!\n")

# 3. Download & Load Dataset
print("Downloading Mobile Phone Rating dataset...\n")

import kagglehub

path = kagglehub.dataset_download("prasertk/mobile-phone-rating")
print(f"Dataset downloaded to: {path}\n")

# Load CSV
data_files = os.listdir(path)
csv_file = [f for f in data_files if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))

print(f"Dataset loaded!")
print(f"   Shape: {df.shape}")
print(f"   Rows: {df.shape[0]:,}")
print(f"   Columns: {df.shape[1]}\n")

print("First 10 rows:")
display(df.head(10))

# 4. Data Preprocessing & Feature Engineering
print("\n" + "="*70)
print("DATA PREPROCESSING & FEATURE ENGINEERING")
print("="*70 + "\n")

# Clean data
print("Cleaning data...")
df_clean = df.copy()

# Remove duplicates
initial_rows = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=['model'])
print(f"  Removed {initial_rows - len(df_clean)} duplicate models")

# Handle missing values
df_clean = df_clean.dropna(subset=['model'])
print(f"  Removed rows with missing model names")

# Remove price column if exists
if 'price' in df_clean.columns:
    df_clean = df_clean.drop('price', axis=1)
    print("  Removed price column")

# Fill numeric missing values with median
numeric_cols = ['camera', 'selfie', 'audio', 'display', 'battery']
print("\nFilling missing values...")
for col in numeric_cols:
    if col in df_clean.columns:
        before = df_clean[col].isnull().sum()
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        after = df_clean[col].isnull().sum()
        if before > 0:
            print(f"  {col}: filled {before} missing values")

print(f"\nFinal shape: {df_clean.shape}")
print("\nCleaned dataset ready!")

# 5. FITUR BARU - TF-IDF Brand Filtering
print("\n" + "="*70)
print("FITUR BARU: TF-IDF BRAND FILTERING SETUP")
print("="*70 + "\n")

# Setup TF-IDF Vectorizer
def setup_tfidf_vectorizer():
    tfidf = TfidfVectorizer(
        lowercase=True,
        max_features=100,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1
    )
    return tfidf

print("Initializing TF-IDF Vectorizer...")
tfidf_vectorizer = setup_tfidf_vectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df_clean['model'])
print(f"TF-IDF matrix created: {tfidf_matrix.shape}")

# Brand filtering function
def extract_brand_group_tfidf(brand_keyword, top_n=None):
    print(f"\n[BRAND FILTERING] Mencari ponsel dengan keyword: '{brand_keyword}'")
    brand_keyword_lower = brand_keyword.lower().strip()
    known_brands = ['apple', 'samsung', 'xiaomi', 'oppo', 'vivo', 'realme',
                    'oneplus', 'google', 'huawei', 'nokia', 'motorola',
                    'sony', 'lg', 'asus', 'lenovo', 'htc', 'blackberry']
    detected_brand = None
    for brand in known_brands:
        if brand in brand_keyword_lower:
            detected_brand = brand
            print(f"  Brand terdeteksi: '{detected_brand}'")
            break
    if detected_brand is None:
        detected_brand = brand_keyword_lower.split()[0]
        print(f"  Menggunakan kata pertama sebagai brand: '{detected_brand}'")
    df_temp = df_clean.copy()
    mask = df_temp['model'].str.lower().str.contains(detected_brand, na=False)
    brand_group = df_temp[mask].copy()
    print(f"  String matching: ditemukan {len(brand_group)} ponsel dari brand '{detected_brand}'")
    if len(brand_group) == 0:
        print(f"  Tidak ada ponsel yang cocok dengan brand '{detected_brand}'")
        return brand_group
    keyword_vector = tfidf_vectorizer.transform([detected_brand])
    tfidf_scores = cosine_similarity(keyword_vector,
                                     tfidf_vectorizer.transform(brand_group['model'])).flatten()
    brand_group['tfidf_score'] = tfidf_scores
    brand_group = brand_group.sort_values('tfidf_score', ascending=False)
    if top_n:
        brand_group = brand_group.head(top_n)
    print(f"  Total hasil akhir: {len(brand_group)} ponsel")
    return brand_group

# Statistical aggregation
def calculate_brand_statistics(brand_df, features=['camera', 'selfie', 'audio', 'display', 'battery']):
    print(f"\n[STATISTICAL AGGREGATION] Menghitung statistik untuk {len(brand_df)} ponsel")
    stats_dict = {}
    for feature in features:
        if feature in brand_df.columns:
            values = brand_df[feature].dropna()
            median_val = values.median()
            mean_val = values.mean()
            try:
                mode_val = stats.mode(values, keepdims=True).mode[0]
            except:
                mode_val = values.mode()[0] if len(values.mode()) > 0 else median_val
            stats_dict[feature] = {
                'median': median_val,
                'mean': mean_val,
                'mode': mode_val
            }
    return stats_dict

# 6. Feature Preparation untuk Similarity
print("\n" + "="*70)
print("PREPARING FEATURES FOR SIMILARITY CALCULATION")
print("="*70 + "\n")

feature_columns = ['camera', 'selfie', 'audio', 'display', 'battery']
df_features = df_clean[feature_columns].copy()
df_features = df_features.fillna(df_features.median())
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_features)
df_features_scaled = pd.DataFrame(features_scaled, columns=feature_columns, index=df_clean.index)
cosine_sim_global = cosine_similarity(df_features_scaled)
cosine_sim_df = pd.DataFrame(cosine_sim_global, index=df_clean['model'], columns=df_clean['model'])

# 7. FUNGSI REKOMENDASI UTAMA (REFACTORED)
def get_recommendations_brand_only(brand_keyword, stat_type='median', n_recommendations=10):
    try:
        exact_match = None
        if brand_keyword in df_clean['model'].values:
            exact_match = brand_keyword
        else:
            matches = df_clean[df_clean['model'].str.lower() == brand_keyword.lower()]
            if len(matches) > 0:
                exact_match = matches.iloc[0]['model']
            else:
                matches = df_clean[df_clean['model'].str.lower().str.contains(brand_keyword.lower(), na=False)]
                if len(matches) == 1:
                    exact_match = matches.iloc[0]['model']
                elif len(matches) > 1:
                    close_matches = [m for m in matches['model'].values if brand_keyword.lower() in m.lower()]
                    if len(close_matches) == 1:
                        exact_match = close_matches[0]
        brand_group = extract_brand_group_tfidf(brand_keyword)
        if len(brand_group) == 0:
            return f"Tidak ditemukan ponsel dengan keyword '{brand_keyword}'", None, None
        reference_profile = []
        reference_profile_dict = {}
        profile_source = ""
        if exact_match:
            phone_data = df_clean[df_clean['model'] == exact_match].iloc[0]
            for feature in feature_columns:
                value = phone_data[feature]
                reference_profile.append(value)
                reference_profile_dict[feature] = value
            profile_source = f"REAL SPECS: {exact_match}"
        else:
            brand_stats = calculate_brand_statistics(brand_group, feature_columns)
            for feature in feature_columns:
                value = brand_stats[feature][stat_type]
                reference_profile.append(value)
                reference_profile_dict[feature] = value
            profile_source = f"VIRTUAL PROFILE ({stat_type.upper()})"
        reference_profile_scaled = scaler.transform([reference_profile])
        brand_indices = brand_group.index
        brand_features_scaled = df_features_scaled.loc[brand_indices]
        similarities = cosine_similarity(reference_profile_scaled, brand_features_scaled).flatten()
        results_df = brand_group.copy()
        results_df['similarity'] = similarities
        if exact_match:
            results_df = results_df[results_df['model'] != exact_match]
        top_recommendations = results_df.nlargest(n_recommendations, 'similarity')
        detected_brand = brand_keyword.split()[0] if exact_match else brand_keyword
        result_text = f"# Rekomendasi Ponsel dari Brand '{detected_brand.upper()}'\n\n"
        if exact_match:
            result_text += f"**Mode:** SPECIFIC PHONE (Real Specs)\n"
            result_text += f"**Reference Phone:** {exact_match}\n"
        else:
            result_text += f"**Mode:** BRAND STATISTICS\n"
            result_text += f"**Metode:** TF-IDF Filtering + Statistical Aggregation ({stat_type.upper()})\n"
        result_text += f"**Total ponsel di grup:** {len(brand_group)}\n"
        result_text += f"**Menampilkan:** Top {min(n_recommendations, len(top_recommendations))} dari brand ini\n\n"
        result_text += f"## Reference Profile ({profile_source}):\n"
        for feature, value in reference_profile_dict.items():
            result_text += f"- **{feature.title()}:** {value:.2f}\n"
        if exact_match:
            result_text += f"\n## Ponsel Serupa dengan {exact_match}:\n\n"
        else:
            result_text += f"\n## Top {min(n_recommendations, len(top_recommendations))} Rekomendasi:\n\n"
        recommendations = []
        for i, (idx, row) in enumerate(top_recommendations.iterrows(), 1):
            result_text += f"### {i}. **{row['model']}**\n"
            result_text += f"   - Similarity: **{row['similarity']:.4f}**\n"
            result_text += f"   - Camera: {row['camera']:.1f} | Display: {row['display']:.1f} | Battery: {row['battery']:.1f}\n\n"
            recommendations.append({
                'Rank': i,
                'Model': row['model'],
                'Similarity': f"{row['similarity']:.4f}",
                'Camera': row['camera'],
                'Display': row['display'],
                'Battery': row['battery'],
                'Audio': row['audio'],
                'Selfie': row['selfie']
            })
        result_df = pd.DataFrame(recommendations)
        return result_text, result_df, None
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"Error: {str(e)}\n\nDetail:\n{error_detail}", None, None

# Fungsi rekomendasi original (masih ada untuk kompatibilitas)
def get_similar_phones_content_based(phone_model, n_recommendations=10):
    try:
        if phone_model not in cosine_sim_df.index:
            matches = [p for p in cosine_sim_df.index if phone_model.lower() in p.lower()]
            if matches:
                phone_model = matches[0]
            else:
                return f"Ponsel '{phone_model}' tidak ditemukan!", None, None
        similar_scores = cosine_sim_df[phone_model].sort_values(ascending=False)
        similar_phones = similar_scores.iloc[1:n_recommendations+1]
        recommendations = []
        for phone, score in similar_phones.items():
            phone_data = df_clean[df_clean['model'] == phone].iloc[0]
            recommendations.append({
                'Model': phone,
                'Similarity': f"{score:.4f}",
                'Camera': phone_data.get('camera', 'N/A'),
                'Selfie': phone_data.get('selfie', 'N/A'),
                'Display': phone_data.get('display', 'N/A'),
                'Battery': phone_data.get('battery', 'N/A'),
                'Audio': phone_data.get('audio', 'N/A')
            })
        result_df = pd.DataFrame(recommendations)
        result_text = f"# Similar phones to **{phone_model}**\n\nTop {n_recommendations} results based on cosine similarity."
        return result_text, result_df, None
    except Exception as e:
        return f"Error: {str(e)}", None, None

# Fungsi utilitas lainnya
def search_phones(query, max_results=20):
    try:
        matches = [phone for phone in df_clean['model'] if query.lower() in phone.lower()]
        if not matches:
            return "Tidak ditemukan!", None
        results = []
        for phone in matches[:max_results]:
            phone_data = df_clean[df_clean['model'] == phone].iloc[0]
            results.append({
                'Model': phone,
                'Camera': phone_data.get('camera', 'N/A'),
                'Selfie': phone_data.get('selfie', 'N/A'),
                'Display': phone_data.get('display', 'N/A'),
                'Battery': phone_data.get('battery', 'N/A'),
                'Audio': phone_data.get('audio', 'N/A'),
            })
        result_df = pd.DataFrame(results)
        result_text = f"# Hasil Pencarian: '{query}'\nDitemukan {len(matches)} ponsel"
        return result_text, result_df
    except Exception as e:
        return f"Error: {str(e)}", None

print("All functions defined!\n")

# 8. GRADIO APP (REFACTORED WITH NEW FEATURES)
def create_advanced_phone_finder():
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', sans-serif;
        max-width: 1400px;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
    }
    """
    with gr.Blocks(css=custom_css, title="Advanced Phone Finder", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # Advanced Phone Recommendation System
        ### Dengan TF-IDF Brand Filtering & Statistical Aggregation
        **Author:** rasiharunart1 | **Updated:** 2025-10-30
        ---
        """)
        with gr.Tab("Brand-Based Recommendations"):
            gr.Markdown("""
            ### Rekomendasi Berdasarkan Brand Group Statistics
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Configuration")
                    brand_input = gr.Textbox(
                        label="Brand Keyword",
                        placeholder="e.g., apple, samsung, xiaomi, oppo",
                        info="Masukkan nama brand untuk filtering",
                        lines=1
                    )
                    stat_method = gr.Radio(
                        choices=['median', 'mean', 'mode'],
                        value='median',
                        label="Metode Statistik",
                        info="Pilih metode agregasi untuk profil virtual"
                    )
                    n_recs_brand = gr.Number(
                        label="Jumlah Rekomendasi",
                        value=10,
                        minimum=1,
                        maximum=50,
                        step=1,
                        info="Masukkan jumlah hasil rekomendasi"
                    )
                    find_brand_btn = gr.Button("Generate Recommendations", variant="primary", size="lg")
                with gr.Column(scale=2):
                    gr.Markdown("#### Hasil Rekomendasi")
                    brand_output = gr.Markdown()
                    brand_table = gr.Dataframe(wrap=True)
            gr.Markdown("#### Coba contoh brand:")
            with gr.Row():
                ex_apple = gr.Button("Apple", size="sm")
                ex_samsung = gr.Button("Samsung", size="sm")
                ex_xiaomi = gr.Button("Xiaomi", size="sm")
                ex_oppo = gr.Button("Oppo", size="sm")
            ex_apple.click(lambda: "apple", outputs=[brand_input])
            ex_samsung.click(lambda: "samsung", outputs=[brand_input])
            ex_xiaomi.click(lambda: "xiaomi", outputs=[brand_input])
            ex_oppo.click(lambda: "oppo", outputs=[brand_input])
            find_brand_btn.click(
                fn=get_recommendations_brand_only,
                inputs=[brand_input, stat_method, n_recs_brand],
                outputs=[brand_output, brand_table]
            )
        with gr.Tab("Find Similar Phones (Original)"):
            gr.Markdown("### Cari ponsel serupa berdasarkan model spesifik")
            with gr.Row():
                with gr.Column(scale=1):
                    phone_input = gr.Textbox(
                        label="Phone Model",
                        placeholder="e.g., iPhone 14 Pro",
                        lines=1
                    )
                    n_similar = gr.Number(
                        label="Jumlah Rekomendasi",
                        value=10,
                        minimum=1,
                        maximum=50,
                        step=1
                    )
                    find_btn = gr.Button("Find Similar", variant="primary")
                with gr.Column(scale=2):
                    similar_output = gr.Markdown()
                    similar_table = gr.Dataframe()
            find_btn.click(
                get_similar_phones_content_based,
                [phone_input, n_similar],
                [similar_output, similar_table]
            )
        with gr.Tab("Search Phones"):
            gr.Markdown("### Cari ponsel berdasarkan nama atau brand")
            with gr.Row():
                search_input = gr.Textbox(label="Search Query", placeholder="e.g., iPhone, Galaxy")
                search_btn = gr.Button("Search", variant="primary")
            search_output = gr.Markdown()
            search_table = gr.Dataframe()
            search_btn.click(search_phones, [search_input], [search_output, search_table])
        with gr.Tab("About"):
            gr.Markdown(f"""
            ## Dataset Information
            - **Total Phones:** {len(df_clean):,}
            - **Features:** {len(feature_columns)}
            - **TF-IDF Vocabulary:** {tfidf_matrix.shape[1]} terms
            ## New Features
            - TF-IDF + String Matching Brand Filtering
            - Statistical Aggregation (Median/Mean/Mode)
            - Virtual Profile Generation
            - Dynamic Recommendation Control
            ## Algoritma
            - Brand Detection, String Matching, TF-IDF Ranking
            - Statistical Aggregation, Cosine Similarity, StandardScaler
            ## Mathematical Formula
            Z-Score: z = (x - μ) / σ
            Cosine Similarity: (A · B) / (||A|| * ||B||)
            ## Developer
            **Author:** rasiharunart1  
            **Date:** 2025-10-30  
            **Version:** 2.0 (Refactored)
            ---
            """)
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #666;">
            <p>Advanced Phone Recommendation System | rasiharunart1 2025</p>
        </div>
        """)
    return app

print("Gradio app created with advanced features!\n")

# 9. Launch App
print("="*70)
print("LAUNCHING ADVANCED PHONE RECOMMENDATION SYSTEM")
print("="*70 + "\n")
print("ADVANCED PHONE FINDER")
print("Version: 2.0 (Refactored)")
print("Features: TF-IDF + String Matching + Statistical Aggregation + Dynamic Control\n")
app = create_advanced_phone_finder()
app.launch(share=True, debug=True, inbrowser=True)
print("\nApp launched successfully!")