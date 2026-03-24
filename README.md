# 🎬 Netflix Movie Recommendation System

<div align="center">

![Netflix](https://img.shields.io/badge/Netflix-E50914?style=for-the-badge&logo=netflix&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

<br/>

> **A complete Content-Based Movie Recommendation System built on ~9,800 Netflix titles.**
> Three recommendation engines + IMDB Weighted Rating — all from scratch using NLP & Machine Learning.

<br/>


</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [How It Works](#-how-it-works)
- [Three Recommendation Engines](#-three-recommendation-engines)
- [IMDB Weighted Rating](#-imdb-weighted-rating-bonus)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Notebook Sections](#-notebook-sections)
- [Getting Started](#-getting-started)
- [Requirements](#-requirements)
- [Results & Quality Analysis](#-results--quality-analysis)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This project builds a **full Movie Recommendation System** from scratch on the Netflix TMDB dataset using **Natural Language Processing (NLP)** and **Machine Learning** techniques — no external APIs, no pre-trained models, just pure data science.

### What Makes This Project Unique?
- ✅ **3 independent recommendation engines** with quality comparison
- ✅ **IMDB Bayesian Weighted Rating** formula implemented from scratch
- ✅ **Genre Overlap Score** metric to evaluate recommendation quality
- ✅ **Interactive search** — find any movie by partial title
- ✅ **Side-by-side engine comparison** with visualizations
- ✅ Clean, well-commented code with **docstrings** for every function
- ✅ **18 structured sections** — beginner to advanced

---

## 🚀 Live Demo

> 📓 View the notebook on Kaggle: **[Netflix Recommendation System — Kaggle Notebook](#https://www.kaggle.com/code/kp009m/netflix-recommendation-system)**

```python
# One-line usage after setup
recommend_for_me('Spider-Man: No Way Home', engine='hybrid', top_n=10)
```

```
────────────────────────────────────────────────────────────
🎬 You watched : Spider-Man: No Way Home
🎭 Genre       : Action, Adventure, Science Fiction
⭐ Rating      : 8.3
📅 Year        : 2021
────────────────────────────────────────────────────────────
🍿 [HYBRID ENGINE] Recommendations:

    Title                              Similarity  Genre                              Vote_Average
1   Spider-Man: Far From Home          0.6821      Action, Adventure, Science Fiction  7.9
2   Spider-Man                         0.6134      Action, Adventure, Science Fiction  7.3
3   Spider-Man 3                       0.5890      Action, Adventure, Science Fiction  6.2
4   Avengers: Infinity War             0.4521      Action, Adventure, Science Fiction  8.3
5   Doctor Strange in the Multiverse  0.4103      Action, Adventure, Fantasy          7.3
...
```

---

## 🧠 How It Works

### Content-Based Filtering

This system uses **Content-Based Filtering** — it recommends movies similar to what you already like, based on the movie's own features (plot, genre, language).

```
Input Movie
     │
     ▼
Feature Extraction
(TF-IDF / CountVectorizer)
     │
     ▼
Vector Representation
[0.12, 0.45, 0.00, 0.89, ...]
     │
     ▼
Cosine Similarity
(Compare with all movies)
     │
     ▼
Top-N Most Similar Movies
```

### TF-IDF Formula

$$TF\text{-}IDF(t, d) = TF(t,d) \times \log\left(\frac{N}{df(t)}\right)$$

- **TF(t, d)** — how often term *t* appears in document *d*
- **IDF(t)** — penalises common words like "the", "is", "a"
- Result: each movie becomes a **high-dimensional vector**

### Cosine Similarity

$$\cos(\theta) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

- Score **1.0** → identical (same movie)
- Score **0.5** → moderately similar
- Score **0.0** → completely different

---

## 🎛️ Three Recommendation Engines

| Engine | Method | Input Features | Best For |
|--------|--------|----------------|----------|
| **Engine 1** | TF-IDF + Cosine Similarity | Movie Overview / Plot | Similar storylines & themes |
| **Engine 2** | CountVectorizer + Cosine Similarity | Genre Tags | Same genre category |
| **Engine 3** | Hybrid TF-IDF + Cosine Similarity | Genre (×2) + Overview + Language | Best overall balance |

### Engine Architecture

```
netflix.csv
    │
    ├─── Engine 1 ──► TF-IDF(Overview)   ──► Cosine Sim ──► Plot-Based Recs
    │
    ├─── Engine 2 ──► CountVec(Genre)    ──► Cosine Sim ──► Genre-Based Recs
    │
    └─── Engine 3 ──► TF-IDF(Genre×2
                          + Overview       ──► Cosine Sim ──► Hybrid Recs
                          + Language)
```

---

## 🏆 IMDB Weighted Rating (Bonus)

A **Bayesian Weighted Rating** prevents low-vote movies with perfect scores from ranking unrealistically high.

$$WR = \frac{v}{v+m} \cdot R + \frac{m}{v+m} \cdot C$$

| Variable | Meaning |
|----------|---------|
| **v** | Number of votes for the movie |
| **m** | Minimum votes required (70th percentile threshold) |
| **R** | Average rating of the movie |
| **C** | Mean rating across all movies |

### Top 5 by Weighted Rating
| Rank | Title | Weighted Rating |
|------|-------|----------------|
| 1 | The Shawshank Redemption | 8.61 |
| 2 | The Godfather | 8.54 |
| 3 | Schindler's List | 8.48 |
| 4 | The Dark Knight | 8.47 |
| 5 | Pulp Fiction | 8.41 |

---

## 📂 Dataset

**Source:** TMDB (The Movie Database) via Kaggle
**File:** `netflix.csv`
**Size:** ~9,800 rows × 9 columns

| Column | Type | Description |
|--------|------|-------------|
| `Release_Date` | datetime | Release date of the title |
| `Title` | string | Name of the movie / TV show |
| `Overview` | string | Short synopsis / plot description |
| `Popularity` | float | TMDB popularity score |
| `Vote_Count` | int | Total number of user votes |
| `Vote_Average` | float | Average user rating (0–10) |
| `Original_Language` | string | ISO language code (e.g. `en`, `ja`, `ko`) |
| `Genre` | string | Comma-separated list of genres |
| `Poster_Url` | string | URL to official poster image |

**Engineered Features:**

| Feature | Description |
|---------|-------------|
| `Overview_Clean` | Lowercased, punctuation-removed overview |
| `Genre_Tags` | Genre as underscore-separated tokens |
| `Tags` | Combined Genre (×2) + Overview + Language |
| `Weighted_Rating` | IMDB Bayesian weighted rating score |
| `Release_Year` | Extracted year from Release_Date |

---

## 🗂️ Project Structure

```
netflix-recommendation-system/
│
├── 📓 Netflix_Recommendation_System.ipynb   ← Main notebook (18 sections)
├── 📄 README.md                             ← Project documentation
├── 📊 netflix.csv                           ← Dataset (place here before running)
```

---

## 📋 Notebook Sections

```
Section 01 — 📦 Import Libraries
Section 02 — 📥 Load & Inspect Dataset
Section 03 — 🧹 Data Cleaning & Preprocessing
           ├── Drop nulls in critical columns
           ├── Type conversions (date, numeric)
           ├── Fill missing values with median
           ├── Remove duplicates
           └── Text cleaning (lowercase, strip punctuation)
Section 04 — 📊 Exploratory Analysis
           ├── Rating & Popularity distributions
           └── Top genres bar chart
Section 05 — 🏆 IMDB Weighted Rating
           ├── Formula explanation
           ├── Top 15 by weighted rating
           └── Top 10 by popularity
Section 06 — 🔤 TF-IDF Vectorization
           ├── Theory + formula
           ├── Build TF-IDF matrix on Overview
           └── Visualise top TF-IDF terms
Section 07 — 🎭 Genre-Based Engine (Engine 2)
           └── CountVectorizer on Genre Tags
Section 08 — 🔀 Hybrid Engine (Engine 3)
           └── TF-IDF on Genre + Overview + Language
Section 09 — 🎯 Recommendation Functions
           ├── get_recommendations()
           └── get_genre_recommendations()
Section 10 — Engine 1 Tests (Overview-Based)
Section 11 — Engine 2 Tests (Genre-Based)
Section 12 — Engine 3 Tests (Hybrid)
Section 13 — ⚖️ Side-by-Side Engine Comparison
Section 14 — 🕵️ Similarity Score Deep Dive
Section 15 — 🔍 Interactive Search Functions
           ├── search_movie()
           └── recommend_for_me()
Section 16 — 📊 Recommendation Quality Analysis
           └── Genre Overlap Score metric
Section 17 — 💾 Export Results
Section 18 — 📝 Summary & Conclusions
```

---


## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/netflix-recommendation-system.git
cd netflix-recommendation-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place `netflix.csv` in the root directory (same folder as the notebook).

### 4. Create image folder

```bash
mkdir images outputs
```

### 5. Launch the notebook

```bash
jupyter notebook Netflix_Recommendation_System.ipynb
```

### 6. Run all cells

**Kernel → Restart & Run All**

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

Install everything:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel
```

---

## 📊 Results & Quality Analysis

Genre Overlap Score measures what % of recommended movies share at least one genre with the input movie.

| Movie | Engine 1 (Plot) | Engine 2 (Genre) | Engine 3 (Hybrid) |
|-------|:-:|:-:|:-:|
| Spider-Man: No Way Home | 80% | 100% | 90% |
| The Batman | 70% | 100% | 80% |
| Encanto | 60% | 100% | 80% |
| Avengers: Endgame | 80% | 100% | 90% |

**Key takeaway:** Engine 3 (Hybrid) delivers the best real-world recommendations by balancing both plot themes and genre relevance.

---

## ⚠️ Limitations

- **Content-Based only** — no user history or collaborative filtering
- **Cold start** — new titles added after training won't appear in recommendations
- **Language bias** — ~77% of overviews are in English, dominating TF-IDF space
- **No actor/director features** — cast and crew data not available in this dataset
- **Static model** — similarity matrices are precomputed; not updated in real time

---

## 🔮 Future Work

- [ ] Add **Collaborative Filtering** using user ratings (SVD, ALS, NMF)
- [ ] Use **BERT / Sentence-Transformers** for deep semantic plot embeddings
- [ ] Incorporate **cast, director, and keywords** as additional features
- [ ] Build a **Streamlit web app** with live search, poster display, and engine toggle
- [ ] Implement **Matrix Factorization** (SVD++) for hybrid collaborative filtering
- [ ] Add a **multi-language NLP pipeline** to handle non-English overviews
- [ ] Deploy as a **Kaggle inference notebook** with interactive widgets

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create your branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please follow PEP 8 style and include comments for any new functions.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset sourced from **TMDB (The Movie Database)** via Kaggle
- Recommendation logic inspired by classic content-based filtering literature
- Weighted rating formula adapted from **IMDB's official methodology**
- Visualizations powered by **Matplotlib** and **Seaborn**

---

<div align="center">

Made with ❤️ and 🐍 Python

⭐ **Star this repo** if you found it helpful!

🔗 Also check out the companion **[Netflix EDA Notebook](#https://github.com/kashyap09m/Netflix_Data_Analysis)**

</div>
