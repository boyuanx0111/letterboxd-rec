# Letterboxd AI Movie Recommender 🎬

A personalised movie recommendation system that uses Letterboxd user data and machine learning to understand individual viewing preferences and generate tailored movie recommendations.

The project combines **content-based filtering**, **collaborative filtering**, and **hybrid recommendation techniques** to create a Netflix-style recommendation experience designed specifically for Letterboxd users.

---

## 🚀 Project Overview

Letterboxd provides users with rich movie taste data through ratings, watch history, diary entries, and watchlists. This project aims to transform that data into meaningful recommendations by learning:

- What genres, directors, and themes a user prefers
- Which users have similar movie tastes
- Which unseen movies are most likely to match their interests

### Input

Users provide:

- Letterboxd CSV export
  - Ratings
  - Watched movies
  - Diary history
  - Watchlist

### Output

The system generates:

- Personalised movie recommendations
- Similar movie suggestions
- Taste profiles
- Recommendation explanations

Example:

> "Recommended because you rated *Parasite*, *Memories of Murder*, and other Korean thrillers highly."

---

# 🏗️ System Architecture

```
                    User
                     |
                     v
              Next.js Frontend
                     |
                     v
              FastAPI Backend
                     |
        +------------+------------+
        |                         |
        v                         v
 PostgreSQL Database        ML Recommendation Engine
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
          Content-Based Model          Collaborative Filtering Model
          (Movie Similarity)           (User Preference Learning)
                    |                               |
                    +---------------+---------------+
                                    |
                                    v
                         Hybrid Ranking System
                                    |
                                    v
                         Personalised Recommendations
```

---

# ✨ Features

## Current Goals

### 📂 Letterboxd Data Import
- Upload Letterboxd CSV exports
- Process user ratings and watch history
- Convert raw data into recommendation-ready formats

### 🎥 Movie Understanding
Movie information is collected using external movie metadata sources:

- Genres
- Directors
- Cast
- Release year
- Plot descriptions
- Keywords

### 🤖 Recommendation Engine

The system combines multiple recommendation approaches:

### Content-Based Filtering

Recommends movies based on similarity between films a user enjoys.

Uses:

- Movie metadata
- Text embeddings
- Genre preferences
- Director preferences

Example:

```
User likes:
- The Godfather
- Goodfellas
- Casino

Recommendations:
- The Departed
- Heat
- Once Upon a Time in America
```

---

### Collaborative Filtering

Learns from patterns across users.

Uses:

- User-movie rating interactions
- Similar user preferences
- Matrix factorisation techniques

Example:

```
Users with similar taste enjoyed:
- Oldboy
- Burning
- Memories of Murder
```

---

### Hybrid Recommendation Model

Combines multiple signals:

```
Recommendation Score =

50% Collaborative Filtering
30% Content Similarity
20% User Preferences
```

---

# 🛠️ Tech Stack

## Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS

## Backend

- Python
- FastAPI
- SQLAlchemy
- Pydantic

## Database

- PostgreSQL

## Machine Learning

- Pandas
- NumPy
- Scikit-learn
- LightFM
- Sentence Transformers

## Data Sources

- Letterboxd CSV exports
- TMDB API

## Deployment

- Vercel (Frontend)
- Render/Railway (Backend)
- Supabase (Database)

---

# 📁 Project Structure

```
letterboxd-ai/

├── frontend/              # Next.js web application
│
├── backend/               # FastAPI backend service
│   ├── routes/
│   ├── services/
│   └── database/
│
├── ml/                    # Machine learning models
│   ├── models/
│   ├── training/
│   └── notebooks/
│
├── pipeline/              # Data processing workflows
│
├── data/
│   └── example/           # Sample datasets
│
├── docs/                  # Documentation
│
├── README.md
└── .gitignore
```

---

# ⚙️ Getting Started

## Prerequisites

Install:

- Python 3.11+
- Node.js
- PostgreSQL
- Git

---

## Backend Setup

Clone the repository:

```bash
git clone https://github.com/yourusername/letterboxd-ai.git

cd letterboxd-ai
```

Create Python environment:

```bash
python -m venv venv
```

Activate:

Windows:

```powershell
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run backend:

```bash
uvicorn backend.main:app --reload
```

---

## Frontend Setup

Navigate to frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Run:

```bash
npm run dev
```

---

# 📊 Machine Learning Evaluation

Recommendation quality will be measured using:

## RMSE

Measures how accurately the model predicts user ratings.

## Precision@K

Measures how many recommended movies are relevant.

## Recall@K

Measures how many relevant movies are successfully discovered.

---

# 🗺️ Development Roadmap

## Phase 1 — Foundation

- [ ] Setup repository structure
- [ ] Create database schema
- [ ] Import Letterboxd data
- [ ] Integrate TMDB metadata

## Phase 2 — Recommendation Engine

- [ ] Build popularity baseline
- [ ] Implement content-based model
- [ ] Implement collaborative filtering
- [ ] Create hybrid ranking system

## Phase 3 — Application

- [ ] Build FastAPI endpoints
- [ ] Connect recommendation service
- [ ] Create frontend interface

## Phase 4 — Improvements

- [ ] Recommendation explanations
- [ ] User taste profiles
- [ ] Letterboxd-style yearly insights
- [ ] Improved ranking algorithms

---

# 👥 Team Workflow

Development uses GitHub branching:

```
main
 |
dev
 |
+-- feature/data-pipeline
+-- feature/ml-model
+-- feature/backend-api
+-- feature/frontend
```

Workflow:

1. Create feature branch
2. Develop and commit changes
3. Open Pull Request
4. Merge into `dev`
5. Release stable versions into `main`

---

# 🎯 Project Goal

Build a personalised movie discovery platform that combines machine learning and user preference modelling to deliver recommendations tailored to each Letterboxd user's unique taste.
