# 🧠 HR Staffing Candidate Ranking System

A machine learning project for ranking and re-ranking job candidates based on their relevance to a given role, combining semantic search with learning-to-rank techniques.

---

## 🚀 Overview

Recruiting the right talent is a complex and time-consuming process. This project aims to **automate candidate ranking** by leveraging Natural Language Processing (NLP) and machine learning.

The system:
1. **Ranks candidates initially** using semantic similarity (BERT)
2. **Incorporates human feedback** (starred candidates)
3. **Re-ranks candidates dynamically** using a pairwise learning-to-rank model (RankNet)

This approach reduces manual effort while continuously improving ranking quality based on recruiter input.

---

## 🎯 Problem Statement

Given a list of candidates and a target role (defined via keywords like *"Aspiring Human Resources"*), the goal is to:

- Predict a **fit score (0–1)** for each candidate
- Rank candidates based on relevance
- Improve ranking using **human-in-the-loop feedback**

---

## 🧩 Pipeline Architecture

### 1️⃣ Initial Ranking — BERT Similarity

- Convert candidate profiles and job keywords into embeddings
- Compute similarity scores using BERT
- Rank candidates based on semantic relevance

➡️ Output: Initial ranked candidate list

---

### 2️⃣ Human Feedback (Starring)

- Recruiters manually review candidates
- A selected candidate is **starred** as the “ideal fit”
- This acts as a **supervisory signal**

---

### 3️⃣ Re-Ranking — RankNet (Pairwise Learning)

- Use the starred candidates to generate pairwise comparisons
- Train a RankNet model to learn preference ordering
- Re-rank the candidate list based on learned preferences

➡️ Output: Improved ranking aligned with recruiter judgment

---

## 📊 Dataset

The dataset consists of anonymized candidate information collected from sourcing efforts.

| Feature        | Description |
|----------------|------------|
| `id`           | Unique identifier for candidate |
| `job_title`    | Candidate’s job title |
| `location`     | Candidate’s geographic location |
| `connections`  | Number of connections (e.g., 500+) |

---

## 🔑 Keywords Used

Example search queries:
- "Aspiring human resources"

> The system is flexible and supports dynamic keyword inputs.

---

## 🛠️ Tech Stack

- Python  
- BERT (Transformers) — semantic similarity  
- RankNet — pairwise ranking model  
- Pandas / NumPy — data processing  
- Scikit-learn / PyTorch / TensorFlow — modeling (depending on implementation)  

---

## 📁 Repository Structure

```
HR_Staffing/
│
├── data/ 
│
├── results/ 
│
├── text_preprocessor.py       # Data cleaning and feature preparation
├── visualization_generator
├── pairwise.py                       
├── ranknet.py                 # RankNet
├── ranking_system.py      
│
├── main.py
│
├── LICENSE                    # MIT License
├── README.md 
└── .gitignore
```

## ⚙️ Installation

```bash
git clone https://github.com/xuanwei20/HR_Staffing.git
cd HR_Staffing
python main.py
```
