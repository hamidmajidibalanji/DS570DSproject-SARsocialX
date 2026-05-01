# Search and Rescue (SAR) Social Media Analyzer 

##  Overview

This project builds an end-to-end data science system to detect and analyze emergency help requests in earthquake-related social media data.

The system:

* Collects publicly available tweet data
* Processes and cleans text data
* Trains machine learning models to detect help requests
* Provides an interactive dashboard for exploration and prediction

---

## Objectives

* Detect potential **search & rescue (SAR)** requests in tweets
* Visualize patterns in disaster-related communication
* Provide an interactive tool for exploring emergency signals

---

##  Data Source

The dataset is automatically downloaded from a public online source (no login required).
It contains earthquake-related tweets labeled for humanitarian categories.

---

##  How to Run (Docker)

### 1. Build the image

```bash
docker build -t sar-app .
```

### 2. Run the container

```bash
docker run -p 8501:8501 sar-app
```

### 3. Open in browser

```
http://localhost:8501
```

---

##  Project Structure

* `data/` → raw and processed datasets
* `src/` → data processing and modeling code
* `app/` → Streamlit dashboard
* `Dockerfile` → container setup

---

##  Features (Planned)

* Data ingestion pipeline
* Data cleaning & preprocessing
* ML classification model
* Interactive dashboard

---

##  Notes

* Data is downloaded automatically at runtime
* No manual setup or login required

---

##  Author:

Hamid Majidi Balanji
