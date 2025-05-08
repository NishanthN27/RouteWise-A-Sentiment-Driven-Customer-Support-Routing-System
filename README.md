# RouteWise: A Sentiment-Driven Customer Support Routing System

**RouteWise** is a sentiment-based customer complaint routing system designed to streamline the way support teams handle high volumes of inquiries. By integrating NLP-driven sentiment scoring and a smart routing mechanism, RouteWise ensures that emotionally urgent complaints are prioritized and assigned to the most suitable agents.

---

## Key Features

- Sentiment classification using a fine-tuned BERT model
- Automated complaint prioritization (Highly Dissatisfied → Satisfied)
- Smart routing logic based on agent specialization
- Flask backend API for seamless model integration
- MySQL database for storing complaints, agents, and routing logs
---

## Project Goal

Develop a system that reads customer complaints, classifies them based on emotional tone, and automatically routes high-priority cases to experienced agents—enhancing resolution speed and customer satisfaction.

---

## Dataset Overview

- **Collection:** Manually curated and labeled from multiple sources
- **Sources:** Kaggle, Twitter, Google Reviews, and public review/comment datasets
- **Labels:** 
  - Satisfied
  - Neutral
  - Slightly Dissatisfied
  - Dissatisfied
  - Highly Dissatisfied
- **Size:** 5,082 labeled complaints
- **Labeling Process:** Manually annotated based on sentiment intensity and urgency
- **Augmentation Techniques:**
  - Paraphrasing (to increase class diversity)
  - Synonym replacement (to improve model generalization)
---

## Model Performance

- **Model:** Fine-tuned BERT
- **Task:** 5-class sentiment classification
- **Accuracy:** 91.93% on test data

---

## Tech Stack

Language         Python, SQL, HTML/CSS \
ML/NLP           TensorFlow, Hugging Face Transformers \
Backend API      Flask \
Database         MySQL \
Frontend         HTML, CSS \
Dev Tools        Google Colab, Visual Studio Code
