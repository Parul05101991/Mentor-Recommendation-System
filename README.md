# Mentor Recommendation System

A **Streamlit web app** that recommends mentors to mentees based on **Interest Field, Location, and Burnout Risk Score** using **K-means Clustering and cosine similarity**. The app returns the top 5 mentors for any mentee input and provides an optional heatmap visualization.

---

## Features

- Select mentee **Interest Field** and **Location** from dropdowns  
- Input **Burnout Risk Score** using a slider  
- Get **Top 5 recommended mentors** with similarity scores  
- Optional **heatmap** showing mentor-mentee similarity  

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Parul05101991/Mentor-Recommendation-System.git
cd Mentor-Recommendation-System
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run/UI.py
```

## Usage

- Select Interest Field and Location from dropdowns
- Set the Burnout Risk Score using the slider.
- Click Find Mentors to see the top 5 mentors with similarity scores.
- Optionally, check Show heatmap to visualize similarity scores.


