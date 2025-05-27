# 🧠 Customer Segmentation with KMeans

This project uses KMeans clustering to segment mall customers based on their **Annual Income** and **Spending Score**. It features an interactive **Streamlit app** that allows users to input custom data and get instant cluster predictions with helpful visualizations.

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `capstone_customer_segmentation.py` | Notebook/script to train and evaluate the clustering model |
| `app.py` | Streamlit app for live cluster prediction |
| `Mall_Customers.csv` | Dataset used for clustering |
| `kmeans_model.pkl` & `scaler.pkl` | Trained model and scaler |
| `requirements.txt` | Required Python libraries |

---

## 🚀 Features

- Predict customer cluster using KMeans
- Visualize customer segmentation by cluster
- Interactive input of income and spending score
- Clean, responsive Streamlit UI

---

## 📊 Cluster Labels

Clusters are interpreted into the following customer types:

- `Cluster 0`: Low Score, Low Income
- `Cluster 1`: Low Score, High Income
- `Cluster 2`: Mid Score, Mid Income
- `Cluster 3`: High Score, Low Income
- `Cluster 4`: High Score, High Income

---

## 🔧 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
