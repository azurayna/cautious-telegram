import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Load dataset
data = load_iris()
X = data.data
y = data.target
labels = data.target_names

# Sidebar UI
st.sidebar.header("Settings")
classifier_name = st.sidebar.selectbox("Choose classifier", ["SVM", "k-NN", "Decision Tree", "Logistic Regression"])
reduction_method = st.sidebar.selectbox("Dimensionality Reduction", ["PCA", "t-SNE"])

# Hyperparameters
if classifier_name == "SVM":
    C = st.sidebar.slider("SVM C", 0.01, 10.0, 1.0)
    clf = SVC(kernel='linear', C=C)
elif classifier_name == "k-NN":
    k = st.sidebar.slider("k in k-NN", 1, 15, 5)
    clf = KNeighborsClassifier(n_neighbors=k)
elif classifier_name == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
    clf = DecisionTreeClassifier(max_depth=max_depth)
elif classifier_name == "Logistic Regression":
    C = st.sidebar.slider("LogReg C", 0.01, 10.0, 1.0)
    clf = LogisticRegression(C=C, max_iter=200)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Dimensionality Reduction
if reduction_method == "PCA":
    reducer = PCA(n_components=2)
else:
    reducer = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = reducer.fit_transform(X)

# Manual input
st.sidebar.subheader("Manual Input")
user_input = [st.sidebar.slider(f, float(X[:,i].min()), float(X[:,i].max()), float(X[:,i].mean())) for i, f in enumerate(data.feature_names)]
user_pred = clf.predict([user_input])[0]
st.sidebar.write("**Prediction:**", labels[user_pred])

# Create decision boundary mesh
if reduction_method == "PCA":
    def plot_decision_boundaries(X_proj, y, model):
        x_min, x_max = X_proj[:, 0].min() - 1, X_proj[:, 0].max() + 1
        y_min, y_max = X_proj[:, 1].min() - 1, X_proj[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Project back to original space to predict
        X_inv = reducer.inverse_transform(grid)
        Z = model.predict(X_inv)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
        scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, cmap='Set1', edgecolor='k')

        # Plot user input
        user_proj = reducer.transform([user_input])
        plt.scatter(user_proj[0, 0], user_proj[0, 1], color='black', marker='*', s=200, label='Your Input')
        plt.legend()

        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f'{reduction_method} Projection with Decision Boundary')
        return scatter
else:
    def plot_decision_boundaries(X_proj, y, model):
        scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, cmap='Set1', edgecolor='k')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f'{reduction_method} Projection (no boundary)')
        return scatter

# Streamlit UI
st.title("Feature Vector Classification & Visualization")
st.write(f"**Dataset:** Iris  ")
st.write(f"**Classifier:** {classifier_name}  ")
st.write(f"**Dimensionality Reduction:** {reduction_method}  ")
st.write(f"**Model Accuracy on Test Set:** {acc:.2f}")

fig, ax = plt.subplots()
plot_decision_boundaries(X_2d, y, clf)
st.pyplot(fig)

# Classification report
st.markdown("---")
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Confusion matrix
st.markdown("---")
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax2)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig2)

# Feature importance
st.markdown("---")
st.subheader("Feature Importances")
if classifier_name == "Decision Tree":
    st.bar_chart(pd.Series(clf.feature_importances_, index=data.feature_names))
elif classifier_name == "Logistic Regression":
    st.bar_chart(pd.Series(clf.coef_[0], index=data.feature_names))
elif classifier_name == "SVM":
    st.bar_chart(pd.Series(clf.coef_[0], index=data.feature_names))
else:
    st.write("Feature importances not available for this model.")

# PCA components
if reduction_method == "PCA":
    st.markdown("---")
    st.subheader("PCA Component Contributions")
    st.dataframe(pd.DataFrame(reducer.components_, columns=data.feature_names, index=['PC1', 'PC2']))

