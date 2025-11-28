

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')



def create_sample_data(n_samples=1000):
    """Create sample customer data for demonstration"""
    np.random.seed(42)
    
    data = {
        'CustomerID': range(1, n_samples + 1),
        'Age': np.random.randint(18, 70, n_samples),
        'Gender': np.random.choice(['M', 'F'], n_samples),
        'Tenure': np.random.randint(1, 72, n_samples),  # months
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(100, 8000, n_samples),
        'ContractType': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaymentMethod': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSupport': np.random.choice(['Yes', 'No'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No'], n_samples),
        'NumProducts': np.random.randint(1, 5, n_samples),
        'Complaints': np.random.randint(0, 5, n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])  # 27% churn rate
    }
    
    return pd.DataFrame(data)

# Load data
df = create_sample_data(1000)
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())
print("\nChurn Distribution:")
print(df['Churn'].value_counts())


def preprocess_data(df):
    """Preprocess the dataset"""
    df_processed = df.copy()
    
    # Handle missing values (if any)
    df_processed = df_processed.dropna()
    
    # Create additional features
    df_processed['ChargePerMonth'] = df_processed['TotalCharges'] / (df_processed['Tenure'] + 1)
    df_processed['AvgProductsPerYear'] = df_processed['NumProducts'] / (df_processed['Tenure']/12 + 1)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['Gender', 'ContractType', 'PaymentMethod', 'InternetService', 
                       'OnlineSupport', 'TechSupport']
    
    encoders = {}
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        df_processed[col + '_Encoded'] = encoders[col].fit_transform(df_processed[col])
    
    return df_processed, encoders

df_processed, encoders = preprocess_data(df)
print("\nPreprocessed Data Shape:", df_processed.shape)



def perform_clustering(df):
    """Perform customer segmentation using multiple clustering algorithms"""
    
    # Select features for clustering
    clustering_features = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges', 
                          'NumProducts', 'Complaints', 'ChargePerMonth']
    
    X_cluster = df[clustering_features].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # 1. K-Means Clustering
    print("\n" + "="*50)
    print("K-MEANS CLUSTERING")
    print("="*50)
    
    # Find optimal number of clusters using elbow method
    inertias = []
    silhouette_scores = []
    K_range = range(2, 8)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Use optimal k (let's say 4 for this example)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    print(f"Optimal K: {optimal_k}")
    print(f"Silhouette Score: {silhouette_score(X_scaled, kmeans_labels):.3f}")
    print(f"Davies-Bouldin Score: {davies_bouldin_score(X_scaled, kmeans_labels):.3f}")
    
    # 2. DBSCAN Clustering
    print("\n" + "="*50)
    print("DBSCAN CLUSTERING")
    print("="*50)
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    print(f"Number of clusters: {n_clusters_dbscan}")
    print(f"Number of noise points: {n_noise}")
    
    if n_clusters_dbscan > 1:
        mask = dbscan_labels != -1
        print(f"Silhouette Score (excluding noise): {silhouette_score(X_scaled[mask], dbscan_labels[mask]):.3f}")
    
    # 3. Hierarchical Clustering
    print("\n" + "="*50)
    print("HIERARCHICAL CLUSTERING")
    print("="*50)
    
    hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    
    print(f"Silhouette Score: {silhouette_score(X_scaled, hierarchical_labels):.3f}")
    print(f"Davies-Bouldin Score: {davies_bouldin_score(X_scaled, hierarchical_labels):.3f}")
    
    # Add cluster labels to dataframe (using K-Means as primary)
    df['Cluster'] = kmeans_labels
    
    # Analyze clusters
    print("\n" + "="*50)
    print("CLUSTER ANALYSIS (K-Means)")
    print("="*50)
    
    for i in range(optimal_k):
        cluster_data = df[df['Cluster'] == i]
        print(f"\nCluster {i} (n={len(cluster_data)}):")
        print(f"  Avg Age: {cluster_data['Age'].mean():.1f}")
        print(f"  Avg Tenure: {cluster_data['Tenure'].mean():.1f} months")
        print(f"  Avg Monthly Charges: ${cluster_data['MonthlyCharges'].mean():.2f}")
        print(f"  Churn Rate: {cluster_data['Churn'].mean()*100:.1f}%")
    
    return df, scaler, kmeans, X_scaled, inertias, silhouette_scores

df_clustered, scaler_cluster, kmeans_model, X_scaled, inertias, silhouette_scores = perform_clustering(df_processed)



def train_churn_models(df):
    """Train multiple classification models for churn prediction"""
    
    # Select features for prediction
    feature_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges', 
                   'NumProducts', 'Complaints', 'ChargePerMonth', 'Cluster',
                   'Gender_Encoded', 'ContractType_Encoded', 'PaymentMethod_Encoded',
                   'InternetService_Encoded', 'OnlineSupport_Encoded', 'TechSupport_Encoded']
    
    X = df[feature_cols]
    y = df['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                          random_state=42, stratify=y)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    print("\n" + "="*50)
    print("CHURN PREDICTION MODEL COMPARISON")
    print("="*50)
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Select best model (based on F1-score)
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest Model: {best_model_name}")
    
    # Feature Importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
    
    return results, best_model, scaler, X_test, y_test, feature_cols

churn_results, best_model, scaler_churn, X_test, y_test, feature_cols = train_churn_models(df_clustered)



def generate_recommendations(df, model, scaler, feature_cols):
    """Generate personalized retention recommendations"""
    
    # Predict churn probability for all customers
    X_all = df[feature_cols]
    X_all_scaled = scaler.transform(X_all)
    churn_probability = model.predict_proba(X_all_scaled)[:, 1]
    
    df['ChurnProbability'] = churn_probability
    df['ChurnRisk'] = pd.cut(churn_probability, bins=[0, 0.3, 0.7, 1.0], 
                              labels=['Low', 'Medium', 'High'])
    
    print("\n" + "="*50)
    print("RETENTION RECOMMENDATIONS")
    print("="*50)
    
    # High-risk customers
    high_risk = df[df['ChurnRisk'] == 'High'].sort_values('ChurnProbability', ascending=False)
    
    print(f"\nHigh-Risk Customers: {len(high_risk)}")
    print("\nTop 5 High-Risk Customers:")
    print(high_risk[['CustomerID', 'ChurnProbability', 'Cluster', 'Tenure', 
                     'MonthlyCharges', 'Complaints']].head())
    
    # Generate recommendations by cluster and risk
    recommendations = []
    
    for idx, row in high_risk.head(20).iterrows():
        customer_id = row['CustomerID']
        cluster = row['Cluster']
        tenure = row['Tenure']
        complaints = row['Complaints']
        monthly_charges = row['MonthlyCharges']
        
        recs = [f"Customer {customer_id} (Churn Prob: {row['ChurnProbability']:.2%}):"]
        
        # Recommendation based on tenure
        if tenure < 12:
            recs.append("  - Offer onboarding support and welcome discount")
        
        # Recommendation based on complaints
        if complaints > 2:
            recs.append("  - Priority customer service and issue resolution")
        
        # Recommendation based on charges
        if monthly_charges > df['MonthlyCharges'].quantile(0.75):
            recs.append("  - Review pricing, offer loyalty discount")
        
        # Recommendation based on cluster
        if cluster == 0:
            recs.append("  - Upsell premium services")
        elif cluster == 1:
            recs.append("  - Offer budget-friendly packages")
        elif cluster == 2:
            recs.append("  - Focus on service quality improvements")
        else:
            recs.append("  - Personalized engagement campaign")
        
        recommendations.append('\n'.join(recs))
    
    print("\n" + "="*50)
    print("SAMPLE RECOMMENDATIONS:")
    print("="*50)
    for rec in recommendations[:5]:
        print(f"\n{rec}")
    
    return df

df_final = generate_recommendations(df_clustered, best_model, scaler_churn, feature_cols)



print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

# Create visualization code (to be run separately)
visualization_code = """
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Elbow Curve
axes[0, 0].plot(range(2, 8), inertias, 'bo-')
axes[0, 0].set_xlabel('Number of Clusters')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].set_title('Elbow Method For Optimal K')

# 2. Silhouette Scores
axes[0, 1].plot(range(2, 8), silhouette_scores, 'go-')
axes[0, 1].set_xlabel('Number of Clusters')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('Silhouette Score vs Number of Clusters')

# 3. Cluster Distribution
cluster_counts = df_final['Cluster'].value_counts().sort_index()
axes[0, 2].bar(cluster_counts.index, cluster_counts.values, color='skyblue')
axes[0, 2].set_xlabel('Cluster')
axes[0, 2].set_ylabel('Number of Customers')
axes[0, 2].set_title('Customer Distribution Across Clusters')

# 4. Churn Rate by Cluster
churn_by_cluster = df_final.groupby('Cluster')['Churn'].mean() * 100
axes[1, 0].bar(churn_by_cluster.index, churn_by_cluster.values, color='coral')
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Churn Rate (%)')
axes[1, 0].set_title('Churn Rate by Customer Segment')

# 5. Model Comparison
model_names = list(churn_results.keys())
f1_scores = [churn_results[m]['f1'] for m in model_names]
axes[1, 1].barh(model_names, f1_scores, color='lightgreen')
axes[1, 1].set_xlabel('F1-Score')
axes[1, 1].set_title('Model Performance Comparison')

# 6. Churn Risk Distribution
risk_counts = df_final['ChurnRisk'].value_counts()
axes[1, 2].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
               colors=['green', 'yellow', 'red'])
axes[1, 2].set_title('Customer Churn Risk Distribution')

plt.tight_layout()
plt.savefig('churn_analysis_results.png', dpi=300, bbox_inches='tight')
print("Visualizations saved as 'churn_analysis_results.png'")
plt.show()
"""

print("\nVisualization code ready. Run the code above to generate plots.")



# Save final dataset with predictions
df_final.to_csv('customer_churn_predictions.csv', index=False)
print("\nResults saved to 'customer_churn_predictions.csv'")

print("\n" + "="*50)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*50)

