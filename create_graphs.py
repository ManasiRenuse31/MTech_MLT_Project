import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("Loading data...")
df = pd.read_csv('customer_churn_predictions.csv')
print(f"Data loaded: {len(df)} customers")

print("\nCreating visualizations...")

# Set style
sns.set_style('whitegrid')

# Graph 1: Main Analysis (6 subplots)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Customer Churn Analysis Results', fontsize=20, fontweight='bold')

# 1. Cluster Distribution
cluster_counts = df['Cluster'].value_counts().sort_index()
axes[0, 0].bar(cluster_counts.index, cluster_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
axes[0, 0].set_xlabel('Cluster', fontsize=11)
axes[0, 0].set_ylabel('Number of Customers', fontsize=11)
axes[0, 0].set_title('Customer Distribution by Cluster', fontsize=13, fontweight='bold')
for i, v in enumerate(cluster_counts.values):
    axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold', fontsize=10)

# 2. Churn Rate by Cluster
churn_by_cluster = df.groupby('Cluster')['Churn'].mean() * 100
axes[0, 1].bar(churn_by_cluster.index, churn_by_cluster.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
axes[0, 1].set_xlabel('Cluster', fontsize=11)
axes[0, 1].set_ylabel('Churn Rate (%)', fontsize=11)
axes[0, 1].set_title('Churn Rate by Customer Segment', fontsize=13, fontweight='bold')
for i, v in enumerate(churn_by_cluster.values):
    axes[0, 1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)

# 3. Risk Distribution
risk_counts = df['ChurnRisk'].value_counts()
colors = ['#2ECC71', '#F39C12', '#E74C3C']
axes[0, 2].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=colors, startangle=90, textprops={'fontsize': 11})
axes[0, 2].set_title('Customer Risk Distribution', fontsize=13, fontweight='bold')

# 4. Tenure vs Monthly Charges
scatter = axes[1, 0].scatter(df['Tenure'], df['MonthlyCharges'], c=df['Churn'], cmap='RdYlGn_r', alpha=0.5, s=20)
axes[1, 0].set_xlabel('Tenure (months)', fontsize=11)
axes[1, 0].set_ylabel('Monthly Charges ($)', fontsize=11)
axes[1, 0].set_title('Tenure vs Charges (Red=Churned)', fontsize=13, fontweight='bold')
cbar = plt.colorbar(scatter, ax=axes[1, 0])
cbar.set_label('Churn', fontsize=10)

# 5. Complaints Impact
complaint_churn = df.groupby('Complaints')['Churn'].mean() * 100
axes[1, 1].plot(complaint_churn.index, complaint_churn.values, marker='o', linewidth=2, markersize=8, color='#E74C3C')
axes[1, 1].set_xlabel('Number of Complaints', fontsize=11)
axes[1, 1].set_ylabel('Churn Rate (%)', fontsize=11)
axes[1, 1].set_title('Impact of Complaints on Churn', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# 6. Churn Probability Distribution
axes[1, 2].hist(df['ChurnProbability'], bins=30, color='#9B59B6', edgecolor='black', alpha=0.7)
axes[1, 2].set_xlabel('Churn Probability', fontsize=11)
axes[1, 2].set_ylabel('Count', fontsize=11)
axes[1, 2].set_title('Distribution of Churn Probabilities', fontsize=13, fontweight='bold')
axes[1, 2].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='50% threshold')
axes[1, 2].legend(fontsize=9)

plt.tight_layout()
plt.savefig('../results/main_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/main_analysis.png")
plt.close()

# Graph 2: Model Comparison
plt.figure(figsize=(10, 6))
models = ['Logistic\nRegression', 'Decision\nTree', 'Random\nForest', 'Gradient\nBoosting']
accuracy = [0.7550, 0.5950, 0.7650, 0.7450]
f1_score = [0.0000, 0.2430, 0.1132, 0.1356]

x = np.arange(len(models))
width = 0.35

bars1 = plt.bar(x - width/2, accuracy, width, label='Accuracy', color='#3498DB')
bars2 = plt.bar(x + width/2, f1_score, width, label='F1-Score', color='#E74C3C')

plt.xlabel('Models', fontweight='bold', fontsize=12)
plt.ylabel('Score', fontweight='bold', fontsize=12)
plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
plt.xticks(x, models, fontsize=10)
plt.legend(fontsize=11)
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../results/model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/model_comparison.png")
plt.close()

# Graph 3: Feature Importance
plt.figure(figsize=(10, 7))
features = ['MonthlyCharges', 'ChargePerMonth', 'Tenure', 'Age', 'TotalCharges', 
            'Complaints', 'NumProducts', 'PaymentMethod', 'OnlineSupport', 'Gender']
importance = [0.2118, 0.1535, 0.1405, 0.1239, 0.1155, 0.0681, 0.0543, 0.0370, 0.0330, 0.0287]

colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
bars = plt.barh(features, importance, color=colors_feat)

plt.xlabel('Importance Score', fontweight='bold', fontsize=12)
plt.title('Top 10 Important Features for Churn Prediction', fontsize=16, fontweight='bold')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, importance)):
    plt.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('../results/feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/feature_importance.png")
plt.close()

print("\n" + "="*60)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*60)
print("\nGenerated 3 graphs in 'results/' folder:")
print("1. main_analysis.png - 6 comprehensive charts")
print("2. model_comparison.png - Algorithm performance")
print("3. feature_importance.png - Important features")
print("\n✅ Next: Open these images and use in your report!")