import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- STEP 1: Data Generation ---
np.random.seed(42)
n_rows = 1000

# Generating sessions and correlated purchase amounts
total_sessions = np.random.randint(1, 50, n_rows)
purchase_amount = (total_sessions * 5) + np.random.normal(50, 20, n_rows)

data = {
    'customer_id': range(1, n_rows + 1),
    'source': np.random.choice(['SEO', 'Google_Ads', 'Social_Media', 'Direct'], n_rows),
    'purchase_amount': purchase_amount.clip(20, 500),
    'days_since_last_purchase': np.random.randint(1, 365, n_rows),
    'total_sessions': total_sessions
}

df = pd.DataFrame(data)

# --- STEP 2: Marketing Performance Analysis ---
# Grouping by source to calculate key metrics
performance_metrics = df.groupby('source').agg({
    'purchase_amount': ['mean', 'sum', 'count']
}).round(2)

performance_metrics.columns = ['avg_purchase', 'total_revenue', 'transaction_count']
performance_metrics = performance_metrics.sort_values(by='total_revenue', ascending=False)

# --- STEP 3: Customer Segmentation ---
# Function to define customer groups
def segment_customer(row):
    if row['purchase_amount'] > 200 and row['days_since_last_purchase'] < 60:
        return 'VIP Customer'
    elif row['purchase_amount'] > 200 and row['days_since_last_purchase'] >= 60:
        return 'At Risk - High Value'
    elif row['purchase_amount'] <= 200 and row['days_since_last_purchase'] < 60:
        return 'Active - Low Spender'
    else:
        return 'Churned'

# Applying segmentation logic
df['customer_segment'] = df.apply(segment_customer, axis=1)

# --- STEP 4: Data Visualization ---
# Setting visual style for the plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Plotting the segment distribution
sns.countplot(data=df, x='customer_segment', palette='viridis', 
              order=df['customer_segment'].value_counts().index)

plt.title('Distribution of Customer Segments', fontsize=15)
plt.xlabel('Customer Segment')
plt.ylabel('Count')
plt.show()

# --- STEP 5: Machine Learning (Prediction) ---
# Preparing features (X) and target (y)
X = df[['total_sessions']] 
y = df['purchase_amount']

# Splitting data into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions and calculating accuracy
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# --- STEP 6: Exporting Results ---
# Creating a summary report for the AI model
model_report = pd.DataFrame({
    'Metric': ['MSE', 'R-squared', 'Coefficient', 'Intercept'],
    'Value': [mse, r2, model.coef_[0], model.intercept_]
})

# Saving all data to a multi-sheet Excel file
with pd.ExcelWriter('Marketing_Analysis_Report.xlsx') as writer:
    df.to_excel(writer, sheet_name='Customer_Data', index=False)
    performance_metrics.to_excel(writer, sheet_name='Source_Performance')
    model_report.to_excel(writer, sheet_name='AI_Model_Insights', index=False)

print(f"Success! Report generated. Model Accuracy: {r2*100:.2f}%")