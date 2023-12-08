import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset from a CSV file
# Assume your CSV file has columns 'feature' and 'target'
df = pd.read_csv('D:\python\Book1.csv')

# Extract features and target variable
X = df[['val_loss', 'val_accuracy']]
y = df['Epoch']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the training and testing data along with the regression line
# plt.scatter(X_train, y_train, label='Training Data')
# plt.scatter(X_test, y_test, color='red', label='Testing Data')
# plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression Line')
# plt.title('Machine Learning - Train/Test Plot - val_loss vs val_accuracy')
# plt.xlabel('val_loss & val_accuracy')
# plt.ylabel('Epoch')
# plt.legend()
# plt.show()

# Plot the training and testing data along with the regression line
plt.scatter(X_train['val_loss'], y_train, label='Training Data (val_loss)')
plt.scatter(X_test['val_loss'], y_test, color='red', label='Testing Data (val_loss)')

plt.scatter(X_train['val_accuracy'], y_train, label='Training Data (val_accuracy)')
plt.scatter(X_test['val_accuracy'], y_test, color='green', label='Testing Data (val_accuracy)')

# plt.plot(X_test['val_loss'], y_pred, color='purple', linewidth=3, label='Regression Line')
plt.title('Machine Learning - Train/Test Plot - val_loss vs val_accuracy')
plt.xlabel('val_loss & val_accuracy')
plt.ylabel('epoch')
plt.legend()
plt.show()