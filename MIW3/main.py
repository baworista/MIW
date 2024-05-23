from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Tworzenie zbioru danych
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

# Podział na zbiór uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Testowanie drzewa decyzyjnego dla różnych kryteriów podziału i głębokości
dt_params = {'criterion': ['entropy', 'gini'],
             'max_depth': [3, 5, 7]}

dt_clf = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, n_jobs=-1)
dt_clf.fit(X_train, y_train)
print(f"Best Decision Tree Parameters: {dt_clf.best_params_}")
dt_accuracy = accuracy_score(y_test, dt_clf.predict(X_test))
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")


# Trenowanie lasu losowego z użyciem GridSearchCV
rf_params = {'n_estimators': [50, 100, 200]}

rf_clf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1)
rf_clf.fit(X_train, y_train)
print(f"Best Random Forest Parameters: {rf_clf.best_params_}")
rf_accuracy = accuracy_score(y_test, rf_clf.predict(X_test))
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")


# Trenowanie regresji logistycznej z użyciem GridSearchCV
lr_params = {'C': [0.1, 1, 10]}

lr_clf = GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=5, n_jobs=-1)
lr_clf.fit(X_train, y_train)
print(f"Best Logistic Regression Parameters: {lr_clf.best_params_}")
lr_accuracy = accuracy_score(y_test, lr_clf.predict(X_test))
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")


# Trenowanie SVM z użyciem GridSearchCV
svm_params = {'C': [0.1, 1, 10],
              'kernel': ['linear', 'rbf']}

svm_clf = GridSearchCV(SVC(random_state=42, probability=True), svm_params, cv=5, n_jobs=-1)
svm_clf.fit(X_train, y_train)
print(f"Best SVM Parameters: {svm_clf.best_params_}")
svm_accuracy = accuracy_score(y_test, svm_clf.predict(X_test))
print(f"SVM Accuracy: {svm_accuracy:.4f}")


# Tworzenie Voting Classifier z użyciem najlepszych klasyfikatorów
voting_clf = VotingClassifier(estimators=[
    ('lr', lr_clf.best_estimator_),
    ('svm', svm_clf.best_estimator_),
    ('rf', rf_clf.best_estimator_)
], voting='soft')

voting_clf.fit(X_train, y_train)
voting_accuracy = accuracy_score(y_test, voting_clf.predict(X_test))
print(f"Voting Classifier Accuracy: {voting_accuracy:.4f}")

# Plotting ROC Curves
classifiers = [dt_clf, rf_clf, lr_clf, svm_clf, voting_clf]
classifier_names = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'SVM', 'Voting Classifier']

plt.figure(figsize=(10, 8))
for clf, name in zip(classifiers, classifier_names):
    y_pred = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_Curves.png')
plt.show()
