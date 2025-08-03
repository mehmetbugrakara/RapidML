"""
Produce core visualizations: regression curve, confusion matrix, ROC curve, feature importances.
"""
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_regression_curve(models, df, target, output_dir):
    """Plot true vs predicted values for regression models."""
    for name, model in models.items():
        preds = model.predict(df.drop(columns=[target]))
        true = df[target].values
        plt.figure()
        plt.plot(true, label='True')
        plt.plot(preds, label='Predicted')
        plt.title(f'{name} Regression Curve')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'regression_curve_{name}.png'))
        plt.close()

def plot_confusion_matrix(models, test_df, target, output_dir):
    """Plot confusion matrix for classification models."""
    for name, model in models.items():
        preds = model.predict(test_df.drop(columns=[target]))
        cm = confusion_matrix(test_df[target], preds)
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False, xticks_rotation=45)

        # Hücrelerin içine ham sayıları yaz
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"confusion_matrix_{name}.png"))
        plt.close(fig)

def plot_roc_curve(models, test_df, target, output_dir):
    """Plot ROC curve for classification models."""
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(test_df.drop(columns=[target]))[:,1]
            fpr, tpr, _ = roc_curve(test_df[target], probs)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0,1], [0,1], linestyle='--')
            plt.title(f'{name} ROC Curve')
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(output_dir, f'roc_curve_{name}.png'))
            plt.close()

def plot_feature_importance(models, output_dir):
    """Save feature importance plots for each model."""
    for name, model in models.items():
        try:
            importances = model.feature_importances_
            indices = importances.argsort()[::-1]
            plt.figure()
            plt.title(f'{name} Feature Importance')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), indices, rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'feature_importance_{name}.png'))
            plt.close()
        except AttributeError:
            continue
