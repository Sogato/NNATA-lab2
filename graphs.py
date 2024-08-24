import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Arial'
FILE_PATH = 'img/graphs_img/ROC_curve.png'


def plot_roc_curve(roc_data):
    """
    Строит и визуализирует ROC-кривые для каждой модели на основе рассчитанных данных.
    Опционально сохраняет график в файл.

    :param roc_data: Словарь с именами моделей и соответствующими ROC-кривыми и значениями AUC.
    """
    plt.figure(figsize=(10, 8))

    for model_name, data in roc_data.items():
        plt.plot(data['fpr'], data['tpr'], lw=2, alpha=0.8,
                 label=f'{model_name} (AUC = {data["roc_auc"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=3, alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Коэффициент ложных срабатываний', fontsize=14, fontweight='bold')
    plt.ylabel('Показатель истинных положительных результатов', fontsize=14, fontweight='bold')
    plt.title('ROC-кривые для моделей', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='-', alpha=0.7, linewidth=1.5)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(FILE_PATH, bbox_inches='tight')
    plt.close()


def plot_feature_importances(importances, feature_names, title, file_path):
    """
    Визуализирует значимость признаков для модели и сохраняет график в файл.

    :param importances: Значимости признаков.
    :param feature_names: Имена признаков.
    :param title: Заголовок графика.
    :param file_path: Путь для сохранения графика.
    """
    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=16, fontweight='bold')
    colors = plt.cm.viridis(np.linspace(0, 1, len(importances)))
    plt.bar(range(len(importances)), importances, align="center", color=colors, alpha=0.8)
    plt.xticks(range(len(importances)), feature_names, rotation=90, fontsize=12)
    plt.xlim([-1, len(importances)])
    plt.ylabel('Значимость', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='-', alpha=0.7, axis='y', linewidth=1.5)
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
