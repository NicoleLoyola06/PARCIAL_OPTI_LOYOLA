import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =============================================================
# PASO 2: Cargar y preprocesar el dataset Iris
# =============================================================
iris = datasets.load_iris()

# Seleccionamos SOLO dos clases: Setosa (0) y Versicolor (1)
# y SOLO dos características: largo del pétalo (col 2) y ancho del pétalo (col 3)
mask = iris.target != 2                  # Excluir Virginica
X = iris.data[mask, 2:4]                # Características: largo y ancho del pétalo
y = iris.target[mask]                   # Etiquetas: 0=Setosa, 1=Versicolor

print("=" * 55)
print("  CLASIFICACIÓN DE FLORES CON SVM - KERNEL LINEAL")
print("=" * 55)
print(f"\n[INFO] Total de muestras: {X.shape[0]}")
print(f"[INFO] Setosa (clase 0):     {np.sum(y == 0)} muestras")
print(f"[INFO] Versicolor (clase 1): {np.sum(y == 1)} muestras")
print(f"[INFO] Características usadas: Largo del pétalo, Ancho del pétalo")

# =============================================================
# PASO 3: Dividir en conjunto de entrenamiento y prueba
# =============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,       # 30% para prueba
    random_state=42      # Semilla para reproducibilidad
)

print(f"\n[SPLIT] Entrenamiento: {len(X_train)} muestras | Prueba: {len(X_test)} muestras")

# =============================================================
# PASO 4: Entrenar el modelo SVM con kernel lineal
# =============================================================
# C=1: parámetro de regularización (penalización por error)
# kernel='linear': frontera de decisión lineal (hiperplano recto)
clf = SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

print(f"\n[MODELO] SVM entrenado con kernel lineal")
print(f"[MODELO] Número de vectores de soporte: {clf.support_vectors_.shape[0]}")
print(f"         - Clase Setosa:     {clf.n_support_[0]} vectores")
print(f"         - Clase Versicolor: {clf.n_support_[1]} vectores")

# =============================================================
# PASO 5: Evaluar el modelo
# =============================================================
y_pred = clf.predict(X_test)

print("\n" + "=" * 55)
print("  MÉTRICAS DE EVALUACIÓN")
print("=" * 55)
print(f"\nPrecisión (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nMatriz de Confusión:")
cm = confusion_matrix(y_test, y_pred)
print(f"             Pred Setosa  Pred Versicolor")
print(f"Real Setosa      {cm[0][0]:^5}        {cm[0][1]:^5}")
print(f"Real Versicolor  {cm[1][0]:^5}        {cm[1][1]:^5}")

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Setosa', 'Versicolor']))

# =============================================================
# PASO 6: Visualizar la frontera de decisión con vectores de soporte
# =============================================================
def plot_svm_decision_boundary(clf, X, y, X_train, X_test, y_test, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('SVM Kernel Lineal — Clasificación de Flores Iris',
                 fontsize=15, fontweight='bold', y=1.02)

    colores = {0: '#2196F3', 1: '#FF5722'}       # Azul=Setosa, Naranja=Versicolor
    nombres = {0: 'Setosa', 1: 'Versicolor'}
    marcadores = {0: 'o', 1: 's'}

    # --- GRÁFICA 1: Frontera de decisión con vectores de soporte ---
    ax = axes[0]
    ax.set_title('Frontera de Decisión y Vectores de Soporte', fontsize=12, pad=10)

    # Crear malla de puntos para graficar la región de decisión
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.15,
                colors=['#2196F3', '#FF5722'])
    ax.contour(xx, yy, Z, colors='black', linewidths=1.5, linestyles='-')

    # Márgenes del hiperplano
    Z_margin = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, Z_margin, levels=[-1, 1],
               colors='gray', linewidths=1, linestyles='--')

    # Graficar todos los puntos del dataset
    for clase in [0, 1]:
        mask = y == clase
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=colores[clase], marker=marcadores[clase],
                   edgecolors='white', linewidths=0.7,
                   s=70, label=nombres[clase], zorder=3)

    # Resaltar los vectores de soporte
    ax.scatter(clf.support_vectors_[:, 0],
               clf.support_vectors_[:, 1],
               s=200, facecolors='none',
               edgecolors='gold', linewidths=2.5,
               label='Vectores de Soporte', zorder=4)

    ax.set_xlabel('Largo del Pétalo (cm)', fontsize=11)
    ax.set_ylabel('Ancho del Pétalo (cm)', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- GRÁFICA 2: Matriz de confusión visual ---
    ax2 = axes[1]
    ax2.set_title('Matriz de Confusión', fontsize=12, pad=10)
    cm = confusion_matrix(y_test, y_pred)

    im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax2)

    clases = ['Setosa', 'Versicolor']
    tick_marks = np.arange(len(clases))
    ax2.set_xticks(tick_marks)
    ax2.set_xticklabels(clases, fontsize=11)
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels(clases, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, str(cm[i, j]),
                     ha='center', va='center', fontsize=20,
                     color='white' if cm[i, j] > thresh else 'black',
                     fontweight='bold')

    ax2.set_ylabel('Clase Real', fontsize=11)
    ax2.set_xlabel('Clase Predicha', fontsize=11)

    acc = accuracy_score(y_test, y_pred)
    ax2.set_title(f'Matriz de Confusión\n(Precisión: {acc*100:.1f}%)', fontsize=12)

    plt.tight_layout()
    plt.savefig('resultado_svm.png', dpi=150, bbox_inches='tight')
    print("\n[OK] Gráfica guardada como 'resultado_svm.png'")
    plt.show()

# Ejecutar visualización
plot_svm_decision_boundary(clf, X, y, X_train, X_test, y_test, y_pred)

print("\n" + "=" * 55)
print("  ANÁLISIS DE RESULTADOS")
print("=" * 55)
print("""
1. Kernel lineal elegido porque Setosa y Versicolor son
   linealmente separables en el espacio pétalo (largo/ancho).

2. Los vectores de soporte (marcados en DORADO) son los
   puntos más cercanos al hiperplano. Definen el margen.

3. Las líneas punteadas representan los márgenes +1 y -1.
   La línea sólida es la frontera de decisión óptima.

4. La matriz de confusión muestra predicciones correctas
   en la diagonal principal.
""")