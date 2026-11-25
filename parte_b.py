"""
PARTE B: Predicción de deserción escolar (Deserción / Matriculado / Graduado)
Dataset: Predict Students' Dropout and Academic Success (UCI, id=697)
Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021).
UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
Licencia: CC BY 4.0
"""
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # solo para importancia de variables
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Traducción de nombres de variables para las gráficas
TRAD = {
    "Curricular units 2nd sem (approved)": "Materias aprobadas 2.º sem",
    "Curricular units 2nd sem (grade)": "Promedio 2.º sem",
    "Curricular units 1st sem (approved)": "Materias aprobadas 1.er sem",
    "Curricular units 1st sem (grade)": "Promedio 1.er sem",
    "Curricular units 2nd sem (evaluations)": "Evaluaciones 2.º sem",
    "Curricular units 1st sem (evaluations)": "Evaluaciones 1.er sem",
    "Curricular units 2nd sem (enrolled)": "Materias inscritas 2.º sem",
    "Curricular units 1st sem (enrolled)": "Materias inscritas 1.er sem",
    "Admission grade": "Calificación de admisión",
    "Previous qualification (grade)": "Calificación previa",
    "Tuition fees up to date": "Colegiatura al corriente",
    "Age at enrollment": "Edad al inscribirse",
    "Scholarship holder": "Becario",
    "Debtor": "Deudor",
    "Course": "Carrera",
}


# 1. CARGA DEL DATASET
def cargar_datos():
    if os.path.exists("data.csv"):
        df = pd.read_csv("data.csv", sep=None, engine="python")
        print("Dataset cargado desde data.csv")
        return df
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=697)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    print("Dataset descargado con ucimlrepo")
    return df


df = cargar_datos()
df.to_csv("dataset_desercion.csv", index=False)  # guarda el dataset original para la entrega
df["Target"] = df["Target"].map({"Dropout": "Deserción", "Enrolled": "Matriculado",
                                 "Graduate": "Graduado"})
resultados = {}

# 2. ANÁLISIS EXPLORATORIO
print("Dimensiones:", df.shape)
print("Valores faltantes totales:", int(df.isna().sum().sum()))
print(df["Target"].value_counts())
resultados["dimensiones"] = list(df.shape)
resultados["nulos"] = int(df.isna().sum().sum())
resultados["distribucion_clases"] = df["Target"].value_counts().to_dict()

df["Target"].value_counts().plot(kind="bar", color=["#2f855a", "#c53030", "#2b6cb0"])
plt.title("Distribución de la variable objetivo")
plt.xlabel("Clase")
plt.ylabel("Estudiantes")
plt.tight_layout()
plt.savefig("eda_clases.png", dpi=150)
plt.show()

# 3. LIMPIEZA Y PREPARACIÓN
df = df.drop_duplicates().dropna()
le = LabelEncoder()
y = le.fit_transform(df["Target"])
clases = list(le.classes_)
X = df.drop(columns=["Target"])
X = pd.get_dummies(X, drop_first=True).astype(float)

# División 70 / 15 / 15 (estratificada)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)
print("Entrenamiento/Validación/Prueba:", len(X_tr), len(X_va), len(X_te))
resultados["division"] = {"entrenamiento": len(X_tr), "validacion": len(X_va), "prueba": len(X_te)}

# Normalización (ajustada SOLO con entrenamiento)
sc = StandardScaler().fit(X_tr)
X_tr_s, X_va_s, X_te_s = sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te)


def metricas(y_real, y_pred):
    return {
        "exactitud": round(accuracy_score(y_real, y_pred), 4),
        "precision": round(precision_score(y_real, y_pred, average="macro", zero_division=0), 4),
        "recall": round(recall_score(y_real, y_pred, average="macro", zero_division=0), 4),
        "f1": round(f1_score(y_real, y_pred, average="macro", zero_division=0), 4),
    }


# 4. MODELOS CLÁSICOS
resultados["modelos"] = {}
modelos = {
    "Regresión logística": LogisticRegression(max_iter=2000, random_state=SEED),
    "Árbol de decisión": DecisionTreeClassifier(max_depth=6, random_state=SEED),
}
for nombre, m in modelos.items():
    m.fit(X_tr_s, y_tr)
    resultados["modelos"][nombre] = {
        "validacion": metricas(y_va, m.predict(X_va_s)),
        "prueba": metricas(y_te, m.predict(X_te_s)),
    }
    print(nombre, resultados["modelos"][nombre]["prueba"])
    ConfusionMatrixDisplay.from_predictions(y_te, m.predict(X_te_s), display_labels=clases,
                                            cmap="Blues")
    plt.title(f"Matriz de confusión - {nombre}")
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.tight_layout()
    plt.savefig(f"matriz_{nombre.split()[0].lower()}.png", dpi=150)
    plt.show()

# Importancia de variables (para la discusión de interpretabilidad)
rf = RandomForestClassifier(n_estimators=200, random_state=SEED).fit(X_tr, y_tr)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(8)
imp.index = [TRAD.get(n, n) for n in imp.index]
resultados["variables_importantes"] = {k: round(float(v), 4) for k, v in imp.items()}
imp[::-1].plot(kind="barh", color="#2b6cb0")
plt.title("Variables más importantes (Random Forest)")
plt.xlabel("Importancia")
plt.tight_layout()
plt.savefig("importancia_variables.png", dpi=150)
plt.show()

# 5. RED NEURONAL (Keras)
import tensorflow as tf
from tensorflow import keras

tf.keras.utils.set_random_seed(SEED)


def construir_red(n_entrada, n_clases, lr=0.001):
    modelo = keras.Sequential([
        keras.layers.Input(shape=(n_entrada,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),  # técnica contra sobreajuste
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(n_clases, activation="softmax"),
    ])
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


# Ajuste de hiperparámetros: 2 tasas de aprendizaje x 2 tamaños de lote
resultados["red_neuronal"] = {"pruebas": {}}
mejor, mejor_f1, mejor_hist, mejor_cfg = None, -1, None, None
for lr in (0.001, 0.01):
    for lote in (32, 64):
        tf.keras.utils.set_random_seed(SEED)
        red = construir_red(X_tr_s.shape[1], len(clases), lr)
        parada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                               restore_best_weights=True)  # parada temprana
        hist = red.fit(X_tr_s, y_tr, validation_data=(X_va_s, y_va), epochs=100,
                       batch_size=lote, callbacks=[parada], verbose=0)
        f1_val = f1_score(y_va, red.predict(X_va_s, verbose=0).argmax(1), average="macro")
        clave = f"lr={lr}, lote={lote}"
        resultados["red_neuronal"]["pruebas"][clave] = {
            "f1_validacion":