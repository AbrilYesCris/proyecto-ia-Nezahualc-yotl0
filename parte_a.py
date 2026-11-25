"""
PROYECTO FINAL - Fundamentos de Inteligencia Artificial (UTNeza)
PARTE A: Problema del agente viajero (TSP) con 10 ciudades del Estado de México
  - Búsqueda ciega: BFS, DFS y Costo Uniforme
  - Búsqueda heurística: Voraz y A*
  - Algoritmo genético (3 configuraciones)
"""
import heapq
import json
import math
import random
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 1. DATOS: 10 ciudades (coordenadas aproximadas: latitud, longitud)
CIUDADES = {
    "Nezahualcóyotl": (19.4003, -99.0146),
    "Toluca": (19.2826, -99.6557),
    "Texcoco": (19.5172, -98.8833),
    "Chalco": (19.2635, -98.8975),
    "Ecatepec": (19.6018, -99.0507),
    "Naucalpan": (19.4785, -99.2396),
    "Tlalnepantla": (19.5400, -99.1950),
    "Cuautitlán Izcalli": (19.6475, -99.2467),
    "Chimalhuacán": (19.4217, -98.9542),
    "Teotihuacán": (19.6925, -98.8438),
}
NOMBRES = list(CIUDADES.keys())
N = len(NOMBRES)


def haversine(a, b):
    """Distancia en km entre dos puntos (lat, lon)."""
    r = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(d))


DIST = [[haversine(CIUDADES[a], CIUDADES[b]) for b in NOMBRES] for a in NOMBRES]

# 2. FORMULACIÓN COMO PROBLEMA DE BÚSQUEDA
#    Estado: ruta parcial. Inicial: [0]. Acción: ir a una ciudad no visitada.
#    Meta: visitar las 10 ciudades y regresar a la 0. Costo: suma de km.
INICIO = 0


def es_meta(ruta):
    return len(ruta) == N + 1 and ruta[-1] == INICIO


def sucesores(ruta):
    if len(ruta) == N:  # ya visitó todas: solo falta regresar
        return [(ruta + [INICIO], DIST[ruta[-1]][INICIO])]
    visitadas = set(ruta)
    return [(ruta + [c], DIST[ruta[-1]][c]) for c in range(N) if c not in visitadas]


def costo_ruta(ruta):
    return sum(DIST[ruta[i]][ruta[i + 1]] for i in range(len(ruta) - 1))


# 3. BÚSQUEDA CIEGA
def bfs():
    t0 = time.perf_counter()
    cola = deque([[INICIO]])
    expandidos = 0
    while cola:
        ruta = cola.popleft()
        if es_meta(ruta):
            return ruta, expandidos, time.perf_counter() - t0
        expandidos += 1
        for nueva, _ in sucesores(ruta):
            cola.append(nueva)


def dfs():
    t0 = time.perf_counter()
    pila = [[INICIO]]
    expandidos = 0
    while pila:
        ruta = pila.pop()
        if es_meta(ruta):
            return ruta, expandidos, time.perf_counter() - t0
        expandidos += 1
        for nueva, _ in reversed(sucesores(ruta)):
            pila.append(nueva)


def costo_uniforme():
    t0 = time.perf_counter()
    cont = 0
    heap = [(0.0, cont, [INICIO])]
    expandidos = 0
    while heap:
        g, _, ruta = heapq.heappop(heap)
        if es_meta(ruta):
            return ruta, expandidos, time.perf_counter() - t0
        expandidos += 1
        for nueva, c in sucesores(ruta):
            cont += 1
            heapq.heappush(heap, (g + c, cont, nueva))


# 4. BÚSQUEDA HEURÍSTICA
#    h(n) = costo del Árbol de Expansión Mínima (MST) de las ciudades
#    pendientes + la actual + el inicio. Es ADMISIBLE: nunca sobrestima.
def mst_costo(nodos):
    nodos = list(nodos)
    if len(nodos) <= 1:
        return 0.0
    en_arbol = {nodos[0]}
    resto = set(nodos[1:])
    total = 0.0
    while resto:
        mejor = min(((DIST[a][b], b) for a in en_arbol for b in resto))
        total += mejor[0]
        en_arbol.add(mejor[1])
        resto.remove(mejor[1])
    return total


def h_mst(ruta):
    if len(ruta) == N + 1:
        return 0.0
    pendientes = set(range(N)) - set(ruta)
    return mst_costo(pendientes | {ruta[-1], INICIO})


def h_vecino(ruta):
    """Distancia al vecino pendiente más cercano (admisible pero más débil)."""
    if len(ruta) == N + 1:
        return 0.0
    if len(ruta) == N:
        return DIST[ruta[-1]][INICIO]
    pendientes = set(range(N)) - set(ruta)
    return min(DIST[ruta[-1]][c] for c in pendientes)


def voraz():
    """Búsqueda voraz: ordena solo por h(n)."""
    t0 = time.perf_counter()
    cont = 0
    heap = [(h_vecino([INICIO]), cont, [INICIO])]
    expandidos = 0
    while heap:
        _, _, ruta = heapq.heappop(heap)
        if es_meta(ruta):
            return ruta, expandidos, time.perf_counter() - t0
        expandidos += 1
        for nueva, _c in sucesores(ruta):
            cont += 1
            heapq.heappush(heap, (h_vecino(nueva), cont, nueva))


def a_estrella(h=h_mst):
    """A*: ordena por f(n) = g(n) + h(n)."""
    t0 = time.perf_counter()
    cont = 0
    heap = [(h([INICIO]), cont, 0.0, [INICIO])]
    expandidos = 0
    while heap:
        _, _, g, ruta = heapq.heappop(heap)
        if es_meta(ruta):
            return ruta, expandidos, time.perf_counter() - t0
        expandidos += 1
        for nueva, c in sucesores(ruta):
            cont += 1
            g2 = g + c
            heapq.heappush(heap, (g2 + h(nueva), cont, g2, nueva))


# 5. ALGORITMO GENÉTICO
#    Cromosoma: permutación de las ciudades 1..N-1 (la 0 es fija).
#    Aptitud: 1/longitud. Selección: torneo. Cruza: OX. Mutación: intercambio.
#    Elitismo: se conserva el mejor individuo.
def longitud(cromo):
    return costo_ruta([INICIO] + cromo + [INICIO])


def torneo(poblacion, fits, k=3):
    elegidos = random.sample(range(len(poblacion)), k)
    return poblacion[max(elegidos, key=lambda i: fits[i])]


def cruza_ox(p1, p2):
    a, b = sorted(random.sample(range(len(p1)), 2))
    hijo = [None] * len(p1)
    hijo[a:b] = p1[a:b]
    resto = [g for g in p2 if g not in hijo[a:b]]
    j = 0
    for i in range(len(hijo)):
        if hijo[i] is None:
            hijo[i] = resto[j]
            j += 1
    return hijo


def mutar(cromo, tasa):
    cromo = cromo[:]
    if random.random() < tasa:
        i, j = random.sample(range(len(cromo)), 2)
        cromo[i], cromo[j] = cromo[j], cromo[i]
    return cromo


def algoritmo_genetico(pob=50, tasa_cruza=0.8, tasa_mut=0.1, generaciones=100, semilla=SEED):
    random.seed(semilla)
    t0 = time.perf_counter()
    base = list(range(1, N))
    poblacion = [random.sample(base, len(base)) for _ in range(pob)]
    hist_mejor, hist_prom, hist_peor = [], [], []
    mejor_global = None
    for _ in range(generaciones):
        longs = [longitud(c) for c in poblacion]
        fits = [1.0 / l for l in longs]
        hist_mejor.append(min(longs))
        hist_prom.append(sum(longs) / len(longs))
        hist_peor.append(max(longs))
        i_best = int(np.argmin(longs))
        if mejor_global is None or longs[i_best] < longitud(mejor_global):
            mejor_global = poblacion[i_best][:]
        nueva = [poblacion[i_best][:]]  # elitismo
        while len(nueva) < pob:
            p1, p2 = torneo(poblacion, fits), torneo(poblacion, fits)
            hijo = cruza_ox(p1, p2) if random.random() < tasa_cruza else p1[:]
            nueva.append(mutar(hijo, tasa_mut))
        poblacion = nueva
    return ([INICIO] + mejor_global + [INICIO], hist_mejor, hist_prom, hist_peor,
            time.perf_counter() - t0)


# 6. EXPERIMENTOS
def nombres(ruta):
    return " -> ".join(NOMBRES[i] for i in ruta)


def main():
    resultados = {"busqueda": {}, "genetico": {}}

    print("== BÚSQUEDA CIEGA Y HEURÍSTICA ==")
    algos = [("BFS", bfs), ("DFS", dfs), ("Costo uniforme", costo_uniforme),
             ("Voraz", voraz), ("A*", a_estrella)]
    for nombre, f in algos:
        ruta, exp, seg = f()
        resultados["busqueda"][nombre] = {
            "ruta": nombres(ruta), "costo_km": round(costo_ruta(ruta), 2),
            "longitud_solucion": len(ruta) - 1, "nodos_expandidos": exp,
            "tiempo_s": round(seg, 4)}
        r = resultados["busqueda"][nombre]
        print(f"{nombre:15s} costo={r['costo_km']:8.2f} km  nodos={exp:8d}  t={seg:.3f}s")

    # Heurística alternativa (Paso 4.4): también admisible pero menos informada
    ruta, exp, seg = a_estrella(h_vecino)
    resultados["busqueda"]["A* con h_vecino"] = {
        "ruta": nombres(ruta), "costo_km": round(costo_ruta(ruta), 2),
        "longitud_solucion": len(ruta) - 1, "nodos_expandidos": exp,
        "tiempo_s": round(seg, 4)}
    print(f"{'A* h_vecino':15s} costo={costo_ruta(ruta):8.2f} km  nodos={exp:8d}  t={seg:.3f}s")

    # Gráfica comparativa
    nom = [k for k in resultados["busqueda"] if k != "A* con h_vecino"]
    nodos = [resultados["busqueda"][k]["nodos_expandidos"] for k in nom]
    costos = [resultados["busqueda"][k]["costo_km"] for k in nom]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(nom, nodos, color="#2b6cb0")
    ax[0].set_yscale("log")
    ax[0].set_title("Nodos expandidos (escala log)")
    ax[0].tick_params(axis="x", rotation=30)
    ax[1].bar(nom, costos, color="#2f855a")
    ax[1].set_title("Costo de la solución (km)")
    ax[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig("grafica_busqueda.png", dpi=150)
    plt.show()

    print("\n== ALGORITMO GENÉTICO ==")
    configs = {
        "Config 1 (pob=30, cruza=0.7, mut=0.05)": dict(pob=30, tasa_cruza=0.7, tasa_mut=0.05),
        "Config 2 (pob=50, cruza=0.8, mut=0.10)": dict(pob=50, tasa_cruza=0.8, tasa_mut=0.10),
        "Config 3 (pob=100, cruza=0.9, mut=0.20)": dict(pob=100, tasa_cruza=0.9, tasa_mut=0.20),
    }
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for k, (nombre, cfg) in enumerate(configs.items()):
        ruta, mejor, prom, peor, seg = algoritmo_genetico(generaciones=100, **cfg)
        resultados["genetico"][nombre] = {
            "ruta": nombres(ruta), "costo_km": round(costo_ruta(ruta), 2),
            "tiempo_s": round(seg, 4), "generaciones": 100}
        print(f"{nombre}: costo={costo_ruta(ruta):.2f} km  t={seg:.3f}s")
        ax[k].plot(mejor, label="Mejor")
        ax[k].plot(prom, label="Promedio")
        ax[k].plot(peor, label="Peor")
        ax[k].set_title(nombre, fontsize=9)
        ax[k].set_xlabel("Generación")
        if k == 0:
            ax[k].set_ylabel("Longitud del recorrido (km)")
            ax[k].legend()
    plt.tight_layout()
    plt.savefig("grafica_genetico.png", dpi=150)
    plt.show()

    # Mapa de la mejor ruta (A*)
    ruta_opt, _, _ = a_estrella()
    lat = [CIUDADES[NOMBRES[i]][0] for i in ruta_opt]
    lon = [CIUDADES[NOMBRES[i]][1] for i in ruta_opt]
    plt.figure(figsize=(6, 5))
    plt.plot(lon, lat, "-o", color="#c53030")
    for i in range(N):
        plt.annotate(NOMBRES[i], (CIUDADES[NOMBRES[i]][1], CIUDADES[NOMBRES[i]][0]), fontsize=7)
    plt.title("Ruta óptima encontrada con A*")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.tight_layout()
    plt.savefig("mapa_ruta_optima.png", dpi=150)
    plt.show()

    with open("resultados_parte_a.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print("\nListo.")


main()