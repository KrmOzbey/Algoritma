import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import heapq
import time
import altair as alt

# --- 1. SAYFA VE STİL AYARLARI ---
st.set_page_config(
    page_title="Gazi Üni - Algoritma Simülasyonu",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- RENK PALETİ (Profesyonel Gri/Antrasit Tema) ---
COLOR_BG_MAIN = "#F8F9FA"       # Ana Sayfa Arka Planı
COLOR_SIDEBAR_BG = "#263238"    # Sidebar (Koyu Antrasit)
COLOR_TEXT_MAIN = "#212121"     # Ana Yazılar
COLOR_SIDEBAR_TEXT = "#ECEFF1"  # Sidebar Yazıları

# Harita Elemanları
COLOR_NODE_DEFAULT = "#90A4AE"  # Pasif Şehirler
COLOR_NODE_START = "#2E7D32"    # Başlangıç (Yeşil)
COLOR_NODE_END = "#C62828"      # Bitiş (Kırmızı)
COLOR_EDGE_DEFAULT = "#CFD8DC"  # Pasif Yollar

# Algoritma Renkleri
COLOR_DIJKSTRA = "#455A64"      # Dijkstra (Gri)
COLOR_ASTAR = "#E65100"         # A* (Turuncu)
COLOR_BELLMAN = "#6A1B9A"       # Bellman (Mor)
COLOR_GREEDY = "#00897B"        # Greedy (Teal/Turkuaz)

# Özel CSS
st.markdown(f"""
    <style>
        .stApp {{ background-color: {COLOR_BG_MAIN}; }}
        h1, h2, h3, h4, h5, p, span, li {{
            color: {COLOR_TEXT_MAIN} !important;
            font-family: 'Segoe UI', Roboto, sans-serif;
        }}
        [data-testid="stSidebar"] {{ background-color: {COLOR_SIDEBAR_BG}; }}
        [data-testid="stSidebar"] * {{ color: {COLOR_SIDEBAR_TEXT} !important; }}
        
        /* Widget Stilleri */
        [data-testid="stSidebar"] .stSelectbox > div > div {{
             background-color: #37474F !important;
             color: {COLOR_SIDEBAR_TEXT} !important;
        }}
        div.stButton > button {{
            background-color: {COLOR_NODE_END};
            color: white !important;
            border-radius: 6px;
            border: none;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
        }}
        div.stButton > button:hover {{
            background-color: #B71C1C;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
    </style>
""", unsafe_allow_html=True)

# --- 2. YARDIMCI FONKSİYONLAR ---

def euclidean_dist(node1, node2, positions):
    x1, y1 = positions[node1]
    x2, y2 = positions[node2]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def create_graph(num_nodes, k_neighbors, min_w, max_w):
    G = nx.Graph()
    pos = {}
    # Rastgele düğümler oluştur
    for i in range(num_nodes):
        pos[i] = (random.randint(0, 1000), random.randint(0, 1000))
        G.add_node(i, pos=pos[i])
    
    # En yakın komşuları bağla
    for i in range(num_nodes):
        dists = []
        x1, y1 = pos[i]
        for j in range(num_nodes):
            if i == j: continue
            x2, y2 = pos[j]
            d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            dists.append((d, j))
        dists.sort(key=lambda x: x[0])
        
        for _, neighbor in dists[:k_neighbors]:
            if not G.has_edge(i, neighbor):
                G.add_edge(i, neighbor, weight=random.randint(min_w, max_w))
    
    # Grafın bağlantılı olduğundan emin ol
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for k in range(len(comps)-1):
            u, v = list(comps[k])[0], list(comps[k+1])[0]
            G.add_edge(u, v, weight=random.randint(min_w, max_w))
            
    return G, pos

# --- 3. ALGORİTMALAR ---

def dijkstra_algo(graph, start, goal):
    queue = [(0, start, [])]
    visited = set()
    expanded = 0
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited: continue
        visited.add(node)
        expanded += 1
        path = path + [node]
        if node == goal: return cost, path, expanded
        for neighbor, attr in graph[node].items():
            if neighbor not in visited:
                heapq.heappush(queue, (cost + attr['weight'], neighbor, path))
    return float('inf'), [], expanded

def a_star_algo(graph, start, goal, positions):
    queue = [(0, 0, start, [])] 
    visited = set()
    expanded = 0
    g_scores = {node: float('inf') for node in graph.nodes}
    g_scores[start] = 0
    
    while queue:
        _, current_g, node, path = heapq.heappop(queue)
        if node == goal: return current_g, path + [node], expanded
        
        if current_g > g_scores[node]: continue
        visited.add(node)
        expanded += 1
        path = path + [node]
        
        for neighbor, attr in graph[node].items():
            weight = attr['weight']
            new_g = current_g + weight
            if new_g < g_scores[neighbor]:
                g_scores[neighbor] = new_g
                h = euclidean_dist(neighbor, goal, positions)
                heapq.heappush(queue, (new_g + h, new_g, neighbor, path))
    return float('inf'), [], expanded

def bellman_ford_algo(graph, start, goal):
    dist = {node: float('inf') for node in graph.nodes}
    pred = {node: None for node in graph.nodes}
    dist[start] = 0
    expanded = 0
    nodes = list(graph.nodes)
    edges = list(graph.edges(data=True))
    
    for _ in range(len(nodes) - 1):
        change = False
        for u, v, data in edges:
            expanded += 1
            w = data['weight']
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                change = True
            elif dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                pred[u] = v
                change = True
        if not change: break
        
    if dist[goal] == float('inf'): return float('inf'), [], expanded
    path = []
    curr = goal
    while curr is not None:
        path.insert(0, curr)
        if curr == start: break
        curr = pred[curr]
    return dist[goal], path, expanded

def greedy_bfs_algo(graph, start, goal, positions):
    """
    Greedy Best-First Search:
    Maliyete (g) bakmaz, sadece hedefe olan kuş uçuşu mesafeye (h) bakar.
    """
    # Kuyruk yapısı: (heuristic_distance, node, path)
    queue = [(euclidean_dist(start, goal, positions), start, [])]
    visited = set()
    expanded = 0
    
    while queue:
        _, node, path = heapq.heappop(queue)
        
        if node in visited: continue
        visited.add(node)
        expanded += 1
        path = path + [node]
        
        if node == goal:
            # Yol bulundu, gerçek maliyeti hesaplayalım
            total_cost = 0
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                total_cost += graph[u][v]['weight']
            return total_cost, path, expanded
            
        for neighbor, attr in graph[node].items():
            if neighbor not in visited:
                h = euclidean_dist(neighbor, goal, positions)
                heapq.heappush(queue, (h, neighbor, path))
                
    return float('inf'), [], expanded

# --- 4. SIDEBAR VE KONTROLLER ---
with st.sidebar:
    st.markdown(f'<div style="text-align: center;"><img src="https://upload.wikimedia.org/wikipedia/tr/6/62/Gazi_%C3%9Cniversitesi_Logosu.png" width="90" style="filter: brightness(0.9);"></div>', unsafe_allow_html=True)
    st.title("Algoritma Labı")
    st.markdown("---", unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Ayarlar")
    with st.expander("🌍 Harita Konfigürasyonu", expanded=True):
        node_count = st.slider("Şehir Sayısı", 20, 200, 60)
        edge_density = st.slider("Bağlantı Yoğunluğu", 2, 6, 3)
    
    with st.expander("⚖️ Yol Maliyetleri", expanded=False):
        min_w = st.number_input("Min Ağırlık", 1, 50, 1)
        max_w = st.number_input("Max Ağırlık", 1, 50, 50)
    
    st.markdown("### 👁️ Görünüm")
    selected_algo_view = st.selectbox(
        "Rotayı Göster:",
        ["Karşılaştırmalı (Hepsi)", "Sadece Dijkstra", "Sadece A*", "Sadece Bellman-Ford", "Sadece Greedy AI"]
    )
    
    st.markdown("---", unsafe_allow_html=True)
    if st.button("🔄 Haritayı Yeniden Oluştur"):
        st.session_state['G'], st.session_state['pos'] = create_graph(node_count, edge_density, min_w, max_w)
        st.rerun()

# --- 5. ANA EKRAN VE HESAPLAMALAR ---

if 'G' not in st.session_state:
    st.session_state['G'], st.session_state['pos'] = create_graph(node_count, edge_density, min_w, max_w)

G = st.session_state['G']
pos = st.session_state['pos']
nodes = list(G.nodes)
start_node = nodes[0]
end_node = nodes[-1]

results = []

# 1. Dijkstra
t1 = time.perf_counter()
d_cost, d_path, d_exp = dijkstra_algo(G, start_node, end_node)
d_time = (time.perf_counter() - t1) * 1000
results.append({"Algoritma": "Dijkstra", "Süre (ms)": d_time, "Maliyet": d_cost, "Genişletilen": d_exp, "Yol": d_path})

# 2. A*
t1 = time.perf_counter()
a_cost, a_path, a_exp = a_star_algo(G, start_node, end_node, pos)
a_time = (time.perf_counter() - t1) * 1000
results.append({"Algoritma": "A*", "Süre (ms)": a_time, "Maliyet": a_cost, "Genişletilen": a_exp, "Yol": a_path})

# 3. Bellman-Ford (Çok düğümde yavaşlayabilir)
t1 = time.perf_counter()
b_cost, b_path, b_exp = bellman_ford_algo(G, start_node, end_node)
b_time = (time.perf_counter() - t1) * 1000
results.append({"Algoritma": "Bellman-Ford", "Süre (ms)": b_time, "Maliyet": b_cost, "Genişletilen": b_exp, "Yol": b_path})

# 4. Greedy Best-First Search (Gerçek AI Yaklaşımı)
t1 = time.perf_counter()
g_cost, g_path, g_exp = greedy_bfs_algo(G, start_node, end_node, pos)
g_time = (time.perf_counter() - t1) * 1000
results.append({"Algoritma": "Greedy Best-First", "Süre (ms)": g_time, "Maliyet": g_cost, "Genişletilen": g_exp, "Yol": g_path})

df_res = pd.DataFrame(results)

# --- GÖRSELLEŞTİRME ---
st.subheader("📍 Algoritma Simülasyon Haritası")

with st.container():
    plt.figure(figsize=(14, 7))
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(COLOR_BG_MAIN)
    ax.set_facecolor(COLOR_BG_MAIN)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

    # Ağ Çizimi
    nx.draw_networkx_nodes(G, pos, node_size=90, node_color=COLOR_NODE_DEFAULT, ax=ax, alpha=1.0, edgecolors='#B0BEC5', linewidths=1)
    nx.draw_networkx_edges(G, pos, edge_color=COLOR_EDGE_DEFAULT, alpha=0.5, width=1.2, ax=ax)

    # Başlangıç ve Bitiş
    nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color=COLOR_NODE_START, edgecolors="white", linewidths=2, node_size=350, ax=ax, label="Başlangıç")
    nx.draw_networkx_nodes(G, pos, nodelist=[end_node], node_color=COLOR_NODE_END, edgecolors="white", linewidths=2, node_size=350, ax=ax, label="Hedef")

    path_width = 5

    # Dijkstra Yolu
    if "Dijkstra" in selected_algo_view or "Hepsi" in selected_algo_view:
        if d_path:
            edges = list(zip(d_path, d_path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=COLOR_DIJKSTRA, width=path_width+2, alpha=0.6, label="Dijkstra", ax=ax)

    # Bellman-Ford Yolu
    if "Bellman" in selected_algo_view or "Hepsi" in selected_algo_view:
        if b_path:
            edges = list(zip(b_path, b_path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=COLOR_BELLMAN, width=path_width, style='dotted', label="Bellman-Ford", ax=ax)

    # A* Yolu
    if "A*" in selected_algo_view or "Hepsi" in selected_algo_view:
        if a_path:
            edges = list(zip(a_path, a_path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=COLOR_ASTAR, width=path_width, style='dashed', label="A*", ax=ax)

    # Greedy Yolu
    if "Greedy" in selected_algo_view or "Hepsi" in selected_algo_view:
        if g_path:
            edges = list(zip(g_path, g_path[1:]))
            # Eğer Greedy optimal yolu bulamazsa kesikli çizgi yap, bulursa düz çizgi
            style = 'solid' if g_cost == d_cost else 'dashdot'
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=COLOR_GREEDY, width=3, style=style, label="Greedy Best-First", ax=ax)

    legend = ax.legend(loc='upper left', frameon=True, facecolor=COLOR_BG_MAIN, edgecolor=COLOR_EDGE_DEFAULT, labelcolor=COLOR_TEXT_MAIN, fontsize=10)
    st.pyplot(fig, use_container_width=True)

# Uyarılar
if a_cost > d_cost:
    st.warning(f"⚠️ A* Algoritması optimalden saptı. ({a_cost - d_cost:.1f} birim fark). Sebep: Rastgele kenar ağırlıkları fiziksel mesafeden küçük.")

if g_cost > d_cost:
    st.info(f"ℹ️ Greedy (Açgözlü) Yaklaşım daha hızlı hesapladı ancak en kısa yolu garanti etmedi. Fazladan maliyet: {g_cost - d_cost:.1f}")

st.divider()

# --- ANALİZ BÖLÜMÜ ---
st.subheader("📊 Performans Analizi")
col_stats, col_charts = st.columns([1, 1], gap="large")

with col_stats:
    st.markdown("##### 📝 Sonuç Tablosu")
    st.dataframe(
        df_res[["Algoritma", "Süre (ms)", "Maliyet", "Genişletilen"]].style.format({"Süre (ms)": "{:.3f}"}),
        use_container_width=True,
        hide_index=True
    )

with col_charts:
    st.markdown("##### ⏱️ Grafiksel Karşılaştırma")
    tab1, tab2 = st.tabs(["Zaman (ms)", "İşlem Yükü"])
    
    axis_config = alt.Axis(labelColor=COLOR_TEXT_MAIN, titleColor=COLOR_TEXT_MAIN, gridColor="#E0E0E0")
    
    color_scale = alt.Scale(
        domain=['Dijkstra', 'A*', 'Bellman-Ford', 'Greedy Best-First'],
        range=[COLOR_DIJKSTRA, COLOR_ASTAR, COLOR_BELLMAN, COLOR_GREEDY]
    )

    with tab1:
        chart_time = alt.Chart(df_res).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X('Süre (ms)', axis=axis_config),
            y=alt.Y('Algoritma', axis=axis_config, sort='-x'),
            tooltip=['Algoritma', alt.Tooltip('Süre (ms)', format='.4f')],
            color=alt.Color('Algoritma', scale=color_scale, legend=None)
        ).properties(height=250, background='transparent').configure_view(strokeWidth=0)
        st.altair_chart(chart_time, use_container_width=True)
        
    with tab2:
        chart_exp = alt.Chart(df_res).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X('Genişletilen', axis=axis_config, title='Genişletilen Düğüm Sayısı'),
            y=alt.Y('Algoritma', axis=axis_config, sort='-x'),
            tooltip=['Algoritma', 'Genişletilen'],
            color=alt.Color('Algoritma', scale=color_scale, legend=None)
        ).properties(height=250, background='transparent').configure_view(strokeWidth=0)
        st.altair_chart(chart_exp, use_container_width=True)
