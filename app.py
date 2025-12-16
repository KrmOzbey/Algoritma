import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import heapq
import time

# --- 1. SAYFA VE STİL AYARLARI ---
st.set_page_config(
    page_title="Pathfinder Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS ile Modern Dashboard Görünümü
st.markdown("""
    <style>
        /* Genel Arka Plan */
        .stApp {
            background-color: #0E1117;
        }
        
        /* Sidebar Özelleştirme */
        [data-testid="stSidebar"] {
            background-color: #161B22;
            border-right: 1px solid #30363D;
        }
        
        /* Başlıklar */
        h1, h2, h3 {
            color: #E6EDF3 !important;
            font-family: 'Segoe UI', sans-serif;
        }
        
        /* Metrik Kutuları */
        [data-testid="stMetricValue"] {
            font-size: 24px;
            color: #58A6FF;
        }
        
        /* Tablo Stili */
        [data-testid="stDataFrame"] {
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 8px;
            padding: 10px;
        }
        
        /* Buton Stili */
        div.stButton > button {
            background-color: #238636;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: bold;
            width: 100%;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #2EA043;
            border-color: #2EA043;
        }
        
        /* Radyo Butonları ve Sliderlar */
        .stSlider > div > div > div > div {
            background-color: #58A6FF;
        }
        [data-testid="stMarkdownContainer"] p {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. ALGORİTMA MANTIKLARI (Aynı Kalıyor) ---
def euclidean_dist(node1, node2, positions):
    x1, y1 = positions[node1]
    x2, y2 = positions[node2]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

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

def create_graph(num_nodes, k_neighbors, min_w, max_w):
    G = nx.Graph()
    pos = {}
    for i in range(num_nodes):
        pos[i] = (random.randint(0, 1000), random.randint(0, 1000))
        G.add_node(i, pos=pos[i])
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
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for k in range(len(comps)-1):
            u, v = list(comps[k])[0], list(comps[k+1])[0]
            G.add_edge(u, v, weight=random.randint(min_w, max_w))
    return G, pos

# --- 3. YENİ SIDEBAR TASARIMI ---
with st.sidebar:
    st.title("🎛️ Kontrol Paneli")
    
    st.markdown("### 1. Harita Konfigürasyonu")
    with st.expander("🌍 Harita Ayarları", expanded=True):
        node_count = st.slider("Şehir Sayısı", 20, 200, 80)
        edge_density = st.slider("Bağlantı Yoğunluğu", 2, 8, 4)
    
    with st.expander("⚖️ Ağırlık/Maliyet", expanded=False):
        min_w = st.number_input("Min Ağırlık", 1, 50, 1)
        max_w = st.number_input("Max Ağırlık", 1, 50, 10)
    
    st.markdown("### 2. Görselleştirme")
    # İSTEĞİNİZ ÜZERİNE BURAYA ALINDI
    selected_algo_view = st.selectbox(
        "Haritada Gösterilecek Yol:",
        ["Karşılaştırmalı (Hepsi)", "Sadece Dijkstra (Mavi)", "Sadece A* (Yeşil)", "Sadece Bellman-Ford (Mor)"]
    )
    
    st.markdown("---")
    if st.button("🔄 Haritayı Yeniden Oluştur"):
        st.session_state['G'], st.session_state['pos'] = create_graph(node_count, edge_density, min_w, max_w)
        st.rerun()

# --- 4. ANA EKRAN DÜZENİ ---
if 'G' not in st.session_state:
    st.session_state['G'], st.session_state['pos'] = create_graph(node_count, edge_density, min_w, max_w)

G = st.session_state['G']
pos = st.session_state['pos']
nodes = list(G.nodes)
start_node = nodes[0]
end_node = nodes[-1]

# Algoritmaları Çalıştır
results = []

# Dijkstra
t1 = time.perf_counter()
d_cost, d_path, d_exp = dijkstra_algo(G, start_node, end_node)
d_time = (time.perf_counter() - t1) * 1000
results.append({"Algoritma": "Dijkstra", "Süre (ms)": d_time, "Maliyet": d_cost, "Genişletilen": d_exp, "Yol": d_path})

# A*
t1 = time.perf_counter()
a_cost, a_path, a_exp = a_star_algo(G, start_node, end_node, pos)
a_time = (time.perf_counter() - t1) * 1000
results.append({"Algoritma": "A*", "Süre (ms)": a_time, "Maliyet": a_cost, "Genişletilen": a_exp, "Yol": a_path})

# Bellman-Ford
if node_count <= 150:
    t1 = time.perf_counter()
    b_cost, b_path, b_exp = bellman_ford_algo(G, start_node, end_node)
    b_time = (time.perf_counter() - t1) * 1000
    results.append({"Algoritma": "Bellman-Ford", "Süre (ms)": b_time, "Maliyet": b_cost, "Genişletilen": b_exp, "Yol": b_path})
else:
    results.append({"Algoritma": "Bellman-Ford", "Süre (ms)": 0, "Maliyet": 0, "Genişletilen": 0, "Yol": []})

df_res = pd.DataFrame(results)

# --- LAYOUT (SOL: HARİTA, SAĞ: TABLO) ---
col_map, col_stats = st.columns([5, 3], gap="medium")

with col_map:
    st.subheader("📍 Simülasyon Haritası")
    
    # Harita Stili - Koyu Tema ile Bütünleşik
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0E1117') # Streamlit arka plan rengiyle aynı
    ax.set_facecolor('#0E1117')
    
    # Ağ Çizimi
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color='#30363D', ax=ax, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edge_color='#30363D', alpha=0.4, ax=ax)
    
    # Başlangıç ve Bitiş
    nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='#238636', node_size=150, ax=ax, label="Start")
    nx.draw_networkx_nodes(G, pos, nodelist=[end_node], node_color='#DA3633', node_size=150, ax=ax, label="End")
    
    path_width = 2.5
    
    # Yolları Çiz
    if "Dijkstra" in selected_algo_view or "Hepsi" in selected_algo_view:
        if d_path:
            edges = list(zip(d_path, d_path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='#58A6FF', width=path_width+2, alpha=0.6, label="Dijkstra", ax=ax)
            
    if "Bellman" in selected_algo_view or "Hepsi" in selected_algo_view:
        if len(results) > 2 and results[2]["Yol"]:
            path = results[2]["Yol"]
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='#A371F7', width=path_width, style='dotted', label="Bellman-Ford", ax=ax)

    if "A*" in selected_algo_view or "Hepsi" in selected_algo_view:
        if a_path:
            edges = list(zip(a_path, a_path[1:]))
            color = '#F1E05A' if a_cost > d_cost else '#3FB950' # Sarı uyarı, yeşil başarılı
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, width=path_width, style='dashed', label="A*", ax=ax)

    ax.legend(loc='upper left', facecolor='#161B22', edgecolor='#30363D', labelcolor='#E6EDF3', fontsize=10)
    ax.axis('off')
    st.pyplot(fig)
    
    # Harita altı uyarı
    if a_cost > d_cost:
        st.warning(f"⚠️ A* algoritması {a_cost - d_cost:.1f} birim sapma yaptı! (Heuristic Uyumsuzluğu)")

with col_stats:
    st.subheader("📊 Performans Analizi")
    
    # 1. Tablo
    st.markdown("##### 🏁 Sonuç Özeti")
    st.dataframe(
        df_res[["Algoritma", "Süre (ms)", "Maliyet", "Genişletilen"]].style.highlight_min(axis=0, color="#1F6FEB"),
        use_container_width=True,
        hide_index=True
    )
    
    # 2. Grafikler
    st.markdown("##### ⏱️ Süre Karşılaştırması (ms)")
    st.bar_chart(df_res.set_index("Algoritma")["Süre (ms)"], color="#58A6FF")
    
    st.markdown("##### 🔍 İşlem Yükü (Node Sayısı)")
    st.bar_chart(df_res.set_index("Algoritma")["Genişletilen"], color="#A371F7")
