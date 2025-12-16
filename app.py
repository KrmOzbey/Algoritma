import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import heapq
import time
import altair as alt # Grafik renk kontrolü için gerekli

# --- 1. SAYFA VE STİL AYARLARI ---
st.set_page_config(
    page_title="Neon Pathfinder Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Koyu Tema CSS
st.markdown("""
    <style>
        /* Genel Arka Plan */
        .stApp {
            background-color: #0E1117;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #161B22;
            border-right: 1px solid #30363D;
        }
        
        /* Başlıklar ve Metinler (Açık Renk) */
        h1, h2, h3, h4, h5, p, label, span, div {
            color: #E6EDF3 !important;
            font-family: 'Segoe UI', Roboto, sans-serif;
        }
        
        /* Tablo Stili */
        [data-testid="stDataFrame"] {
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 8px;
        }
        
        /* Buton Stili (Neon Yeşil) */
        div.stButton > button {
            background-color: #238636;
            color: white !important;
            border: 1px solid #2EA043;
            border-radius: 6px;
            padding: 0.6rem 1rem;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #2EA043;
            box-shadow: 0 0 10px #2EA043;
        }
        
        /* Expander Stili */
        .streamlit-expanderHeader {
            background-color: #21262D;
            border-radius: 6px;
        }

        /* Uyarı Kutuları */
        .stAlert {
            background-color: #161B22;
            color: #E6EDF3;
            border: 1px solid #30363D;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. ALGORİTMA FONKSİYONLARI (Aynı) ---
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

# --- 3. SIDEBAR (Koyu Tema) ---
with st.sidebar:
    st.title("🎛️ Kontrol Paneli")
    st.markdown("---")
    
    with st.expander("🌍 Harita Ayarları", expanded=True):
        node_count = st.slider("Şehir Sayısı", 20, 250, 100)
        edge_density = st.slider("Bağlantı Yoğunluğu", 2, 8, 4)
    
    with st.expander("⚖️ Maliyet Ayarları", expanded=False):
        min_w = st.number_input("Min Ağırlık", 1, 50, 1)
        max_w = st.number_input("Max Ağırlık", 1, 100, 20)
    
    st.markdown("### 👁️ Görünüm")
    selected_algo_view = st.selectbox(
        "Rotayı Göster:",
        ["Karşılaştırmalı (Hepsi)", "Sadece Dijkstra", "Sadece A*", "Sadece Bellman-Ford"]
    )
    
    st.markdown("---")
    if st.button("🔄 HARİTAYI YENİLE"):
        st.session_state['G'], st.session_state['pos'] = create_graph(node_count, edge_density, min_w, max_w)
        st.rerun()

# --- 4. ANA EKRAN HESAPLAMALARI ---
if 'G' not in st.session_state:
    st.session_state['G'], st.session_state['pos'] = create_graph(node_count, edge_density, min_w, max_w)

G = st.session_state['G']
pos = st.session_state['pos']
nodes = list(G.nodes)
start_node = nodes[0]
end_node = nodes[-1]

results = []
# Dijkstra
t1 = time.perf_counter()
d_cost, d_path, d_exp = dijkstra_algo(G, start_node, end_node)
results.append({"Algoritma": "Dijkstra", "Süre (ms)": (time.perf_counter() - t1) * 1000, "Maliyet": d_cost, "Genişletilen": d_exp, "Yol": d_path})
# A*
t1 = time.perf_counter()
a_cost, a_path, a_exp = a_star_algo(G, start_node, end_node, pos)
results.append({"Algoritma": "A*", "Süre (ms)": (time.perf_counter() - t1) * 1000, "Maliyet": a_cost, "Genişletilen": a_exp, "Yol": a_path})
# Bellman-Ford
if node_count <= 180: 
    t1 = time.perf_counter()
    b_cost, b_path, b_exp = bellman_ford_algo(G, start_node, end_node)
    results.append({"Algoritma": "Bellman-Ford", "Süre (ms)": (time.perf_counter() - t1) * 1000, "Maliyet": b_cost, "Genişletilen": b_exp, "Yol": b_path})
else:
    results.append({"Algoritma": "Bellman-Ford", "Süre (ms)": 0, "Maliyet": 0, "Genişletilen": 0, "Yol": []})

df_res = pd.DataFrame(results)

# --- BÖLÜM 1: GELİŞMİŞ HARİTA GÖRÜNÜMÜ ---
col_map, col_stats = st.columns([5, 3], gap="medium")

with col_map:
    st.subheader("📍 Simülasyon Haritası")
    
    # Koyu Tema ve Çerçeve Ayarları
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0E1117') # Sayfa arka planı
    ax.set_facecolor('#0E1117') # Grafik arka planı
    
    # Çerçeve Rengi (Neon Mavi)
    FRAME_COLOR = '#58A6FF'
    for spine in ax.spines.values():
        spine.set_edgecolor(FRAME_COLOR)
        spine.set_linewidth(2)

    # --- RENK PALETİ (NEON/PARLAK) ---
    NODE_COLOR = '#4A5568'   # Metalik Gri-Mavi
    EDGE_COLOR = '#2D3748'   # Koyu Mavi-Gri
    START_COLOR = '#00FF7F'  # Neon Yeşil (Spring Green)
    END_COLOR = '#FF4500'    # Neon Kırmızı (Orange Red)
    DIJKSTRA_COLOR = '#00BFFF' # Elektrik Mavisi
    ASTAR_COLOR = '#FFD700'    # Altın Sarısı
    BELLMAN_COLOR = '#FF00FF'  # Parlak Macenta

    # Temel Ağ Çizimi
    nx.draw_networkx_nodes(G, pos, node_size=40, node_color=NODE_COLOR, ax=ax, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color=EDGE_COLOR, alpha=0.6, ax=ax)
    
    # Başlangıç ve Bitiş
    nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color=START_COLOR, node_size=180, ax=ax, label="Başlangıç")
    nx.draw_networkx_nodes(G, pos, nodelist=[end_node], node_color=END_COLOR, node_size=180, ax=ax, label="Hedef")
    
    path_width = 2.5
    
    # Rotaları Çiz
    if "Dijkstra" in selected_algo_view or "Hepsi" in selected_algo_view:
        if d_path:
            edges = list(zip(d_path, d_path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=DIJKSTRA_COLOR, width=path_width+2, alpha=0.7, label="Dijkstra", ax=ax)
            
    if "Bellman" in selected_algo_view or "Hepsi" in selected_algo_view:
        if len(results) > 2 and results[2]["Yol"]:
            path = results[2]["Yol"]
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=BELLMAN_COLOR, width=path_width, style='dotted', label="Bellman-Ford", ax=ax)

    if "A*" in selected_algo_view or "Hepsi" in selected_algo_view:
        if a_path:
            edges = list(zip(a_path, a_path[1:]))
            # Hata varsa turuncu yap, yoksa altın sarısı
            color = '#FF8C00' if a_cost > d_cost else ASTAR_COLOR
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, width=path_width, style='dashed', label="A*", ax=ax)

    # Lejant (Legend) Ayarları - Okunabilir Gri Yazı
    legend = ax.legend(loc='upper left', facecolor='#161B22', edgecolor=FRAME_COLOR, labelcolor='#B0BEC5', fontsize=10)
    ax.axis('off') # Eksenleri gizle ama çerçeveyi koru
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)
    
    if a_cost > d_cost:
        st.warning(f"⚠️ A* Algoritması {a_cost - d_cost:.1f} birim sapma yaptı! (Heuristic Yanılgısı)")

# --- BÖLÜM 2: ANALİZ (Okunabilir Grafikler) ---
with col_stats:
    st.subheader("📊 Performans Analizi")
    
    # Tablo
    st.markdown("##### 📝 Sonuç Özeti")
    st.dataframe(
        df_res[["Algoritma", "Süre (ms)", "Maliyet", "Genişletilen"]].style.format({"Süre (ms)": "{:.2f}"}),
        use_container_width=True,
        hide_index=True
    )
    
    # --- ALTAIR GRAFİKLERİ (Gri Yazı Rengi İçin) ---
    st.markdown("##### 📈 Grafiksel Karşılaştırma")
    tab1, tab2 = st.tabs(["⏱️ Zaman (ms)", "🔍 İşlem Yükü"])
    
    # Ortak Grafik Ayarları (Gri Eksen Yazıları)
    axis_config = alt.Axis(labelColor='#B0BEC5', titleColor='#B0BEC5', gridColor='#30363D')
    
    base_chart = alt.Chart(df_res).encode(
        x=alt.X('Algoritma', axis=axis_config),
        tooltip=['Algoritma', 'Süre (ms)', 'Genişletilen', 'Maliyet']
    )
    
    with tab1:
        # Zaman Grafiği (Mavi)
        chart_time = base_chart.mark_bar(color='#58A6FF').encode(
            y=alt.Y('Süre (ms)', axis=axis_config)
        ).properties(background='transparent') # Şeffaf arka plan
        st.altair_chart(chart_time, use_container_width=True)
        
    with tab2:
        # İşlem Yükü Grafiği (Mor)
        chart_exp = base_chart.mark_bar(color='#A371F7').encode(
            y=alt.Y('Genişletilen', axis=axis_config)
        ).properties(background='transparent')
        st.altair_chart(chart_exp, use_container_width=True)
