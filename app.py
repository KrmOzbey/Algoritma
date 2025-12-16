import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import heapq
import time

# --- SAYFA YAPILANDIRMASI (Geniş Mod) ---
st.set_page_config(
    page_title="Algoritma Laboratuvarı",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. ALGORİTMA MANTIKLARI (Rapordan) ---

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
    queue = [(0, 0, start, [])] # (f, g, node, path)
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
    expanded = 0 # İşlem yükü sayacı
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

# --- 2. HARİTA OLUŞTURMA ---
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

# --- 3. ARAYÜZ TASARIMI ---

# Yan Menü (Sidebar)
with st.sidebar:
    st.header("⚙️ Harita Ayarları")
    node_count = st.slider("Şehir Sayısı (Node)", 20, 200, 80)
    edge_density = st.slider("Bağlantı Yoğunluğu", 2, 8, 4)
    st.divider()
    st.header("⚖️ Ağırlık Ayarları")
    min_w = st.number_input("Min Maliyet", 1, 50, 1)
    max_w = st.number_input("Max Maliyet", 1, 50, 10)
    
    if st.button("🔄 Yeni Harita Oluştur", type="primary"):
        st.session_state['G'], st.session_state['pos'] = create_graph(node_count, edge_density, min_w, max_w)
        st.rerun()

# Session State Başlatma
if 'G' not in st.session_state:
    st.session_state['G'], st.session_state['pos'] = create_graph(node_count, edge_density, min_w, max_w)

G = st.session_state['G']
pos = st.session_state['pos']
nodes = list(G.nodes)
start_node = nodes[0]
end_node = nodes[-1]

# --- ANA EKRAN DÜZENİ (2 KOLON) ---
col_map, col_stats = st.columns([3, 2]) # Sol taraf geniş (Harita), Sağ taraf dar (Veriler)

with col_stats:
    st.subheader("📊 Analiz Paneli")
    
    # Tüm algoritmaları çalıştır ve verileri topla
    results = []
    
    # Dijkstra
    t1 = time.perf_counter()
    d_cost, d_path, d_exp = dijkstra_algo(G, start_node, end_node)
    d_time = (time.perf_counter() - t1) * 1000
    results.append({"Algoritma": "Dijkstra", "Süre (ms)": d_time, "Maliyet": d_cost, "İşlem (Node)": d_exp, "Yol": d_path})
    
    # A*
    t1 = time.perf_counter()
    a_cost, a_path, a_exp = a_star_algo(G, start_node, end_node, pos)
    a_time = (time.perf_counter() - t1) * 1000
    results.append({"Algoritma": "A* (A-Star)", "Süre (ms)": a_time, "Maliyet": a_cost, "İşlem (Node)": a_exp, "Yol": a_path})
    
    # Bellman-Ford (Sadece node sayısı düşükse hızlı çalışır)
    if node_count <= 150:
        t1 = time.perf_counter()
        b_cost, b_path, b_exp = bellman_ford_algo(G, start_node, end_node)
        b_time = (time.perf_counter() - t1) * 1000
        results.append({"Algoritma": "Bellman-Ford", "Süre (ms)": b_time, "Maliyet": b_cost, "İşlem (Node)": b_exp, "Yol": b_path})
    else:
        results.append({"Algoritma": "Bellman-Ford", "Süre (ms)": 0, "Maliyet": 0, "İşlem (Node)": 0, "Yol": []})
        st.info("Bellman-Ford performans koruması nedeniyle 150 node üzerinde devre dışı.")

    # DataFrame Oluştur
    df_res = pd.DataFrame(results)
    
    # 1. Tablo Gösterimi (Resimdeki "Run Tablosu")
    st.markdown("##### 1. Karşılaştırma Tablosu")
    st.dataframe(df_res[["Algoritma", "Süre (ms)", "Maliyet", "İşlem (Node)"]].style.format({"Süre (ms)": "{:.2f}"}), use_container_width=True)
    
    # 2. Grafikler (Resimdeki Bar Chartlar)
    st.markdown("##### 2. Performans Grafikleri")
    tab1, tab2 = st.tabs(["⏱️ Süre Karşılaştırması", "🔍 İşlem Yükü (Expanded)"])
    
    with tab1:
        st.bar_chart(df_res.set_index("Algoritma")["Süre (ms)"], color="#3498db")
    with tab2:
        st.bar_chart(df_res.set_index("Algoritma")["İşlem (Node)"], color="#e74c3c")

    # Görselleştirme Seçimi
    st.divider()
    selected_algo_view = st.selectbox("Haritada Hangi Yolu Göster?", ["Karşılaştırmalı (Hepsi)", "Sadece Dijkstra", "Sadece A*", "Sadece Bellman-Ford"])


with col_map:
    st.subheader("🗺️ Harita Simülasyonu")
    
    # Grafik Çizimi - Matplotlib ile "Dark Mode" havası
    plt.style.use('dark_background') # Koyu tema
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0E1117') # Streamlit arka plan rengine uyum
    ax.set_facecolor('#0E1117')
    
    # Tüm ağı çiz
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color='#444444', ax=ax, alpha=0.6)
    nx.draw_networkx_edges(G, pos, edge_color='#555555', alpha=0.3, ax=ax)
    
    # Başlangıç ve Bitiş
    nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='#2ecc71', node_size=150, ax=ax, label="Başlangıç")
    nx.draw_networkx_nodes(G, pos, nodelist=[end_node], node_color='#e74c3c', node_size=150, ax=ax, label="Hedef")
    
    # Yolları Çiz
    path_width = 2
    
    if "Dijkstra" in selected_algo_view or "Hepsi" in selected_algo_view:
        if d_path:
            edges = list(zip(d_path, d_path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='#3498db', width=path_width+2, alpha=0.8, label="Dijkstra", ax=ax)
            
    if "Bellman" in selected_algo_view or "Hepsi" in selected_algo_view:
        # Bellman path genelde Dijkstra ile aynıdır, hafif kaydırarak veya farklı stille çizelim
        if len(results) > 2 and results[2]["Yol"]:
            path = results[2]["Yol"]
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='#9b59b6', width=path_width, style='dotted', label="Bellman-Ford", ax=ax)

    if "A*" in selected_algo_view or "Hepsi" in selected_algo_view:
        if a_path:
            edges = list(zip(a_path, a_path[1:]))
            # Eğer hatalı (suboptimal) ise sarı yap, değilse yeşil
            color = '#f1c40f' if a_cost > d_cost else '#2ecc71'
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, width=path_width, style='dashed', label="A*", ax=ax)

    ax.legend(loc='upper left', facecolor='#262730', edgecolor='white', labelcolor='white')
    ax.axis('off')
    st.pyplot(fig)
    
    # Alt Bilgi Notu
    if a_cost > d_cost:
        st.warning(f"⚠️ DİKKAT: A* algoritması {a_cost - d_cost:.2f} birim daha maliyetli bir yol buldu. (Heuristic tutarsızlığı)")
    else:
        st.success("✅ A* algoritması optimal yolu başarıyla buldu.")
