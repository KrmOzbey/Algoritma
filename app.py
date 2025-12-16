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
    page_title="Algoritma Simülasyonu",
    layout="wide",
    initial_sidebar_state="expanded"
)

# RENK PALETİ TANIMLARI
COLOR_DARK_BLUE = "#326789"
COLOR_MED_BLUE = "#78A6C8"
COLOR_BG_LIGHT = "#E9EEF2"
COLOR_ACCENT_RED = "#E65C4F"

# Özel CSS ile Renk Paleti Uygulaması
st.markdown(f"""
    <style>
        /* Genel Arka Plan */
        .stApp {{
            background-color: {COLOR_BG_LIGHT};
        }}
        
        /* Sidebar Stili */
        [data-testid="stSidebar"] {{
            background-color: #ffffff;
            border-right: 1px solid {COLOR_MED_BLUE};
        }}
        
        /* Başlıklar */
        h1, h2, h3, h4, h5 {{
            color: {COLOR_DARK_BLUE} !important;
            font-family: 'Helvetica Neue', sans-serif;
        }}
        
        /* Metinler */
        p, label, span {{
            color: #2c3e50;
        }}
        
        /* Tablo Stili */
        [data-testid="stDataFrame"] {{
            background-color: #ffffff;
            border: 1px solid {COLOR_MED_BLUE};
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        /* Buton Stili (Accent Red) */
        div.stButton > button {{
            background-color: {COLOR_ACCENT_RED};
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.6rem 1rem;
            font-weight: 600;
            width: 100%;
            transition: all 0.2s ease;
        }}
        div.stButton > button:hover {{
            background-color: #c0392b;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        /* Expander ve Kutular */
        .streamlit-expanderHeader {{
            color: {COLOR_DARK_BLUE};
            font-weight: bold;
        }}
    </style>
""", unsafe_allow_html=True)

# --- 2. ALGORİTMA FONKSİYONLARI ---
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

# --- 3. SIDEBAR (Kontrol Paneli) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/tr/6/62/Gazi_%C3%9Cniversitesi_Logosu.png", width=100)
    st.title("Algoritma Labı")
    
    st.markdown("### ⚙️ Ayarlar")
    
    # Harita Ayarları
    with st.expander("🌍 Harita Konfigürasyonu", expanded=True):
        node_count = st.slider("Şehir Sayısı", 20, 300, 100)
        edge_density = st.slider("Bağlantı Yoğunluğu", 2, 8, 4)
    
    # Ağırlık Ayarları
    with st.expander("⚖️ Yol Maliyetleri", expanded=False):
        min_w = st.number_input("Min Ağırlık", 1, 50, 1)
        max_w = st.number_input("Max Ağırlık", 1, 50, 20)
    
    # Görselleştirme Seçimi
    st.markdown("### 👁️ Görünüm")
    selected_algo_view = st.selectbox(
        "Gösterilecek Rota:",
        ["Karşılaştırmalı (Hepsi)", "Sadece Dijkstra (Mavi)", "Sadece A* (Kırmızı)", "Sadece Bellman-Ford (Mor)"]
    )
    
    st.markdown("---")
    if st.button("🔄 Haritayı Yeniden Oluştur"):
        st.session_state['G'], st.session_state['pos'] = create_graph(node_count, edge_density, min_w, max_w)
        st.rerun()

# --- 4. ANA EKRAN MANTIĞI ---

if 'G' not in st.session_state:
    st.session_state['G'], st.session_state['pos'] = create_graph(node_count, edge_density, min_w, max_w)

G = st.session_state['G']
pos = st.session_state['pos']
nodes = list(G.nodes)
start_node = nodes[0]
end_node = nodes[-1]

# Algoritmaları Hesapla
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
if node_count <= 200: # Performans için limit
    t1 = time.perf_counter()
    b_cost, b_path, b_exp = bellman_ford_algo(G, start_node, end_node)
    b_time = (time.perf_counter() - t1) * 1000
    results.append({"Algoritma": "Bellman-Ford", "Süre (ms)": b_time, "Maliyet": b_cost, "Genişletilen": b_exp, "Yol": b_path})
else:
    results.append({"Algoritma": "Bellman-Ford", "Süre (ms)": 0, "Maliyet": 0, "Genişletilen": 0, "Yol": []})

df_res = pd.DataFrame(results)

# --- BÖLÜM 1: HARİTA (Tam Genişlik) ---
st.subheader("📍 Simülasyon Haritası")

# Grafik Ayarları (Palete Uygun)
plt.figure(figsize=(14, 6)) # Daha geniş ve büyük harita
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(COLOR_BG_LIGHT) # Arka plan rengi
ax.set_facecolor(COLOR_BG_LIGHT)

# Ağ Çizimi
# Düğümler: Koyu Mavi, Kenarlar: Orta Mavi
nx.draw_networkx_nodes(G, pos, node_size=40, node_color=COLOR_DARK_BLUE, ax=ax, alpha=0.8)
nx.draw_networkx_edges(G, pos, edge_color=COLOR_MED_BLUE, alpha=0.3, ax=ax)

# Başlangıç (Yeşil) ve Bitiş (Kiremit Kırmızısı)
nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='#2ecc71', node_size=200, ax=ax, label="Başlangıç")
nx.draw_networkx_nodes(G, pos, nodelist=[end_node], node_color=COLOR_ACCENT_RED, node_size=200, ax=ax, label="Hedef")

path_width = 3

# Rotaları Çiz
if "Dijkstra" in selected_algo_view or "Hepsi" in selected_algo_view:
    if d_path:
        edges = list(zip(d_path, d_path[1:]))
        # Dijkstra: Koyu Mavi Rota
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=COLOR_DARK_BLUE, width=path_width+2, alpha=0.5, label="Dijkstra", ax=ax)
        
if "Bellman" in selected_algo_view or "Hepsi" in selected_algo_view:
    if len(results) > 2 and results[2]["Yol"]:
        path = results[2]["Yol"]
        edges = list(zip(path, path[1:]))
        # Bellman: Morumsu (Palet dışı kontrast için)
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='#9b59b6', width=path_width, style='dotted', label="Bellman-Ford", ax=ax)

if "A*" in selected_algo_view or "Hepsi" in selected_algo_view:
    if a_path:
        edges = list(zip(a_path, a_path[1:]))
        # A*: Paletteki Kiremit Rengi (Hata varsa Sarı)
        color = '#f1c40f' if a_cost > d_cost else COLOR_ACCENT_RED
        style = 'dashed'
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, width=path_width, style=style, label="A*", ax=ax)

ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor=COLOR_MED_BLUE)
ax.axis('off')
st.pyplot(fig, use_container_width=True)

# Harita altı mesaj
if a_cost > d_cost:
    st.warning(f"⚠️ A* Algoritması {a_cost - d_cost:.1f} birim daha maliyetli bir yol buldu! (Heuristic Yanılgısı)")

st.divider()

# --- BÖLÜM 2: PERFORMANS ANALİZİ (Yan Yana) ---
st.subheader("📊 Performans Analizi")

col_stats, col_charts = st.columns([1, 1], gap="large")

with col_stats:
    st.markdown("##### 📝 Sonuç Tablosu")
    # Tabloyu stilize et
    st.dataframe(
        df_res[["Algoritma", "Süre (ms)", "Maliyet", "Genişletilen"]].style.format({"Süre (ms)": "{:.2f}"}),
        use_container_width=True,
        hide_index=True
    )
    
    st.info("""
    **Tablo Yorumu:**
    * **Süre:** Algoritmanın çalışma hızı. A* genellikle en hızlıdır.
    * **Maliyet:** Bulunan yolun toplam uzunluğu.
    * **Genişletilen:** Algoritmanın kaç şehri ziyaret ettiği.
    """)

with col_charts:
    st.markdown("##### ⏱️ Grafiksel Karşılaştırma")
    
    tab1, tab2 = st.tabs(["Zaman (ms)", "İşlem Yükü"])
    
    with tab1:
        # Renk paletindeki Koyu Maviyi kullan
        st.bar_chart(df_res.set_index("Algoritma")["Süre (ms)"], color=COLOR_DARK_BLUE)
        
    with tab2:
        # Renk paletindeki Kiremit Rengini kullan
        st.bar_chart(df_res.set_index("Algoritma")["Genişletilen"], color=COLOR_ACCENT_RED)
