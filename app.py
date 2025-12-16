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
    page_title="Algoritma Simülasyonu",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- RENK PALETİ ---
COLOR_BG_LIGHT = "#E3F2FD"      # Ana Arka Plan
COLOR_SIDEBAR_BG = "#154360"    # Sidebar Arka Planı
COLOR_TEXT_MAIN = "#000000"     # Ana Ekran Yazıları
# YENİ: Sidebar için özel gri tonu
COLOR_SIDEBAR_TEXT_GRAY = "#B0BEC5"  # Sidebar Yazıları (Okunaklı Gri)
COLOR_ACCENT_RED = "#C0392B"    # Kırmızı Vurgular
COLOR_NODE_BRIGHT = "#3498DB"   # Düğüm Rengi
COLOR_EDGE_LIGHT = "#CFD8DC"    # Kenar Rengi
COLOR_CHART_TEXT = "#546E7A"    # Ana Ekran Grafik Yazıları (Koyu Gri)

# Özel CSS
st.markdown(f"""
    <style>
        /* 1. Genel Sayfa Arka Planı */
        .stApp {{
            background-color: {COLOR_BG_LIGHT};
        }}
        
        /* 2. ANA EKRAN YAZILARI (SİYAH) */
        h1, h2, h3, h4, h5, p, span, li {{
            color: {COLOR_TEXT_MAIN} !important;
            font-family: 'Segoe UI', sans-serif;
        }}
        
        /* 3. Sidebar Genel Ayarları */
        [data-testid="stSidebar"] {{
            background-color: {COLOR_SIDEBAR_BG};
        }}
        
        /* --- SIDEBAR YAZI RENGİ DÜZENLEMESİ (GRİ YAPILDI) --- */
        /* Sidebar'daki Başlıklar, Label'lar ve normal yazılar GRİ olsun */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] div {{
            color: {COLOR_SIDEBAR_TEXT_GRAY} !important;
        }}
        
        /* Dropdown kutusunun içindeki seçili metin rengi */
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div {{
            color: {COLOR_SIDEBAR_TEXT_GRAY} !important;
            -webkit-text-fill-color: {COLOR_SIDEBAR_TEXT_GRAY} !important;
        }}
        
        /* Dropdown ok simgesi rengi */
        [data-testid="stSidebar"] .stSelectbox svg {{
            fill: {COLOR_SIDEBAR_TEXT_GRAY} !important;
        }}
        /* -------------------------------------------------- */
        
        /* 4. Buton Stili */
        div.stButton > button {{
            background-color: {COLOR_ACCENT_RED};
            color: white !important;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            transition: 0.3s;
        }}
        div.stButton > button:hover {{
            background-color: #A93226;
        }}
        
        /* 5. Expander Başlıkları (Sidebar içi) */
        [data-testid="stSidebar"] .streamlit-expanderHeader {{
            color: {COLOR_SIDEBAR_BG} !important; /* Başlık koyu mavi */
            background-color: {COLOR_SIDEBAR_TEXT_GRAY}; /* Zemin gri */
        }}
        
        /* Harita Konteyner (Dış Gölge Efekti) */
        .map-container {{
            box-shadow: 0 6px 14px rgba(0,0,0,0.2);
            border-radius: 4px; /* Matplotlib çerçevesi ile uyum için köşe yuvarlaklığını azalttım */
            overflow: hidden;
            padding: 5px;
            background-color: white; /* Çerçevenin daha net durması için beyaz zemin */
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

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/tr/6/62/Gazi_%C3%9Cniversitesi_Logosu.png", width=100)
    st.title("Algoritma Labı")
    st.markdown("---")
    
    st.markdown("### ⚙️ Ayarlar")
    
    with st.expander("🌍 Harita Konfigürasyonu", expanded=True):
        node_count = st.slider("Şehir Sayısı", 20, 300, 80)
        edge_density = st.slider("Bağlantı Yoğunluğu", 2, 8, 3)
    
    with st.expander("⚖️ Yol Maliyetleri", expanded=False):
        min_w = st.number_input("Min Ağırlık", 1, 50, 1)
        max_w = st.number_input("Max Ağırlık", 1, 50, 50)
    
    # BU KISIMDAKİ YAZILAR ARTIK GRİ OLACAK
    st.markdown("### 👁️ Görünüm")
    selected_algo_view = st.selectbox(
        "Rotayı Göster:",
        ["Karşılaştırmalı (Hepsi)", "Sadece Dijkstra", "Sadece A*", "Sadece Bellman-Ford"]
    )
    
    st.markdown("---")
    if st.button("🔄 Haritayı Yeniden Oluştur"):
        st.session_state['G'], st.session_state['pos'] = create_graph(node_count, edge_density, min_w, max_w)
        st.rerun()

# --- 4. ANA EKRAN ---

if 'G' not in st.session_state:
    st.session_state['G'], st.session_state['pos'] = create_graph(node_count, edge_density, min_w, max_w)

G = st.session_state['G']
pos = st.session_state['pos']
nodes = list(G.nodes)
start_node = nodes[0]
end_node = nodes[-1]

# Hesaplamalar
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
if node_count <= 200: 
    t1 = time.perf_counter()
    b_cost, b_path, b_exp = bellman_ford_algo(G, start_node, end_node)
    b_time = (time.perf_counter() - t1) * 1000
    results.append({"Algoritma": "Bellman-Ford", "Süre (ms)": b_time, "Maliyet": b_cost, "Genişletilen": b_exp, "Yol": b_path})
else:
    results.append({"Algoritma": "Bellman-Ford", "Süre (ms)": 0, "Maliyet": 0, "Genişletilen": 0, "Yol": []})

df_res = pd.DataFrame(results)

# --- HARİTA GÖRSELLEŞTİRME ---
st.subheader("📍 Simülasyon Haritası")

with st.container():
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    
    plt.figure(figsize=(14, 7))
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(COLOR_BG_LIGHT)
    ax.set_facecolor(COLOR_BG_LIGHT)

    # --- HARİTA ÇERÇEVESİ EKLENDİ ---
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)  # Çerçeveyi görünür yap
        spine.set_color(COLOR_SIDEBAR_BG) # Koyu mavi renk
        spine.set_linewidth(3)   # Kalınlık

    # Ağ Çizimi
    nx.draw_networkx_nodes(G, pos, node_size=60, node_color=COLOR_NODE_BRIGHT, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color=COLOR_EDGE_LIGHT, alpha=0.6, width=1, ax=ax)

    # Başlangıç ve Bitiş
    nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color="white", edgecolors=COLOR_SIDEBAR_BG, linewidths=3, node_size=250, ax=ax, label="Başlangıç")
    nx.draw_networkx_nodes(G, pos, nodelist=[end_node], node_color=COLOR_ACCENT_RED, edgecolors="white", linewidths=2, node_size=250, ax=ax, label="Hedef")

    path_width = 4

    # Rotalar
    if "Dijkstra" in selected_algo_view or "Hepsi" in selected_algo_view:
        if d_path:
            edges = list(zip(d_path, d_path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=COLOR_SIDEBAR_BG, width=path_width+1, alpha=0.7, label="Dijkstra", ax=ax)
            
    if "Bellman" in selected_algo_view or "Hepsi" in selected_algo_view:
        if len(results) > 2 and results[2]["Yol"]:
            path = results[2]["Yol"]
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='#9B59B6', width=path_width-1, style='dotted', label="Bellman-Ford", ax=ax)

    if "A*" in selected_algo_view or "Hepsi" in selected_algo_view:
        if a_path:
            edges = list(zip(a_path, a_path[1:]))
            color = '#F39C12' if a_cost > d_cost else COLOR_ACCENT_RED
            style = 'dashed'
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, width=path_width, style=style, label="A*", ax=ax)

    # Lejant
    legend = ax.legend(
        loc='upper left', 
        frameon=True, 
        facecolor='white', 
        edgecolor=COLOR_SIDEBAR_BG,
        framealpha=1,
        labelcolor='black',
        fontsize=11,
        borderpad=1
    )
    
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if a_cost > d_cost:
    st.error(f"⚠️ A* Algoritması {a_cost - d_cost:.1f} birim daha maliyetli bir yol buldu! (Heuristic Yanılgısı)")

st.divider()

# --- ANALİZ BÖLÜMÜ ---
st.subheader("📊 Performans Analizi")

col_stats, col_charts = st.columns([1, 1], gap="large")

with col_stats:
    st.markdown("##### 📝 Sonuç Tablosu")
    st.dataframe(
        df_res[["Algoritma", "Süre (ms)", "Maliyet", "Genişletilen"]].style.format({"Süre (ms)": "{:.2f}"}),
        use_container_width=True,
        hide_index=True
    )

with col_charts:
    st.markdown("##### ⏱️ Grafiksel Karşılaştırma")
    tab1, tab2 = st.tabs(["Zaman (ms)", "İşlem Yükü"])
    
    chart_text_color = COLOR_CHART_TEXT
    
    # Altair Eksen Konfigürasyonu
    axis_config = alt.Axis(
        labelColor=chart_text_color, 
        titleColor=chart_text_color, 
        gridColor="#CFD8DC"
    )

    with tab1:
        # Zaman Grafiği
        chart_time = alt.Chart(df_res).mark_bar(color=COLOR_SIDEBAR_BG, cornerRadiusEnd=5).encode(
            x=alt.X('Süre (ms)', axis=axis_config),
            y=alt.Y('Algoritma', axis=axis_config, sort='-x'),
            tooltip=['Algoritma', alt.Tooltip('Süre (ms)', format='.2f')]
        ).properties(
            height=250,
            background='transparent'
        ).configure_text(color=chart_text_color).configure_axis(
            labelColor=chart_text_color,
            titleColor=chart_text_color
        )
        st.altair_chart(chart_time, use_container_width=True)
        
    with tab2:
        # İşlem Yükü Grafiği
        chart_exp = alt.Chart(df_res).mark_bar(color=COLOR_ACCENT_RED, cornerRadiusEnd=5).encode(
            x=alt.X('Genişletilen', axis=axis_config, title='Genişletilen Düğüm Sayısı'),
            y=alt.Y('Algoritma', axis=axis_config, sort='-x'),
            tooltip=['Algoritma', 'Genişletilen']
        ).properties(
            height=250,
            background='transparent'
        ).configure_axis(
            labelColor=chart_text_color,
            titleColor=chart_text_color
        )
        st.altair_chart(chart_exp, use_container_width=True)
