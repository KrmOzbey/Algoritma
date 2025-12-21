import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import heapq
import time
import altair as alt
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
# GCNConv modelinize uygun olarak:
from torch_geometric.nn import GCNConv

# --- 1. SAYFA VE STİL AYARLARI ---
st.set_page_config(
    page_title="Algoritma Simülasyonu & AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- RENK PALETİ ---
COLOR_BG_LIGHT = "#E3F2FD"      
COLOR_SIDEBAR_BG = "#154360"    
COLOR_TEXT_MAIN = "#000000"     
COLOR_SIDEBAR_TEXT_GRAY = "#B0BEC5"  
COLOR_ACCENT_RED = "#C0392B"    
COLOR_NODE_BRIGHT = "#3498DB"   
COLOR_EDGE_LIGHT = "#CFD8DC"    
COLOR_CHART_TEXT = "#546E7A"    
COLOR_AI_CYAN = "#00E5FF" # Yapay Zeka Rengi (Neon Turkuaz)

# Özel CSS
st.markdown(f"""
    <style>
        .stApp {{ background-color: {COLOR_BG_LIGHT}; }}
        h1, h2, h3, h4, h5, p, span, li {{ color: {COLOR_TEXT_MAIN} !important; font-family: 'Segoe UI', sans-serif; }}
        [data-testid="stSidebar"] {{ background-color: {COLOR_SIDEBAR_BG}; }}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] div {{
            color: {COLOR_SIDEBAR_TEXT_GRAY} !important;
        }}
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div {{
            color: {COLOR_SIDEBAR_TEXT_GRAY} !important;
            -webkit-text-fill-color: {COLOR_SIDEBAR_TEXT_GRAY} !important;
        }}
        [data-testid="stSidebar"] .stSelectbox svg {{ fill: {COLOR_SIDEBAR_TEXT_GRAY} !important; }}
        div.stButton > button {{
            background-color: {COLOR_ACCENT_RED}; color: white !important; border: none;
            border-radius: 6px; font-weight: bold; transition: 0.3s;
        }}
        div.stButton > button:hover {{ background-color: #A93226; }}
        [data-testid="stSidebar"] .streamlit-expanderHeader {{
            color: {COLOR_SIDEBAR_BG} !important; background-color: {COLOR_SIDEBAR_TEXT_GRAY};
        }}
        .map-container {{
            box-shadow: 0 6px 14px rgba(0,0,0,0.2); border-radius: 4px; overflow: hidden;
            padding: 5px; background-color: white;
        }}
    </style>
""", unsafe_allow_html=True)

# --- 2. YAPAY ZEKA MODEL MİMARİSİ (Eğitim Kodundan Alındı) ---
# DİKKAT: Buradaki parametreler (hidden_channels=256 vb.) eğitimdeki ile AYNI olmalıdır.

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(0.2) # dropout_p
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.gcn1(x, edge_index)
        x = torch.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        x = torch.relu(self.bn2(x))
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, node_dim, hidden_dim, out_dim):
        super().__init__()
        self.lstm = nn.LSTM(node_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

class GNNPathModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, max_path_len):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, num_nodes)
        self.decoder = Decoder(num_nodes, 512, num_nodes) # lstm_hidden_dim = 512

# --- Model Yükleme ve Veri Hazırlama Yardımcıları ---

@st.cache_resource
def load_ai_model():
    # EĞİTİMDE KULLANILAN SABİTLER (Bunları eğitim kodunuzdan teyit edin)
    TRAIN_NUM_NODES = 81  # Örneğin 81 il ile eğittiyseniz
    HIDDEN_CHANNELS = 256
    IN_CHANNELS = 6       # 4 Feature (Degree, Cent, Clust, Page) + 2 Mask
    
    # Model mimarisini başlat
    model = GNNPathModel(
        in_channels=IN_CHANNELS, 
        hidden_channels=HIDDEN_CHANNELS, 
        out_channels=TRAIN_NUM_NODES, # Output layer boyutu eğitimdeki node sayısına sabitlenir
        num_nodes=TRAIN_NUM_NODES,
        max_path_len=50 # Tahmini
    )
    
    try:
        # GitHub'a yüklediğiniz model dosyasının adı
        model.load_state_dict(torch.load('Model3_2.pt', map_location=torch.device('cpu')))
        model.eval()
        return model, TRAIN_NUM_NODES
    except FileNotFoundError:
        return None, TRAIN_NUM_NODES

def prepare_data_for_ai(G, start_node, end_node, train_num_nodes):
    # NetworkX özelliklerini çıkar (Eğitimdeki gibi)
    num_nodes_current = len(G.nodes)
    
    # Eğer harita, modelin eğitildiği boyuttan büyükse model çalışamaz (Fixed Output Layer)
    # Bu yüzden sadece modelin kapasitesi dahilindeyse çalıştıracağız.
    
    degree = np.array([val for (node, val) in G.degree()])
    try:
        centrality = np.array([val for (node, val) in nx.betweenness_centrality(G).items()])
        clustering = np.array([val for (node, val) in nx.clustering(G).items()])
        pagerank = np.array([val for (node, val) in nx.pagerank(G).items()])
    except:
        # Hata durumunda dummy veri
        centrality = np.zeros(num_nodes_current)
        clustering = np.zeros(num_nodes_current)
        pagerank = np.zeros(num_nodes_current)

    # Feature Matrix (N x 4)
    features = np.column_stack((degree, centrality, clustering, pagerank))
    base_features = torch.tensor(features, dtype=torch.float)
    
    # Maskeler (Start/End)
    start_mask = torch.zeros(num_nodes_current, 1)
    end_mask = torch.zeros(num_nodes_current, 1)
    start_mask[start_node] = 1
    end_mask[end_node] = 1
    
    # Tüm featureları birleştir (N x 6)
    x = torch.cat([base_features, start_mask, end_mask], dim=1)
    
    # Edge Index
    edges = list(G.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # Yönsüz graf olduğu için ters yönleri de ekleyelim
    edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
    
    return x, edge_index

def run_ai_inference(model, G, start_node, end_node, train_num_nodes):
    # Eğer haritadaki düğüm sayısı modelin output layerından büyükse tahmin yapamayız
    if len(G.nodes) > train_num_nodes:
        return 0, [], False
        
    x, edge_index = prepare_data_for_ai(G, start_node, end_node, train_num_nodes)
    
    t_start = time.perf_counter()
    path = [start_node]
    
    with torch.no_grad():
        # Encoder
        node_emb = model.encoder(x, edge_index)
        
        # Decoder (LSTM Loop)
        input_emb = node_emb[start_node].unsqueeze(0).unsqueeze(0)
        hidden = None
        visited = set([start_node])
        curr = start_node
        
        # Max 50 adım tahmin et
        for _ in range(50):
            out, hidden = model.decoder.lstm(input_emb, hidden)
            logits = model.decoder.fc_out(out.squeeze(1))
            
            # Maskeleme (Gidilebilecek komşular)
            neighbors = list(G.neighbors(curr))
            allowed = set(neighbors) - visited
            if end_node in neighbors: allowed.add(end_node)
            
            if not allowed: break # Gidecek yer yok
            
            # Logits maskeleme (sadece allowed indexler kalsın)
            full_mask = torch.ones_like(logits) * -float('inf')
            allowed_indices = torch.tensor(list(allowed), dtype=torch.long)
            full_mask[0, allowed_indices] = logits[0, allowed_indices]
            
            pred_node = full_mask.argmax(dim=-1).item()
            
            path.append(pred_node)
            visited.add(pred_node)
            curr = pred_node
            input_emb = node_emb[pred_node].unsqueeze(0).unsqueeze(0)
            
            if curr == end_node:
                break
                
    t_end = time.perf_counter()
    return (t_end - t_start) * 1000, path, True

# --- 3. KLASİK ALGORİTMALAR ---
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

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/tr/6/62/Gazi_%C3%9Cniversitesi_Logosu.png", width=100)
    st.title("Algoritma & AI")
    st.markdown("---")
    
    with st.expander("🌍 Harita Konfigürasyonu", expanded=True):
        node_count = st.slider("Şehir Sayısı", 20, 300, 80)
        edge_density = st.slider("Bağlantı Yoğunluğu", 2, 8, 3)
    
    with st.expander("⚖️ Yol Maliyetleri", expanded=False):
        min_w = st.number_input("Min Ağırlık", 1, 50, 1)
        max_w = st.number_input("Max Ağırlık", 1, 50, 50)
    
    st.markdown("### 👁️ Görünüm")
    selected_algo_view = st.selectbox(
        "Rotayı Göster:",
        ["Karşılaştırmalı (Hepsi)", "Sadece Dijkstra", "Sadece A*", "Sadece Bellman-Ford", "Sadece Yapay Zeka"]
    )
    
    st.markdown("---")
    if st.button("🔄 Haritayı Yeniden Oluştur"):
        st.session_state['G'], st.session_state['pos'] = create_graph(node_count, edge_density, min_w, max_w)
        st.rerun()

# --- 5. ANA EKRAN VE ÇALIŞTIRMA ---

if 'G' not in st.session_state:
    st.session_state['G'], st.session_state['pos'] = create_graph(node_count, edge_density, min_w, max_w)

G = st.session_state['G']
pos = st.session_state['pos']
nodes = list(G.nodes)
start_node = nodes[0]
end_node = nodes[-1]

# Model Yükleme
ai_model, train_num_nodes = load_ai_model()

# Sonuçları Hesapla
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

# 3. Bellman-Ford
if node_count <= 200: 
    t1 = time.perf_counter()
    b_cost, b_path, b_exp = bellman_ford_algo(G, start_node, end_node)
    b_time = (time.perf_counter() - t1) * 1000
    results.append({"Algoritma": "Bellman-Ford", "Süre (ms)": b_time, "Maliyet": b_cost, "Genişletilen": b_exp, "Yol": b_path})
else:
    results.append({"Algoritma": "Bellman-Ford", "Süre (ms)": 0, "Maliyet": 0, "Genişletilen": 0, "Yol": []})

# 4. Yapay Zeka (GNN + LSTM)
if ai_model is not None:
    ai_time, ai_path, success = run_ai_inference(ai_model, G, start_node, end_node, train_num_nodes)
    
    # AI maliyeti (Bulduğu yolun ağırlıklarını topla)
    ai_cost = 0
    if ai_path:
        for i in range(len(ai_path)-1):
            if G.has_edge(ai_path[i], ai_path[i+1]):
                ai_cost += G[ai_path[i]][ai_path[i+1]]['weight']
                
    if success:
        results.append({"Algoritma": "Yapay Zeka (GNN)", "Süre (ms)": ai_time, "Maliyet": ai_cost, "Genişletilen": 0, "Yol": ai_path})
    else:
        # Eğer node sayısı modelin limitini aşarsa
        st.toast(f"AI Model {train_num_nodes} node ile eğitildi, şu an {len(G.nodes)} node var. AI devre dışı.", icon="⚠️")
else:
    # Model dosyası yoksa
    st.toast("Model dosyası (Model3_2.pt) bulunamadı.", icon="📁")

df_res = pd.DataFrame(results)

# --- HARİTA GÖRSELLEŞTİRME ---
st.subheader("📍 Simülasyon Haritası")

with st.container():
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    
    plt.figure(figsize=(14, 7))
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(COLOR_BG_LIGHT)
    ax.set_facecolor(COLOR_BG_LIGHT)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLOR_SIDEBAR_BG)
        spine.set_linewidth(3)

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
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=COLOR_SIDEBAR_BG, width=path_width+2, alpha=0.7, label="Dijkstra", ax=ax)
            
    if "Bellman" in selected_algo_view or "Hepsi" in selected_algo_view:
        if len(results) > 2 and results[2]["Yol"]:
            path = results[2]["Yol"]
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='#9B59B6', width=path_width, style='dotted', label="Bellman-Ford", ax=ax)

    if "A*" in selected_algo_view or "Hepsi" in selected_algo_view:
        if a_path:
            edges = list(zip(a_path, a_path[1:]))
            color = '#F39C12' if a_cost > d_cost else COLOR_ACCENT_RED
            style = 'dashed'
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, width=path_width, style=style, label="A*", ax=ax)

    # YAPAY ZEKA GÖRSELLEŞTİRMESİ
    if "Yapay Zeka" in selected_algo_view or "Hepsi" in selected_algo_view:
        ai_res = next((r for r in results if r["Algoritma"] == "Yapay Zeka (GNN)"), None)
        if ai_res and ai_res["Yol"]:
            path = ai_res["Yol"]
            edges = list(zip(path, path[1:]))
            # AI, en üste çizilir, parlak turkuaz
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=COLOR_AI_CYAN, width=path_width-1, style='solid', label="Yapay Zeka (GNN)", ax=ax)

    legend = ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor=COLOR_SIDEBAR_BG, framealpha=1, labelcolor='black', fontsize=11, borderpad=1)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if a_cost > d_cost:
    st.error(f"⚠️ A* Algoritması {a_cost - d_cost:.1f} birim sapma yaptı!")

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
    axis_config = alt.Axis(labelColor=chart_text_color, titleColor=chart_text_color, gridColor="#CFD8DC")

    with tab1:
        chart_time = alt.Chart(df_res).mark_bar(color=COLOR_SIDEBAR_BG, cornerRadiusEnd=5).encode(
            x=alt.X('Süre (ms)', axis=axis_config),
            y=alt.Y('Algoritma', axis=axis_config, sort='-x'),
            tooltip=['Algoritma', alt.Tooltip('Süre (ms)', format='.2f')],
            color=alt.condition(
                alt.datum.Algoritma == 'Yapay Zeka (GNN)',
                alt.value(COLOR_AI_CYAN),  # AI için özel renk
                alt.value(COLOR_SIDEBAR_BG)
            )
        ).properties(height=250, background='transparent').configure_text(color=chart_text_color).configure_axis(
            labelColor=chart_text_color, titleColor=chart_text_color
        )
        st.altair_chart(chart_time, use_container_width=True)
        
    with tab2:
        chart_exp = alt.Chart(df_res).mark_bar(color=COLOR_ACCENT_RED, cornerRadiusEnd=5).encode(
            x=alt.X('Genişletilen', axis=axis_config, title='Genişletilen Düğüm Sayısı'),
            y=alt.Y('Algoritma', axis=axis_config, sort='-x'),
            tooltip=['Algoritma', 'Genişletilen']
        ).properties(height=250, background='transparent').configure_axis(
            labelColor=chart_text_color, titleColor=chart_text_color
        )
        st.altair_chart(chart_exp, use_container_width=True)
