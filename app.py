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
from torch_geometric.nn import GCNConv

# --- 1. SAYFA VE STİL AYARLARI ---
st.set_page_config(
    page_title="AI vs Algoritmalar (Türkiye)",
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
COLOR_EDGE_LIGHT = "#90A4AE"    
COLOR_CHART_TEXT = "#546E7A"    
COLOR_AI_CYAN = "#00E5FF" 

# Özel CSS
st.markdown(f"""
    <style>
        .stApp {{ background-color: {COLOR_BG_LIGHT}; }}
        h1, h2, h3, h4, h5, p, span, li {{ color: {COLOR_TEXT_MAIN} !important; font-family: 'Segoe UI', sans-serif; }}
        [data-testid="stSidebar"] {{ background-color: {COLOR_SIDEBAR_BG}; }}
        [data-testid="stSidebar"] * {{ color: {COLOR_SIDEBAR_TEXT_GRAY} !important; }}
        div.stButton > button {{
            background-color: {COLOR_ACCENT_RED}; color: white !important; border: none;
            border-radius: 6px; font-weight: bold; transition: 0.3s;
        }}
        .map-container {{
            box-shadow: 0 6px 14px rgba(0,0,0,0.2); border-radius: 4px; overflow: hidden;
            padding: 5px; background-color: white; border: 2px solid {COLOR_SIDEBAR_BG};
        }}
    </style>
""", unsafe_allow_html=True)

# --- 2. TÜRKİYE HARİTASI VERİSİ (SABİT TOPOLOJİ) ---
RAW_GRAPH_DATA = {
    "Adana": [("Mersin", 6), ("Osmaniye", 14), ("Hatay", 8), ("Kahramanmaraş", 19), ("Niğde", 4)],
    "Adıyaman": [("Malatya", 15), ("Kahramanmaraş", 13), ("Gaziantep", 5), ("Şanlıurfa", 7)],
    "Afyonkarahisar": [("Uşak", 2), ("Kütahya", 17), ("Eskişehir", 9), ("Konya", 11), ("Isparta", 6), ("Denizli", 20)],
    "Ağrı": [("Iğdır", 3), ("Kars", 15), ("Erzurum", 8), ("Muş", 11), ("Bitlis", 6), ("Van", 17)],
    "Aksaray": [("Konya", 10), ("Nevşehir", 13), ("Niğde", 14), ("Ankara", 5), ("Kırşehir", 12)],
    "Amasya": [("Samsun", 7), ("Tokat", 2), ("Çorum", 15)],
    "Ankara": [("Kırıkkale", 4), ("Çankırı", 9), ("Eskişehir", 16), ("Konya", 8), ("Aksaray", 18)],
    "Antalya": [("Isparta", 6), ("Burdur", 15), ("Muğla", 12), ("Konya", 10), ("Mersin", 9)],
    "Ardahan": [("Kars", 17), ("Artvin", 6)],
    "Artvin": [("Rize", 8), ("Ardahan", 12)],
    "Aydın": [("İzmir", 13), ("Manisa", 13), ("Denizli", 7), ("Muğla", 20)],
    "Balıkesir": [("Çanakkale", 11), ("Bursa", 16), ("Kütahya", 10), ("Manisa", 18), ("İzmir", 5)],
    "Bartın": [("Kastamonu", 6), ("Zonguldak", 14)],
    "Batman": [("Diyarbakır", 18), ("Siirt", 4), ("Mardin", 15), ("Şırnak", 12)],
    "Bayburt": [("Trabzon", 5), ("Erzurum", 7), ("Gümüşhane", 17)],
    "Bilecik": [("Bursa", 14), ("Eskişehir", 13), ("Kütahya", 16), ("Sakarya", 7)],
    "Bingöl": [("Erzurum", 10), ("Erzincan", 9), ("Elazığ", 6), ("Diyarbakır", 13), ("Muş", 17)],
    "Bitlis": [("Van", 18), ("Muş", 7), ("Siirt", 13), ("Ağrı", 15)],
    "Bolu": [("Düzce", 12), ("Zonguldak", 13), ("Karabük", 7), ("Çankırı", 11), ("Ankara", 8), ("Sakarya", 12)],
    "Burdur": [("Isparta", 6), ("Antalya", 15), ("Denizli", 3)],
    "Bursa": [("Balıkesir", 13), ("Kütahya", 13), ("Bilecik", 5), ("Yalova", 17)],
    "Çanakkale": [("Balıkesir", 18), ("Tekirdağ", 19), ("Edirne", 15)],
    "Çankırı": [("Kastamonu", 9), ("Karabük", 17), ("Bolu", 6), ("Ankara", 8), ("Kırıkkale", 15)],
    "Çorum": [("Amasya", 12), ("Samsun", 20), ("Sinop", 5), ("Kastamonu", 18), ("Yozgat", 9), ("Kırıkkale", 13)],
    "Denizli": [("Uşak", 14), ("Afyonkarahisar", 20), ("Burdur", 3), ("Muğla", 19), ("Aydın", 7)],
    "Diyarbakır": [("Elazığ", 13), ("Bingöl", 19), ("Muş", 17), ("Batman", 4), ("Mardin", 11), ("Şanlıurfa", 18)],
    "Düzce": [("Zonguldak", 20), ("Bolu", 6), ("Sakarya", 13)],
    "Edirne": [("Kırklareli", 9), ("Tekirdağ", 13), ("Çanakkale", 7)],
    "Elazığ": [("Malatya", 13), ("Bingöl", 12), ("Tunceli", 17), ("Diyarbakır", 11)],
    "Erzincan": [("Erzurum", 17), ("Bingöl", 15), ("Tunceli", 9), ("Sivas", 6)],
    "Erzurum": [("Bayburt", 6), ("Erzincan", 13), ("Bingöl", 19), ("Ağrı", 18), ("Kars", 5)],
    "Eskişehir": [("Kütahya", 18), ("Afyonkarahisar", 20), ("Ankara", 5), ("Bilecik", 14)],
    "Gaziantep": [("Kilis", 5), ("Şanlıurfa", 16), ("Adıyaman", 14), ("Osmaniye", 7)],
    "Giresun": [("Trabzon", 9), ("Gümüşhane", 19), ("Ordu", 7), ("Sivas", 16)],
    "Gümüşhane": [("Trabzon", 14), ("Bayburt", 12), ("Erzincan", 17), ("Giresun", 8)],
    "Hakkari": [("Şırnak", 16), ("Van", 7), ("Siirt", 10)],
    "Hatay": [("Osmaniye", 18), ("Adana", 14)],
    "Iğdır": [("Kars", 8), ("Ağrı", 17)],
    "Isparta": [("Afyonkarahisar", 19), ("Burdur", 7), ("Antalya", 12), ("Konya", 17)],
    "İstanbul": [("Kocaeli", 5), ("Tekirdağ", 13)],
    "İzmir": [("Manisa", 13), ("Aydın", 18)],
    "Kahramanmaraş": [("Osmaniye", 10), ("Gaziantep", 11), ("Adıyaman", 15), ("Malatya", 8), ("Adana", 12)],
    "Karabük": [("Bartın", 12), ("Kastamonu", 4), ("Çankırı", 19), ("Bolu", 6)],
    "Karaman": [("Konya", 16), ("Mersin", 13)],
    "Kars": [("Ardahan", 7), ("Iğdır", 14), ("Ağrı", 11), ("Erzurum", 16)],
    "Kastamonu": [("Sinop", 9), ("Çorum", 12), ("Karabük", 8), ("Bartın", 20), ("Çankırı", 7)],
    "Kayseri": [("Nevşehir", 19), ("Yozgat", 8), ("Sivas", 10), ("Niğde", 4)],
    "Kırıkkale": [("Kırşehir", 15), ("Ankara", 18), ("Çankırı", 8), ("Yozgat", 19), ("Çorum", 13)],
    "Kırklareli": [("Tekirdağ", 17), ("Edirne", 13)],
    "Kırşehir": [("Nevşehir", 10), ("Aksaray", 19), ("Kırıkkale", 9), ("Yozgat", 7)],
    "Kilis": [("Gaziantep", 6), ("Şanlıurfa", 20)],
    "Kocaeli": [("Sakarya", 8), ("Bursa", 18), ("Yalova", 16), ("İstanbul", 11)],
    "Konya": [("Aksaray", 6), ("Niğde", 15), ("Karaman", 12), ("Antalya", 10), ("Isparta", 16), ("Afyonkarahisar", 9), ("Eskişehir", 7), ("Ankara", 14)],
    "Kütahya": [("Uşak", 16), ("Afyonkarahisar", 8), ("Eskişehir", 7), ("Bilecik", 19), ("Balıkesir", 2)],
    "Malatya": [("Elazığ", 18), ("Kahramanmaraş", 14), ("Adıyaman", 4), ("Sivas", 6)],
    "Manisa": [("İzmir", 12), ("Aydın", 6), ("Balıkesir", 14)],
    "Mardin": [("Şırnak", 18), ("Batman", 17), ("Diyarbakır", 8), ("Şanlıurfa", 14)],
    "Mersin": [("Karaman", 19), ("Antalya", 7), ("Adana", 17)],
    "Muğla": [("Aydın", 20), ("Denizli", 3), ("Antalya", 9)],
    "Muş": [("Bingöl", 6), ("Bitlis", 9), ("Van", 5), ("Ağrı", 18), ("Diyarbakır", 13)],
    "Nevşehir": [("Aksaray", 14), ("Kırşehir", 8), ("Niğde", 19), ("Kayseri", 5)],
    "Niğde": [("Aksaray", 17), ("Nevşehir", 11), ("Kayseri", 14), ("Adana", 8), ("Konya", 19)],
    "Ordu": [("Samsun", 12), ("Giresun", 15), ("Sivas", 8)],
    "Osmaniye": [("Hatay", 18), ("Adana", 11), ("Kahramanmaraş", 13), ("Gaziantep", 15)],
    "Rize": [("Artvin", 9), ("Trabzon", 16)],
    "Sakarya": [("Kocaeli", 12), ("Bilecik", 13), ("Bolu", 11), ("Düzce", 6)],
    "Samsun": [("Sinop", 7), ("Amasya", 18), ("Tokat", 4), ("Ordu", 5), ("Çorum", 10)],
    "Siirt": [("Bitlis", 13), ("Batman", 4), ("Şırnak", 8), ("Hakkari", 13)],
    "Sinop": [("Kastamonu", 14), ("Samsun", 5), ("Çorum", 10)],
    "Sivas": [("Yozgat", 13), ("Kayseri", 7), ("Malatya", 19), ("Erzincan", 16), ("Giresun", 8), ("Ordu", 5)],
    "Şanlıurfa": [("Gaziantep", 13), ("Adıyaman", 11), ("Mardin", 16), ("Diyarbakır", 4), ("Kilis", 18)],
    "Şırnak": [("Mardin", 17), ("Siirt", 5), ("Hakkari", 6), ("Batman", 18)],
    "Tekirdağ": [("İstanbul", 10), ("Kırklareli", 19), ("Edirne", 12), ("Çanakkale", 13)],
    "Tokat": [("Amasya", 18), ("Sivas", 9), ("Samsun", 5)],
    "Trabzon": [("Rize", 20), ("Gümüşhane", 7), ("Bayburt", 8), ("Giresun", 16)],
    "Tunceli": [("Elazığ", 8), ("Erzincan", 12)],
    "Uşak": [("Kütahya", 10), ("Afyonkarahisar", 17), ("Denizli", 6)],
    "Van": [("Bitlis", 17), ("Ağrı", 3), ("Hakkari", 11)],
    "Yalova": [("Kocaeli", 13), ("Bursa", 12)],
    "Yozgat": [("Çorum", 14), ("Kırıkkale", 11), ("Kırşehir", 7), ("Kayseri", 10), ("Sivas", 16)],
    "Zonguldak": [("Bartın", 17), ("Karabük", 18), ("Bolu", 7), ("Düzce", 9)]
}

# Şehirleri alfabetik sıraya dizip indexleme
SORTED_NODES = sorted(list(RAW_GRAPH_DATA.keys()))
NODE_TO_IDX = {node: i for i, node in enumerate(SORTED_NODES)}
NUM_NODES = len(SORTED_NODES) # 81

# --- 3. MODEL MİMARİSİ ---
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(0.2)
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
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, max_path_len, lstm_hidden_dim=512):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, num_nodes)
        self.decoder = Decoder(num_nodes, lstm_hidden_dim, num_nodes)

# --- 4. YARDIMCI FONKSİYONLAR ---

@st.cache_resource
def load_ai_model():
    model_path = 'Model3_2.pt'
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        # Otomatik boyut algılama
        if 'encoder.fc.weight' in state_dict:
            weight_shape = state_dict['encoder.fc.weight'].shape
            num_nodes = weight_shape[0]  # Modelin eğitildiği node sayısı
            hidden_dim = weight_shape[1]
            lstm_dim = state_dict.get('decoder.fc_out.weight', torch.zeros(1, 512)).shape[1]
        else:
            return None

        model = GNNPathModel(6, hidden_dim, num_nodes, num_nodes, 50, lstm_dim)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except:
        return None

def create_randomized_graph(min_w, max_w):
    """Türkiye haritasını oluşturur ancak ağırlıkları rastgele atar"""
    G = nx.Graph()
    # Düğümleri ekle
    for node in SORTED_NODES:
        G.add_node(node)
    
    # Kenarları ekle (Ağırlıkları rastgele ver)
    for u, edges in RAW_GRAPH_DATA.items():
        for v, _ in edges:
            if not G.has_edge(u, v):
                w = random.randint(min_w, max_w)
                G.add_edge(u, v, weight=w)
    
    pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)
    return G, pos

def prepare_features_for_ai(G, start_node, end_node):
    # Modelin tanıdığı sırayla özellik çıkar
    degree = np.array([val for (node, val) in G.degree(SORTED_NODES)])
    centrality = np.array([nx.betweenness_centrality(G)[n] for n in SORTED_NODES])
    clustering = np.array([nx.clustering(G)[n] for n in SORTED_NODES])
    pagerank = np.array([nx.pagerank(G)[n] for n in SORTED_NODES])

    features = np.column_stack((degree, centrality, clustering, pagerank))
    base_features = torch.tensor(features, dtype=torch.float)

    start_mask = torch.zeros(NUM_NODES, 1)
    end_mask = torch.zeros(NUM_NODES, 1)
    start_mask[NODE_TO_IDX[start_node]] = 1
    end_mask[NODE_TO_IDX[end_node]] = 1

    # [81, 6]
    x = torch.cat([base_features, start_mask, end_mask], dim=1)

    edges = []
    for u, v in G.edges():
        edges.append([NODE_TO_IDX[u], NODE_TO_IDX[v]])
        edges.append([NODE_TO_IDX[v], NODE_TO_IDX[u]])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return x, edge_index

def run_ai_inference(model, G, start_node, end_node):
    x, edge_index = prepare_features_for_ai(G, start_node, end_node)
    
    t_start = time.perf_counter()
    start_idx = NODE_TO_IDX[start_node]
    end_idx = NODE_TO_IDX[end_node]
    path_indices = [start_idx]
    
    with torch.no_grad():
        node_emb = model.encoder(x, edge_index)
        input_emb = node_emb[start_idx].unsqueeze(0).unsqueeze(0)
        hidden = None
        curr = start_idx
        
        for _ in range(50):
            out, hidden = model.decoder.lstm(input_emb, hidden)
            logits = model.decoder.fc_out(out.squeeze(1))
            
            # --- INDEX HATASINI ÇÖZEN BÖLÜM ---
            curr_node_name = SORTED_NODES[curr]
            neighbors = list(G.neighbors(curr_node_name))
            
            # Komşuların indexlerini al, ancak modelin boyutunu aşanları ele
            model_vocab_size = logits.size(1) # Örn: 40
            valid_neighbor_indices = []
            
            for n in neighbors:
                idx = NODE_TO_IDX[n]
                # Eğer haritadaki şehir indeksi, modelin bildiği şehirlerden büyükse onu yok say
                if idx < model_vocab_size:
                    valid_neighbor_indices.append(idx)
            
            # Gidecek geçerli bir komşu yoksa dur
            if not valid_neighbor_indices: break

            full_mask = torch.ones_like(logits) * -float('inf')
            valid_indices_tensor = torch.tensor(valid_neighbor_indices, dtype=torch.long)
            
            # Güvenli atama
            if valid_indices_tensor.max() < logits.size(1):
                full_mask[0, valid_indices_tensor] = logits[0, valid_indices_tensor]
                pred_idx = full_mask.argmax(dim=-1).item()
                
                if pred_idx == curr: break 
                
                path_indices.append(pred_idx)
                curr = pred_idx
                input_emb = node_emb[pred_idx].unsqueeze(0).unsqueeze(0)
                
                if curr == end_idx: break
            else:
                break
            
    t_end = time.perf_counter()
    return (t_end - t_start) * 1000, [SORTED_NODES[i] for i in path_indices]

# --- 5. ALGORİTMALAR ---
def dijkstra_algo(graph, start, goal):
    queue = [(0, start, [])]
    visited = set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited: continue
        visited.add(node)
        path = path + [node]
        if node == goal: return cost, path
        for neighbor, attr in graph[node].items():
            if neighbor not in visited:
                heapq.heappush(queue, (cost + attr['weight'], neighbor, path))
    return 0, []

def bellman_ford_algo(graph, start, goal):
    try:
        path = nx.bellman_ford_path(graph, start, goal, weight='weight')
        cost = nx.bellman_ford_path_length(graph, start, goal, weight='weight')
        return cost, path
    except:
        return 0, []

def a_star_algo(graph, start, goal):
    try:
        path = nx.astar_path(graph, start, goal, weight='weight')
        cost = nx.shortest_path_length(graph, start, goal, weight='weight')
        return cost, path
    except:
        return 0, []

# --- 6. SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/tr/6/62/Gazi_%C3%9Cniversitesi_Logosu.png", width=100)
    st.title("Türkiye Rota Analizi")
    st.markdown("---")
    
    st.subheader("🛠️ Harita Ayarları")
    min_w = st.number_input("Min Yol Maliyeti", 1, 50, 1)
    max_w = st.number_input("Max Yol Maliyeti", 1, 50, 20)
    
    if st.button("🔄 Trafiği (Ağırlıkları) Yenile"):
        st.session_state['G'], st.session_state['pos'] = create_randomized_graph(min_w, max_w)
        st.success("Yeni trafik durumu oluşturuldu!")

    st.markdown("---")
    col1, col2 = st.columns(2)
    start_city = col1.selectbox("Başlangıç", SORTED_NODES, index=34)
    end_city = col2.selectbox("Hedef", SORTED_NODES, index=6)
    
    if st.button("🚀 Hesapla"):
        st.session_state['run'] = True

if 'G' not in st.session_state:
    st.session_state['G'], st.session_state['pos'] = create_randomized_graph(1, 20)

G = st.session_state['G']
pos = st.session_state['pos']

# --- 7. ANA EKRAN ---
if 'run' in st.session_state and st.session_state['run']:
    ai_model = load_ai_model()
    results = []
    
    # 1. Dijkstra
    t1 = time.perf_counter()
    d_cost, d_path = dijkstra_algo(G, start_city, end_city)
    d_time = (time.perf_counter() - t1) * 1000
    results.append({"Algoritma": "Dijkstra", "Süre (ms)": d_time, "Maliyet": d_cost, "Yol": d_path})
    
    # 2. A*
    t1 = time.perf_counter()
    a_cost, a_path = a_star_algo(G, start_city, end_city)
    a_time = (time.perf_counter() - t1) * 1000
    results.append({"Algoritma": "A*", "Süre (ms)": a_time, "Maliyet": a_cost, "Yol": a_path})
    
    # 3. Bellman-Ford
    t1 = time.perf_counter()
    b_cost, b_path = bellman_ford_algo(G, start_city, end_city)
    b_time = (time.perf_counter() - t1) * 1000
    results.append({"Algoritma": "Bellman-Ford", "Süre (ms)": b_time, "Maliyet": b_cost, "Yol": b_path})
    
    # 4. Yapay Zeka
    if ai_model:
        ai_time, ai_path = run_ai_inference(ai_model, G, start_city, end_city)
        ai_cost = 0
        if len(ai_path) > 1:
            for i in range(len(ai_path)-1):
                if G.has_edge(ai_path[i], ai_path[i+1]):
                    ai_cost += G[ai_path[i]][ai_path[i+1]]['weight']
        results.append({"Algoritma": "Yapay Zeka (GNN)", "Süre (ms)": ai_time, "Maliyet": ai_cost, "Yol": ai_path})
    
    df_res = pd.DataFrame(results)

    st.subheader(f"🗺️ Rota Analizi: {start_city} ➝ {end_city}")
    
    with st.container():
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        plt.figure(figsize=(14, 8))
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor(COLOR_BG_LIGHT)
        ax.set_facecolor(COLOR_BG_LIGHT)
        
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color="#B0BEC5", ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="#CFD8DC", width=1, ax=ax)
        
        path_width = 4
        
        if d_path:
            edges = list(zip(d_path, d_path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=COLOR_NODE_BRIGHT, width=path_width+2, label="Dijkstra", ax=ax)
            
        ai_res = next((r for r in results if r["Algoritma"] == "Yapay Zeka (GNN)"), None)
        if ai_res and ai_res["Yol"]:
            edges = list(zip(ai_res["Yol"], ai_res["Yol"][1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=COLOR_AI_CYAN, width=path_width, style='solid', label="Yapay Zeka", ax=ax)

        nx.draw_networkx_nodes(G, pos, nodelist=[start_city], node_color="green", node_size=300, ax=ax, label="Başlangıç")
        nx.draw_networkx_nodes(G, pos, nodelist=[end_city], node_color="red", node_size=300, ax=ax, label="Hedef")
        
        ax.legend(loc='upper left', frameon=True)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📊 Performans Tablosu")
        st.dataframe(
            df_res[["Algoritma", "Süre (ms)", "Maliyet", "Yol"]].style.format({"Süre (ms)": "{:.4f}"}),
            use_container_width=True
        )
    with col2:
        st.markdown("### ⏱️ Hız Karşılaştırması")
        chart = alt.Chart(df_res).mark_bar().encode(
            x=alt.X('Süre (ms)', title='Hesaplama Süresi (ms)'),
            y=alt.Y('Algoritma', sort='-x'),
            color=alt.Color('Algoritma', scale=alt.Scale(
                domain=['Dijkstra', 'A*', 'Bellman-Ford', 'Yapay Zeka (GNN)'],
                range=[COLOR_NODE_BRIGHT, '#F39C12', '#9B59B6', COLOR_AI_CYAN]
            )),
            tooltip=['Algoritma', 'Süre (ms)', 'Maliyet']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

else:
    st.info("👈 Sol menüden başlangıç/bitiş seçip 'Hesapla' butonuna basın.")
