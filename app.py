import streamlit as st
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# -----------------------------------------------------------------------------
# 1. MODEL SINIFLARI (Eğitim kodunuzla birebir aynı olmalı)
# -----------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_p=0.2):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(dropout_p)
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
        # LSTM Hidden Dim sabit olarak eğitim kodundan alındı
        lstm_hidden_dim = 512 
        self.encoder = Encoder(in_channels, hidden_channels, num_nodes)
        self.decoder = Decoder(num_nodes, lstm_hidden_dim, num_nodes)

# -----------------------------------------------------------------------------
# 2. YARDIMCI FONKSİYONLAR
# -----------------------------------------------------------------------------

@st.cache_resource
def load_model(model_path):
    try:
        # Önce state_dict yükleyip boyutları analiz edelim
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Modelin eğitildiği output layer boyutunu bul (num_nodes)
        weight_shape = state_dict['decoder.fc_out.weight'].shape
        num_nodes_trained = weight_shape[0]  # Output dim (e.g., 82)
        
        # Hyperparametreler (Eğitim kodunuzdaki varsayılanlar)
        hidden_channels = 256
        in_channels = 6  # 4 features + 2 masks
        out_channels = num_nodes_trained
        max_path_len = 100 # Sembolik, mimariyi etkilemez
        
        model = GNNPathModel(in_channels, hidden_channels, out_channels, num_nodes_trained, max_path_len)
        model.load_state_dict(state_dict)
        model.eval()
        return model, num_nodes_trained
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None, 0

def get_graph_features(G, num_nodes_trained):
    """
    Eğitim verisindeki feature extraction mantığını uygular.
    """
    # Gerekirse düğüm sayısını eşitlemek için padding yapılabilir ama
    # burada sadece mevcut grafın özelliklerini alacağız.
    
    # Featurelar: Degree, Centrality, Clustering, PageRank
    degree = np.array([val for (node, val) in G.degree()])
    centrality = np.array([val for (node, val) in nx.betweenness_centrality(G).items()])
    clustering = np.array([val for (node, val) in nx.clustering(G).items()])
    pagerank = np.array([val for (node, val) in nx.pagerank(G).items()])

    # Reshape
    degree = degree.reshape(-1, 1)
    centrality = centrality.reshape(-1, 1)
    clustering = clustering.reshape(-1, 1)
    pagerank = pagerank.reshape(-1, 1)

    base_features = np.concatenate([degree, centrality, clustering, pagerank], axis=1)
    return torch.tensor(base_features, dtype=torch.float)

def safe_mask_logits(logits, allowed_indices):
    all_indices = set(range(logits.shape[-1]))
    mask_indices = torch.tensor(list(all_indices - set(allowed_indices)), dtype=torch.long)
    logits = logits.clone()
    if logits.dim() == 2:
        logits[0, mask_indices] = -float('inf')
    else:
        logits[mask_indices] = -float('inf')
    return logits

def get_neighbors(edge_index, num_nodes):
    neighbors = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        src, dst = edge_index[:, i]
        neighbors[src.item()].append(dst.item())
        neighbors[dst.item()].append(src.item()) # Undirected
    return neighbors

def run_ai_inference(model, G, start_node, end_node, num_nodes_trained):
    """
    Eğitilmiş modeli kullanarak yol tahmini yapar.
    """
    # 1. Grafı PyG formatına çevir
    adj = nx.to_numpy_array(G)
    edge_index = []
    edge_attr = [] # Ağırlıklar (model kullanıyorsa)
    
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i][j] != 0:
                edge_index.append([i, j])
                edge_attr.append([adj[i][j]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # 2. Featureları hazırla
    base_features = get_graph_features(G, num_nodes_trained)
    
    # Eğer oluşturulan graf modelin eğitim grafından küçükse, featureları pad etmeliyiz
    # Çünkü Linear katmanlar sabit boyut bekler.
    current_nodes = base_features.shape[0]
    if current_nodes < num_nodes_trained:
        pad_size = num_nodes_trained - current_nodes
        padding = torch.zeros(pad_size, 4)
        base_features = torch.cat([base_features, padding], dim=0)
    
    # Masklar
    start_mask = torch.zeros(num_nodes_trained, 1)
    end_mask = torch.zeros(num_nodes_trained, 1)
    start_mask[start_node] = 1
    end_mask[end_node] = 1
    
    x = torch.cat([base_features, start_mask, end_mask], dim=1)
    
    # 3. Model Tahmini
    path = [start_node]
    visited = set([start_node])
    
    with torch.no_grad():
        node_emb = model.encoder(x, edge_index, edge_attr) # edge_attr opsiyonel, modelde varsa kullanılır
        neighbors = get_neighbors(edge_index, num_nodes_trained)
        
        input_emb = node_emb[start_node].unsqueeze(0).unsqueeze(0)
        hidden = None
        curr_idx = start_node
        
        for _ in range(num_nodes_trained): # Max adım sayısı
            out, hidden = model.decoder.lstm(input_emb, hidden)
            logits = model.decoder.fc_out(out.squeeze(1))
            
            # Masking (Gidilebilecek komşular)
            # Dikkat: Rastgele grafın komşuları, eğitim grafının indekslerinden farklıdır.
            # Ancak model topolojiyi GCN ile öğrendiği için node_emb üzerinden karar verir.
            
            current_neighbors = []
            if curr_idx < len(neighbors):
                current_neighbors = neighbors[curr_idx]
            
            allowed = set(current_neighbors) - visited
            
            # Eğer hedef komşudaysa oraya gitmeye zorla/izin ver
            if end_node in current_neighbors:
                allowed.add(end_node)
            
            if not allowed:
                break
                
            logits = safe_mask_logits(logits, allowed)
            pred_node = logits.argmax(dim=-1).item()
            
            path.append(pred_node)
            visited.add(pred_node)
            
            if pred_node == end_node:
                break
            
            curr_idx = pred_node
            input_emb = node_emb[curr_idx].unsqueeze(0).unsqueeze(0)
            
    return path

# -----------------------------------------------------------------------------
# 3. STREAMLIT ARAYÜZÜ
# -----------------------------------------------------------------------------

st.set_page_config(page_title="AI vs Algorithms: Pathfinding", layout="wide")
st.title("🗺️ AI Destekli Yol Bulma Algoritmaları Karşılaştırması")

# Sidebar - Ayarlar
st.sidebar.header("Harita Ayarları")

# Model Yükleme
model_path = "Model3_2.pt" # Github reponuzda bu dosya aynı dizinde olmalı
model, max_trained_nodes = load_model(model_path)

if model:
    st.sidebar.success(f"Model Yüklendi! (Maksimum Node Kapasitesi: {max_trained_nodes})")
else:
    st.sidebar.error("Model dosyası (Model3_2.pt) bulunamadı.")
    st.stop()

# Harita Parametreleri
num_nodes = st.sidebar.slider("Düğüm Sayısı (Node Count)", min_value=5, max_value=max_trained_nodes, value=20)
edge_prob = st.sidebar.slider("Bağlantı Olasılığı (Edge Probability)", 0.1, 1.0, 0.3)
min_weight = st.sidebar.number_input("Min Edge Ağırlığı", 1, 10, 1)
max_weight = st.sidebar.number_input("Max Edge Ağırlığı", 10, 100, 20)

if st.sidebar.button("Yeni Harita Oluştur"):
    # Rastgele Graf Oluşturma
    # Connected olması için döngü
    connected = False
    while not connected:
        G = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob, seed=None)
        if nx.is_connected(G):
            connected = True
    
    # Ağırlık atama
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = np.random.randint(min_weight, max_weight + 1)
    
    # Layout belirle ve kaydet (Görsel tutarlılık için)
    pos = nx.spring_layout(G, seed=42)
    st.session_state['G'] = G
    st.session_state['pos'] = pos
    st.session_state['map_generated'] = True

# Harita varsa işlem yap
if 'map_generated' in st.session_state and st.session_state['map_generated']:
    G = st.session_state['G']
    pos = st.session_state['pos']
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Yol Seçimi")
        nodes = list(G.nodes())
        start_node = st.selectbox("Başlangıç (Start)", nodes, index=0)
        end_node = st.selectbox("Bitiş (End)", nodes, index=len(nodes)-1)
        
        run_btn = st.button("Algoritmaları Çalıştır")

    with col2:
        # Haritayı Çiz (Temiz Hal)
        fig, ax = plt.subplots(figsize=(10, 6))
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightgray', edge_color='gray', node_size=500)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        if run_btn:
            results = []
            
            # 1. Dijkstra
            try:
                dijkstra_path = nx.dijkstra_path(G, start_node, end_node, weight='weight')
                dijkstra_len = nx.dijkstra_path_length(G, start_node, end_node, weight='weight')
                results.append(("Dijkstra", dijkstra_path, dijkstra_len, 'red', 'solid'))
            except nx.NetworkXNoPath:
                st.warning("Dijkstra yol bulamadı.")

            # 2. A* (Heuristic = 0, Dijkstra gibi davranır ama yapı olarak A*)
            try:
                astar_path = nx.astar_path(G, start_node, end_node, weight='weight')
                # A* için maliyet hesapla
                astar_len = sum(G[u][v]['weight'] for u, v in zip(astar_path[:-1], astar_path[1:]))
                results.append(("A*", astar_path, astar_len, 'blue', 'dashed'))
            except nx.NetworkXNoPath:
                pass

            # 3. Bellman-Ford
            try:
                bellman_path = nx.bellman_ford_path(G, start_node, end_node, weight='weight')
                bellman_len = sum(G[u][v]['weight'] for u, v in zip(bellman_path[:-1], bellman_path[1:]))
                results.append(("Bellman-Ford", bellman_path, bellman_len, 'purple', 'dotted'))
            except nx.NetworkXNoPath:
                pass

            # 4. AI Model
            try:
                ai_path = run_ai_inference(model, G, start_node, end_node, max_trained_nodes)
                # AI yol maliyeti (Eğer geçerli bir yol ise)
                ai_len = 0
                valid_ai_path = True
                for u, v in zip(ai_path[:-1], ai_path[1:]):
                    if G.has_edge(u, v):
                        ai_len += G[u][v]['weight']
                    else:
                        valid_ai_path = False
                        ai_len = float('inf')
                
                label = "AI Model" + (" (Geçersiz Yol)" if not valid_ai_path else "")
                results.append((label, ai_path, ai_len, 'green', 'dashdot'))
            except Exception as e:
                st.error(f"AI Model hatası: {e}")

            # Sonuçları Görselleştir
            offset = 0
            st.write("### Sonuçlar")
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            
            cols = [res_col1, res_col2, res_col3, res_col4]
            
            for idx, (name, path, length, color, style) in enumerate(results):
                # Metrikleri yaz
                cols[idx].metric(label=name, value=f"{length}", delta=f"Adım: {len(path)}")
                
                # Yolu çiz
                path_edges = list(zip(path[:-1], path[1:]))
                # Çizgileri üst üste binmemesi için hafif kaydırarak (width ve alpha ile) çiziyoruz
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=color, width=4-(idx*0.5), style=style, label=name)
            
            plt.title(f"Yol Karşılaştırması: {start_node} -> {end_node}")
            
            # Legend oluşturma (Manuel handle ile)
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color=r[3], lw=2, linestyle=r[4]) for r in results]
            ax.legend(custom_lines, [r[0] for r in results])
            
            st.pyplot(fig)
            
            # Detaylı Yol Listesi
            with st.expander("Detaylı Yol Listesi"):
                for name, path, length, _, _ in results:
                    st.write(f"**{name}:** {path} (Maliyet: {length})")

        else:
            st.pyplot(fig)

else:
    st.info("Lütfen sol menüden 'Yeni Harita Oluştur' butonuna basın.")
