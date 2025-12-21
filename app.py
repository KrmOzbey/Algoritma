import streamlit as st
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# -----------------------------------------------------------------------------
# 1. MODEL SINIFLARI (Aynı kalmalı)
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
        lstm_hidden_dim = 512 
        self.encoder = Encoder(in_channels, hidden_channels, num_nodes)
        self.decoder = Decoder(num_nodes, lstm_hidden_dim, num_nodes)

# -----------------------------------------------------------------------------
# 2. YARDIMCI VE KRİTİK DÜZELTME FONKSİYONLARI
# -----------------------------------------------------------------------------

@st.cache_resource
def load_model(model_path):
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        weight_shape = state_dict['decoder.fc_out.weight'].shape
        num_nodes_trained = weight_shape[0]
        
        hidden_channels = 256
        in_channels = 6
        out_channels = num_nodes_trained
        max_path_len = 100
        
        model = GNNPathModel(in_channels, hidden_channels, out_channels, num_nodes_trained, max_path_len)
        model.load_state_dict(state_dict)
        model.eval()
        return model, num_nodes_trained
    except Exception as e:
        st.error(f"Model yüklenirken hata: {e}")
        return None, 0

def get_graph_features(G):
    # Düğüm featurelarını hesapla
    # G küçükse veya büyükse fark etmez, önce ham değerleri alalım
    degree = np.array([val for (node, val) in G.degree()])
    try:
        centrality = np.array([val for (node, val) in nx.betweenness_centrality(G).items()])
        clustering = np.array([val for (node, val) in nx.clustering(G).items()])
        pagerank = np.array([val for (node, val) in nx.pagerank(G).items()])
    except:
        # Hata durumunda (örn: graph bağlantısızsa) default değerler
        nodes_len = len(G.nodes())
        centrality = np.zeros(nodes_len)
        clustering = np.zeros(nodes_len)
        pagerank = np.zeros(nodes_len)

    degree = degree.reshape(-1, 1)
    centrality = centrality.reshape(-1, 1)
    clustering = clustering.reshape(-1, 1)
    pagerank = pagerank.reshape(-1, 1)

    base_features = np.concatenate([degree, centrality, clustering, pagerank], axis=1)
    return torch.tensor(base_features, dtype=torch.float)

def run_ai_inference_strict(model, G, start_node, end_node, num_nodes_trained):
    """
    Bu fonksiyon modelin SADECE geçerli komşulara gitmesini zorlar.
    """
    # 1. Grafı Tensor'a çevir
    adj = nx.to_numpy_array(G)
    edge_index = []
    
    # NetworkX grafındaki node sayısını al
    current_num_nodes = len(G.nodes())

    for i in range(current_num_nodes):
        for j in range(current_num_nodes):
            if adj[i][j] != 0:
                edge_index.append([i, j])
    
    if not edge_index: # Eğer hiç kenar yoksa
        return [start_node]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # 2. Featureları Hazırla ve Padding Yap
    base_features = get_graph_features(G)
    
    # Model sabit boyutta feature bekler (num_nodes_trained).
    # Eğer şu anki graf küçükse, feature matrisini 0 ile doldur (padding).
    if current_num_nodes < num_nodes_trained:
        pad_size = num_nodes_trained - current_num_nodes
        padding = torch.zeros(pad_size, 4) # 4 temel feature
        # DİKKAT: Featureları pad ediyoruz
        base_features = torch.cat([base_features, padding], dim=0)
    
    # 3. Maskları Hazırla
    start_mask = torch.zeros(num_nodes_trained, 1)
    end_mask = torch.zeros(num_nodes_trained, 1)
    
    # Eğer seçilen node indexi model boyutundan büyükse hata vermemesi için kontrol (genelde olmaz ama tedbir)
    if start_node < num_nodes_trained: start_mask[start_node] = 1
    if end_node < num_nodes_trained: end_mask[end_node] = 1
    
    x = torch.cat([base_features, start_mask, end_mask], dim=1)
    
    # 4. Encoder Çalıştır
    # edge_index'i de modele vermeden önce kontrol etmeliyiz ama 
    # model yapısı gereği edge_index sadece node embedding için kullanılır.
    # Modelin fc katmanı feature boyutuna bağlıdır. 
    # Burada GCNConv kullanıldığı için edge_index'in boyutu dinamiktir, sorun çıkarmaz.
    
    with torch.no_grad():
        node_emb = model.encoder(x, edge_index)
        
        # LSTM Başlangıç
        input_emb = node_emb[start_node].unsqueeze(0).unsqueeze(0)
        hidden = None
        
        path = [start_node]
        visited = set([start_node])
        curr_idx = start_node
        
        # Maksimum adım sayısı (infinite loop koruması)
        max_steps = current_num_nodes * 2 
        
        for _ in range(max_steps):
            out, hidden = model.decoder.lstm(input_emb, hidden)
            logits = model.decoder.fc_out(out.squeeze(1))
            
            # --- KRİTİK BÖLÜM: MASKING ---
            # Modelin tüm çıktıları arasından SADECE şu anki düğümün komşularını seçmesine izin ver.
            
            # 1. Mevcut graf üzerindeki komşuları bul
            neighbors = list(G.neighbors(curr_idx))
            
            # 2. Ziyaret edilmemiş komşuları belirle
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            # 3. Eğer hedef düğüm komşular arasındaysa, direkt oraya git (Greedy finish)
            if end_node in neighbors:
                path.append(end_node)
                break
                
            # 4. Gidilecek yer kalmadıysa (Dead end)
            valid_candidates = unvisited_neighbors if unvisited_neighbors else neighbors # Ziyaret edilmemiş yoksa, geri dönmeye izin ver
            
            if not valid_candidates:
                break # Çıkmaz sokak
            
            # 5. Logits maskeleme
            # Tüm değerleri -sonsuz yap
            masked_logits = torch.full_like(logits, -float('inf'))
            
            # Sadece geçerli adayların indekslerini orijinal logit değerleriyle doldur
            # DİKKAT: Modelin output boyutu (81) ile mevcut graf boyutu (örn 20) farklı olabilir.
            # Sadece modelin tanıdığı indeks aralığındakileri alabiliriz.
            safe_candidates = [c for c in valid_candidates if c < num_nodes_trained]
            
            if not safe_candidates:
                break
                
            masked_logits[0, safe_candidates] = logits[0, safe_candidates]
            
            # 6. En yüksek olasılıklı komşuyu seç
            pred_node = masked_logits.argmax(dim=-1).item()
            
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

st.set_page_config(page_title="AI Pathfinding", layout="wide")
st.title("🤖 AI vs Algoritmalar: Yol Bulma Simülasyonu")

# Sidebar
st.sidebar.header("Ayarlar")
model_path = "Model3_3.pt"
model, max_trained_nodes = load_model(model_path)

if not model:
    st.error("Model dosyası bulunamadı.")
    st.stop()

st.sidebar.info(f"Yüklü Model Kapasitesi: {max_trained_nodes} Node")

# Harita Ayarları
# Kullanıcı modelin kapasitesinden fazla node seçerse hata alır, o yüzden max değeri sınırlıyoruz.
num_nodes = st.sidebar.slider("Düğüm Sayısı", 5, max_trained_nodes, 15)
edge_prob = st.sidebar.slider("Bağlantı Sıklığı", 0.1, 1.0, 0.25)
seed = st.sidebar.number_input("Rastgelelik Tohumu (Seed)", 1, 1000, 42)

if st.sidebar.button("Harita Oluştur / Yenile"):
    # Rastgele Graf
    G = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob, seed=seed)
    
    # İzole düğümleri bağla (Graph connected olsun)
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components)-1):
            # Her bileşenden bir düğümü diğerine bağla
            u = list(components[i])[0]
            v = list(components[i+1])[0]
            G.add_edge(u, v)

    # Ağırlık ata
    np.random.seed(seed)
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = np.random.randint(1, 20)
        
    pos = nx.spring_layout(G, seed=seed)
    st.session_state['G'] = G
    st.session_state['pos'] = pos
    st.session_state['map_ready'] = True

if 'map_ready' in st.session_state:
    G = st.session_state['G']
    pos = st.session_state['pos']
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Rota Belirle")
        nodes = list(G.nodes())
        start_node = st.selectbox("Başlangıç", nodes, index=0)
        end_node = st.selectbox("Bitiş", nodes, index=len(nodes)-1)
        
        if st.button("Başlat"):
            results = []
            
            # --- 1. Klasik Algoritmalar ---
            try:
                path = nx.dijkstra_path(G, start_node, end_node, weight='weight')
                dist = nx.dijkstra_path_length(G, start_node, end_node, weight='weight')
                results.append(("Dijkstra (Optimal)", path, dist, 'red', 'solid'))
            except:
                results.append(("Dijkstra", [], float('inf'), 'red', 'solid'))

            try:
                path = nx.astar_path(G, start_node, end_node, weight='weight')
                dist = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                results.append(("A*", path, dist, 'blue', 'dashed'))
            except: pass

            try:
                path = nx.bellman_ford_path(G, start_node, end_node, weight='weight')
                dist = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                results.append(("Bellman-Ford", path, dist, 'purple', 'dotted'))
            except: pass

            # --- 2. AI Model ---
            try:
                ai_path = run_ai_inference_strict(model, G, start_node, end_node, max_trained_nodes)
                
                # AI yol maliyeti hesapla
                ai_dist = 0
                is_valid = True
                if len(ai_path) < 2 or ai_path[-1] != end_node:
                    is_valid = False
                
                for u, v in zip(ai_path[:-1], ai_path[1:]):
                    if G.has_edge(u, v):
                        ai_dist += G[u][v]['weight']
                    else:
                        is_valid = False
                        ai_dist = float('inf')
                
                label = "Yapay Zeka"
                if not is_valid: label += " (Hedefe Ulaşamadı)"
                
                results.append((label, ai_path, ai_dist, 'green', 'dashdot'))
                
            except Exception as e:
                st.error(f"AI Hatası: {e}")

            # --- Görselleştirme ---
            fig, ax = plt.subplots(figsize=(10, 6))
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightgray', edge_color='#cccccc', node_size=600)
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

            st.write("### Sonuçlar Tablosu")
            
            cols = st.columns(len(results))
            for idx, (name, path, dist, color, style) in enumerate(results):
                # Tablo
                val_str = f"{dist}" if dist != float('inf') else "Başarısız"
                cols[idx].metric(name, val_str, f"{len(path)-1} Adım")
                
                # Çizim
                if len(path) > 1:
                    edges = list(zip(path[:-1], path[1:]))
                    # Çakışmayı önlemek için her çizgiyi biraz kaydır (offset) veya kalınlığı değiştir
                    width = 6 - (idx * 1.5)
                    alpha = 0.8 - (idx * 0.1)
                    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, width=width, style=style, alpha=alpha, label=name)
            
            # Legend
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color=r[3], lw=2, linestyle=r[4]) for r in results]
            ax.legend(custom_lines, [r[0] for r in results], loc='upper left')
            
            st.pyplot(fig)

    with col2:
        # Harita önizleme (Boş halini göstermek için)
        if 'map_ready' in st.session_state and not st.button("Sonuçları Temizle", key="clean"):
            pass
