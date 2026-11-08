"""Arquitecturas de aprendizaje profundo (PyTorch) y su entrenamiento con parada temprana."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from .config import SEMILLA

DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


def fijar_semilla(semilla):
    """Fija las semillas de NumPy y PyTorch (CPU y GPU)."""
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PreparadorRed:
    """Convierte la matriz del Codificador en tensores: categóricas (enteros) y numéricas (estandarizadas).
    El escalador se ajusta SOLO con entrenamiento."""

    def __init__(self, codificador):
        self.cod = codificador

    def fit(self, M):
        self.escalador = StandardScaler().fit(M[:, self.cod.idx_num])
        return self

    def tensores(self, M):
        cat = torch.tensor(M[:, self.cod.idx_cat] + 1, dtype=torch.long)   # -1 (desconocida) pasa a 0
        num = torch.tensor(self.escalador.transform(M[:, self.cod.idx_num]), dtype=torch.float32)
        return cat, num


def bloque_denso(entrada, capas, dropout):
    """Capas ocultas densas con ReLU y dropout, más una neurona de salida (logit)."""
    partes, n = [], entrada
    for h in capas:
        partes += [nn.Linear(n, h), nn.ReLU(), nn.Dropout(dropout)]
        n = h
    partes.append(nn.Linear(n, 1))
    return nn.Sequential(*partes)


class MLP(nn.Module):
    """DL1 – Perceptrón multicapa: one-hot de cada categórica + numéricas."""

    def __init__(self, cardinalidad, n_num, capas=(128, 64), dropout=0.3):
        super().__init__()
        self.tamanos = [c + 1 for c in cardinalidad]
        self.red = bloque_denso(sum(self.tamanos) + n_num, capas, dropout)

    def forward(self, cat, num):
        oh = [nn.functional.one_hot(cat[:, i], t).float() for i, t in enumerate(self.tamanos)]
        return self.red(torch.cat(oh + [num], dim=1)).squeeze(1)


class MLPEmbeddings(nn.Module):
    """DL2 – MLP con embeddings: cada categórica se representa con un vector denso aprendido."""

    def __init__(self, cardinalidad, n_num, capas=(128, 64), dropout=0.3, emb_max=16):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c + 1, min(emb_max, (c + 1) // 2 + 1)) for c in cardinalidad])
        self.red = bloque_denso(sum(e.embedding_dim for e in self.emb) + n_num, capas, dropout)

    def forward(self, cat, num):
        e = [emb(cat[:, i]) for i, emb in enumerate(self.emb)]
        return self.red(torch.cat(e + [num], dim=1)).squeeze(1)


class TransformerTabular(nn.Module):
    """DL3 – Transformer tabular (tipo FT-Transformer): cada variable es un token; token [CLS] para clasificar."""

    def __init__(self, cardinalidad, n_num, d=32, capas=2, cabezas=4, dropout=0.2):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c + 1, d) for c in cardinalidad])
        self.w_num = nn.Parameter(torch.randn(n_num, d) * 0.1)
        self.b_num = nn.Parameter(torch.zeros(n_num, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        capa = nn.TransformerEncoderLayer(d_model=d, nhead=cabezas, dim_feedforward=2 * d,
                                          dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(capa, num_layers=capas, enable_nested_tensor=False)
        self.salida = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, cat, num):
        tok_cat = torch.stack([emb(cat[:, i]) for i, emb in enumerate(self.emb)], dim=1)
        tok_num = num.unsqueeze(2) * self.w_num + self.b_num
        cls = self.cls.expand(cat.shape[0], -1, -1)
        h = self.encoder(torch.cat([cls, tok_cat, tok_num], dim=1))
        return self.salida(h[:, 0]).squeeze(1)


ARQUITECTURAS = {"MLP": MLP, "MLP con embeddings": MLPEmbeddings, "Transformer tabular": TransformerTabular}


def perdida_bce(y, p, pos_weight=1.0):
    """Entropía cruzada binaria (con el mismo peso de positivos que en entrenamiento)."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(pos_weight * y * np.log(p) + (1 - y) * np.log(1 - p)))


def predecir(modelo, cat, num, lote=4096):
    """Probabilidades (0 a 1) para los brotes entregados."""
    modelo.eval()
    salidas = []
    with torch.no_grad():
        for i in range(0, len(cat), lote):
            logit = modelo(cat[i:i + lote].to(DISPOSITIVO), num[i:i + lote].to(DISPOSITIVO))
            salidas.append(torch.sigmoid(logit).cpu())
    return torch.cat(salidas).numpy()


def entrenar_red(nombre, preparador, M, y, M_va=None, y_va=None, ponderar=False, arq=None,
                 epocas=60, paciencia=8, lote=256, lr=1e-3, weight_decay=1e-4, semilla=SEMILLA):
    """Entrena una red.

    nombre: clave de ARQUITECTURAS. arq: diccionario con hiperparámetros de la arquitectura
    (p. ej. {"capas": (128, 64), "dropout": 0.3}).
    Con M_va/y_va: parada temprana según el AUC-PR de validación y se devuelve la mejor época.
    Sin M_va: entrena un número fijo de épocas (reentrenamiento final).
    Devuelve (modelo, mejor_epoca, historial) con la pérdida de entrenamiento y de validación por época.
    """
    fijar_semilla(semilla)
    arq = arq or {}
    modelo = ARQUITECTURAS[nombre](preparador.cod.cardinalidad, len(preparador.cod.idx_num), **arq).to(DISPOSITIVO)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    pw = torch.tensor((y == 0).sum() / (y == 1).sum() if ponderar else 1.0, dtype=torch.float32, device=DISPOSITIVO)
    perdida = nn.BCEWithLogitsLoss(pos_weight=pw)

    cat, num = preparador.tensores(M)
    yt = torch.tensor(y, dtype=torch.float32)
    validar = M_va is not None
    if validar:
        cat_va, num_va = preparador.tensores(M_va)

    hist = {"perdida_tr": [], "perdida_va": [], "auc_pr_va": []}
    mejor_ap, mejor_ep, mejor_estado, sin_mejora = -1.0, epocas, None, 0
    gen = torch.Generator().manual_seed(semilla)                   # orden de los lotes reproducible

    for ep in range(epocas):
        modelo.train()
        orden = torch.randperm(len(yt), generator=gen)
        suma = 0.0
        for i in range(0, len(yt), lote):
            b = orden[i:i + lote]
            opt.zero_grad()
            l = perdida(modelo(cat[b].to(DISPOSITIVO), num[b].to(DISPOSITIVO)), yt[b].to(DISPOSITIVO))
            l.backward()
            opt.step()
            suma += l.item() * len(b)
        hist["perdida_tr"].append(suma / len(yt))
        if not validar:
            continue
        p_va = predecir(modelo, cat_va, num_va)
        hist["perdida_va"].append(perdida_bce(y_va, p_va, float(pw)))   # misma pérdida que en entrenamiento
        ap = average_precision_score(y_va, p_va)
        hist["auc_pr_va"].append(ap)
        if ap > mejor_ap + 1e-4:
            mejor_ap, mejor_ep, sin_mejora = ap, ep + 1, 0
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    if validar:
        modelo.load_state_dict(mejor_estado)
    return modelo, mejor_ep, hist


def contar_parametros(modelo):
    """Número de pesos entrenables de una red."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
