"""
NyayaSetu (DCP-LRN) Contract Inference Engine
Takes any custom contract and scenario/dispute, retrieves relevant clauses,
classifies stance (Entailment / Contradiction), and extracts deontic modalities.
"""
import sys
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CHECKPOINT_PATH = Path("/Users/sarah/.gemini/antigravity/scratch/nyayasetu_contractnli/best_dcp_lrn_full.pt")

AUX_TASKS = [
    'negation', 'modality', 'obligation', 'permission',
    'prohibition', 'conditionality', 'exception', 'temporal_scope'
]

CUE_PATTERNS = {
    'negation':        r'\b(not|never|no\b|without)\b',
    'modality':        r'\b(shall|must|may|might|should|will|can|cannot|required|permitted|prohibited)\b',
    'obligation':      r'\b(shall|must|required to|obliged to|agrees to|undertakes to|is to\b)',
    'permission':      r'\b(may|permitted to|entitled to|has the right to|is allowed to)\b',
    'prohibition':     r'\b(shall not|must not|may not|prohibited from|not permitted to|cannot)\b',
    'conditionality':  r'\b(if|unless|provided that|subject to|only with|conditional|requires? the\b)',
    'exception':       r'\b(except|excluding|excludes|does not include|does not apply|nothing in|other than|public(?:ly)? (?:domain|available)|lawfully|already)\b',
    'temporal_scope':  r'\b(within|before|after|prior to|upon|during|day|days|hour|hours|month|months|termination|times|immediately)\b',
}

def cue_vector(text):
    t = text.lower()
    return [1.0 if re.search(CUE_PATTERNS[k], t) else 0.0 for k in AUX_TASKS]

# Exact Architecture matching train_full_contractnli.py
class MultiHeadBidirCrossAttention(nn.Module):
    def __init__(self, d=768, heads=4, dropout=0.1):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.scale = self.dk ** -0.5
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.norm_q = nn.LayerNorm(d); self.norm_kv = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def _attend(self, q, k, v, k_mask):
        B, Lq, d = q.shape; Lk = k.shape[1]
        sh = lambda x: x.view(x.shape[0], -1, self.h, self.dk).transpose(1, 2)
        qh, kh, vh = sh(self.q(q)), sh(self.k(k)), sh(self.v(v))
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale
        if k_mask is not None:
            mask = k_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.drop(scores.softmax(-1))
        ctx = torch.matmul(attn, vh).transpose(1, 2).contiguous().view(B, Lq, d)
        return self.out(ctx), attn

    def forward(self, Hc, Mc, Hd, Md):
        n, Lc, d = Hc.shape; Ld = Hd.shape[1]
        Hd_exp = Hd.expand(n, -1, -1); Md_exp = Md.expand(n, -1)
        Hc_att, w_c2d = self._attend(self.norm_q(Hc), self.norm_kv(Hd_exp), self.norm_kv(Hd_exp), Md_exp)
        Hd_att, w_d2c = self._attend(self.norm_q(Hd_exp), self.norm_kv(Hc), self.norm_kv(Hc), Mc)
        return Hc + self.drop(Hc_att), Hd_exp + self.drop(Hd_att), w_c2d, w_d2c

class PropositionAwareFusion(nn.Module):
    def __init__(self, d=768, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(5 * d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.prop_score = nn.Linear(d, 1)
        self.aux_head = nn.Linear(d, len(AUX_TASKS))

    def fuse(self, H, H_att, R_ctx):
        feats = torch.cat([H, H_att, H * H_att, (H - H_att).abs(), R_ctx], dim=-1)
        return self.norm(H + self.drop(self.proj(feats)))

    def propositions(self, R, mask):
        s = self.prop_score(R).squeeze(-1)
        s = s.masked_fill(mask == 0, -1e9)
        alpha = s.softmax(-1)
        p = (alpha.unsqueeze(-1) * R).sum(1)
        return p, alpha

class PropositionGating(nn.Module):
    def __init__(self, d=768, d_e=256, dropout=0.1):
        super().__init__()
        self.proj_h = nn.Linear(d, d_e); self.proj_r = nn.Linear(d, d_e); self.proj_p = nn.Linear(d, d_e)
        self.gate_mlp = nn.Sequential(nn.Linear(4 * d, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 3))
        self.norm = nn.LayerNorm(d_e)

    def forward(self, h, r, p, h_D):
        n = h.shape[0]
        hD = h_D.unsqueeze(0).expand(n, -1)
        g = self.gate_mlp(torch.cat([h, r, p, hD], dim=-1)).softmax(-1)
        z = g[:, 0:1] * self.proj_h(h) + g[:, 1:2] * self.proj_r(r) + g[:, 2:3] * self.proj_p(p)
        return self.norm(z), g

class RelevanceScorer(nn.Module):
    def __init__(self, d_e=256, d_hidden=768):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_e + d_hidden, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, z, h_D):
        n = z.shape[0]
        hD = h_D.unsqueeze(0).expand(n, -1)
        return self.mlp(torch.cat([z, hD], dim=-1)).squeeze(-1)

class StanceClassifier(nn.Module):
    def __init__(self, d_e=256, d_hidden=768):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_e + d_hidden, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 3))

    def forward(self, z, h_D):
        n = z.shape[0]
        hD = h_D.unsqueeze(0).expand(n, -1)
        return self.mlp(torch.cat([z, hD], dim=-1))

class DCPLRN(nn.Module):
    def __init__(self, d=768, d_e=256):
        super().__init__()
        self.cross = MultiHeadBidirCrossAttention(d, heads=4, dropout=0.1)
        self.fusion = PropositionAwareFusion(d, dropout=0.1)
        self.gate = PropositionGating(d, d_e, dropout=0.1)
        self.relevance = RelevanceScorer(d_e, d)
        self.stance = StanceClassifier(d_e, d)

    def _mean(self, X, mask):
        m = mask.unsqueeze(-1).to(X.dtype)
        return (X * m).sum(1) / m.sum(1).clamp(min=1e-6)

    def forward(self, Hc, Mc, Hd, Md, h_D):
        h = self._mean(Hc, Mc)
        Hc_att, Hd_att, _, _ = self.cross(Hc, Mc, Hd, Md)
        R_ctx = Hd_att.mean(1, keepdim=True).expand_as(Hc_att)
        R = self.fusion.fuse(Hc, Hc_att, R_ctx)
        p, _ = self.fusion.propositions(R, Mc)
        r = self._mean(R, Mc)
        aux_logits = self.fusion.aux_head(p)

        z, g = self.gate(h, r, p, h_D)
        z = F.normalize(z, dim=-1)

        relevance_logits = self.relevance(z, h_D)
        stance_logits = self.stance(z, h_D)

        return {
            'z': z, 'gate': g,
            'aux_logits': aux_logits,
            'relevance_logits': relevance_logits,
            'stance_logits': stance_logits
        }

class NyayaSetuInference:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        print("--> Loading Legal-BERT encoder...")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.bert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased").to(DEVICE).eval()
        for p in self.bert.parameters(): p.requires_grad = False
        
        print(f"--> Initializing and loading NyayaSetu checkpoint from {checkpoint_path}...")
        self.model = DCPLRN().to(DEVICE)
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state_dict = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("--> Model loaded and ready!")

    @torch.no_grad()
    def _encode_texts(self, texts, max_len=96, batch_size=64):
        all_H, all_M = [], []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len).to(DEVICE)
            out = self.bert(**enc)
            all_H.append(out.last_hidden_state)
            all_M.append(enc['attention_mask'])
        H = torch.cat(all_H, dim=0)
        M = torch.cat(all_M, dim=0)
        m = M.unsqueeze(-1).to(H.dtype)
        h_pooled = (H * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return H, M, h_pooled

    @torch.no_grad()
    def analyze(self, contract_clauses, dispute_scenario, top_k=5):
        if isinstance(contract_clauses, str):
            lines = [l.strip() for l in contract_clauses.split('\n') if len(l.strip()) > 15]
            if len(lines) < 3:
                lines = [s.strip() for s in re.split(r'(?<=[.?!])\s+', contract_clauses) if len(s.strip()) > 15]
            contract_clauses = lines
            
        Hc, Mc, _ = self._encode_texts(contract_clauses, max_len=96)
        Hd, Md, hd = self._encode_texts([dispute_scenario], max_len=64)
        h_D = hd[0]
        
        out = self.model(Hc, Mc, Hd, Md, h_D)
        
        rel_scores = torch.sigmoid(out['relevance_logits']).cpu().numpy()
        stance_logits = out['stance_logits'].cpu().numpy()
        stance_probs = F.softmax(out['stance_logits'], dim=-1).cpu().numpy()
        gate_weights = out['gate'].cpu().numpy()
        
        ranking = np.argsort(-rel_scores)
        
        results = []
        for rank, idx in enumerate(ranking[:top_k], 1):
            clause_text = contract_clauses[idx]
            rel_score = float(rel_scores[idx])
            probs = stance_probs[idx]
            
            # Binary normalized stance between Entailment (0) and Contradiction (1)
            binary_probs = probs[:2] / max(1e-6, probs[:2].sum())
            st_choice = int(np.argmax(binary_probs))
            st_label = "ENTAILMENT (Agrees/Allowed)" if st_choice == 0 else "CONTRADICTION (Conflicts/Prohibited)"
            confidence = float(binary_probs[st_choice])
            
            cues_found = [k for k, p in CUE_PATTERNS.items() if re.search(p, clause_text.lower())]
            
            results.append({
                'rank': rank,
                'clause_index': int(idx),
                'clause_text': clause_text,
                'relevance_score': round(rel_score, 4),
                'predicted_stance': st_label,
                'stance_confidence': round(confidence, 4),
                'deontic_cues': cues_found,
                'gate_weights': {
                    'surface_text': round(float(gate_weights[idx, 0]), 3),
                    'cross_dispute': round(float(gate_weights[idx, 1]), 3),
                    'deontic_props': round(float(gate_weights[idx, 2]), 3)
                }
            })
            
        return results

if __name__ == '__main__':
    engine = NyayaSetuInference()
    
    sample_clauses = [
        "1. Definitions: 'Confidential Information' refers to proprietary software code, designs, and financial specifications.",
        "2. Non-Disclosure Duty: Receiving Party shall hold all Confidential Information in strict confidence and shall not disclose it to any third party without prior written consent.",
        "3. Permitted Disclosure: Receiving Party may disclose Confidential Information solely to its employees and legal advisors who have a need to know.",
        "4. Return or Destruction: Upon termination of this Agreement, Receiving Party shall promptly return or destroy all physical and digital copies of Confidential Information.",
        "5. Governing Law: This agreement shall be governed by and construed under the laws of the State of California.",
        "6. Public Domain Exception: Receiving Party shall not be restricted regarding information that becomes publicly available without breach of this Agreement."
    ]
    
    scenario_1 = "Receiving Party may share some Confidential Information with some of Receiving Party's employees."
    scenario_2 = "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement."
    scenario_3 = "Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement."
    
    print("\n" + "="*80)
    print(f"DISPUTE 1: '{scenario_1}'")
    print("="*80)
    for r in engine.analyze(sample_clauses, scenario_1, top_k=2):
        print(f"\n[Rank {r['rank']}] Relevance: {r['relevance_score']:.1%} | Stance: {r['predicted_stance']} (Confidence: {r['stance_confidence']:.1%})")
        print(f"  Clause: \"{r['clause_text']}\"")
        print(f"  Detected Cues: {r['deontic_cues']}")

    print("\n" + "="*80)
    print(f"DISPUTE 2: '{scenario_2}'")
    print("="*80)
    for r in engine.analyze(sample_clauses, scenario_2, top_k=2):
        print(f"\n[Rank {r['rank']}] Relevance: {r['relevance_score']:.1%} | Stance: {r['predicted_stance']} (Confidence: {r['stance_confidence']:.1%})")
        print(f"  Clause: \"{r['clause_text']}\"")
        print(f"  Detected Cues: {r['deontic_cues']}")

    print("\n" + "="*80)
    print(f"DISPUTE 3: '{scenario_3}'")
    print("="*80)
    for r in engine.analyze(sample_clauses, scenario_3, top_k=2):
        print(f"\n[Rank {r['rank']}] Relevance: {r['relevance_score']:.1%} | Stance: {r['predicted_stance']} (Confidence: {r['stance_confidence']:.1%})")
        print(f"  Clause: \"{r['clause_text']}\"")
        print(f"  Detected Cues: {r['deontic_cues']}")
