import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 2:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(y_true.mean())) ** 2))
    return 0.0 if ss_tot <= 0 else (1.0 - ss_res / ss_tot)


def pcc(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 2:
        return 0.0
    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))
    if (not np.isfinite(std_true)) or (not np.isfinite(std_pred)) or std_true < 1e-12 or std_pred < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def pearson_corr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.numel() < 2:
        return x.new_tensor(0.0)
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    vx = torch.sqrt(torch.mean(x * x) + eps)
    vy = torch.sqrt(torch.mean(y * y) + eps)
    corr = torch.mean(x * y) / (vx * vy + eps)
    return torch.clamp(corr, -1.0, 1.0)


def metrics_from_log_arrays(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "pcc": pcc(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def val_selection_score(metrics: dict[str, float], selection_metric: str) -> float:
    metric = str(selection_metric).lower().strip()
    if metric == "r2":
        return -float(metrics["r2"])
    if metric == "rmse":
        return float(metrics["rmse"])
    if metric == "mae":
        return float(metrics["mae"])
    if metric == "mae_pcc":
        return float(metrics["mae"] + (1.0 - metrics["pcc"]))
    if metric == "pcc":
        return -float(metrics["pcc"])
    raise ValueError(f"Unknown selection_metric={selection_metric!r}")


def residual_diagnostics(y_true, y0, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    correction = y_pred - y0
    base_err = np.abs(y0 - y_true)
    final_err = np.abs(y_pred - y_true)
    true_residual = y_true - y0
    y0_m = metrics_from_log_arrays(y_true, y0)
    final_m = metrics_from_log_arrays(y_true, y_pred)
    return {
        "y0_mae": y0_m["mae"],
        "y0_rmse": y0_m["rmse"],
        "y0_pcc": y0_m["pcc"],
        "y0_r2": y0_m["r2"],
        "final_mae": final_m["mae"],
        "final_rmse": final_m["rmse"],
        "final_pcc": final_m["pcc"],
        "final_r2": final_m["r2"],
        "corr_residual": pcc(true_residual, correction) if y_true.size >= 2 else 0.0,
        "mean_abs_correction": float(np.mean(np.abs(correction))) if correction.size else 0.0,
        "p95_abs_correction": float(np.quantile(np.abs(correction), 0.95)) if correction.size else 0.0,
        "improve_rate": float(np.mean(final_err < base_err)) if final_err.size else 0.0,
        "mean_abs_base_err": float(np.mean(base_err)) if base_err.size else 0.0,
        "mean_abs_final_err": float(np.mean(final_err)) if final_err.size else 0.0,
    }


def compute_uicl_three_term_loss(
    out: dict[str, torch.Tensor],
    y_true: torch.Tensor,
    reg_loss: str = "huber",
    beta: float = 0.3,
    residual_target_weight: float = 0.5,
    residual_clip: float = 1.0,
    ratio_loss_weight: float = 0.0,
    calib_reg_weight: float = 0.0,
    ratio_eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    def _reg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kind = str(reg_loss).lower().strip()
        if kind == "mae":
            return F.l1_loss(pred, target)
        if kind == "mse":
            return F.mse_loss(pred, target)
        if kind == "huber":
            return F.smooth_l1_loss(pred, target)
        raise ValueError(f"Unknown reg_loss={reg_loss!r}")

    y_pred = out["y_pred"]
    y0 = out["y0"]
    correction = out.get("correction", y_pred - y0)
    mu_ref = out.get("local_mu", y0).detach()
    loss_pred = _reg(y_pred, y_true)
    loss_retr = _reg(y0, y_true)
    residual_target = (y_true - y0.detach()).clamp(min=-float(residual_clip), max=float(residual_clip))
    loss_res = F.smooth_l1_loss(correction, residual_target)
    y_true_pos = y_true.clamp_min(0.0)
    y_pred_pos = y_pred.clamp_min(0.0)
    mu_pos = mu_ref.clamp_min(0.0)
    ratio_true = torch.log((y_true_pos + float(ratio_eps)) / (mu_pos + float(ratio_eps)))
    ratio_pred = torch.log((y_pred_pos + float(ratio_eps)) / (mu_pos + float(ratio_eps)))
    loss_ratio = F.smooth_l1_loss(ratio_pred, ratio_true)
    if "lambda_calib" in out and "lambda" in out:
        loss_calib = F.mse_loss(out["lambda_calib"], out["lambda"].detach())
    else:
        loss_calib = y_true.new_tensor(0.0)
    loss = (
        loss_pred
        + float(beta) * loss_retr
        + float(residual_target_weight) * loss_res
        + float(ratio_loss_weight) * loss_ratio
        + float(calib_reg_weight) * loss_calib
    )
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "loss_pred": float(loss_pred.detach().cpu().item()),
        "loss_retr": float(loss_retr.detach().cpu().item()),
        "loss_res": float(loss_res.detach().cpu().item()),
        "loss_ratio": float(loss_ratio.detach().cpu().item()),
        "loss_calib": float(loss_calib.detach().cpu().item()),
    }


class _UrbanICLRetrievalMixin:
    @staticmethod
    def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return torch.einsum("bd,nd->bn", a, b)

    @staticmethod
    def _pairwise_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return torch.einsum("bvd,nwd->bnvw", a, b)

    @staticmethod
    def _standardize(sim: torch.Tensor) -> torch.Tensor:
        mu = sim.mean(dim=1, keepdim=True)
        sd = sim.std(dim=1, keepdim=True, unbiased=False)
        return (sim - mu) / (sd + 1e-6)

    @staticmethod
    def _mask_logits(logits: torch.Tensor, sim_mask: torch.Tensor | None) -> torch.Tensor:
        if sim_mask is None:
            return logits
        if sim_mask.dim() == 1:
            mask = sim_mask.view(1, -1).to(logits.device, dtype=logits.dtype).expand_as(logits)
        else:
            mask = sim_mask.to(logits.device, dtype=logits.dtype)
        return logits.masked_fill(mask <= 0.0, -1e9)

    @staticmethod
    def _stabilize_view_weights(view_w: torch.Tensor, sim_mask: torch.Tensor | None, floor: float = 0.05) -> torch.Tensor:
        if sim_mask is None:
            return view_w
        if sim_mask.dim() == 1:
            mask = sim_mask.view(1, -1).to(view_w.device, dtype=view_w.dtype).expand_as(view_w)
        else:
            mask = sim_mask.to(view_w.device, dtype=view_w.dtype)
        active = (mask > 0).to(view_w.dtype)
        n_active = active.sum(dim=-1, keepdim=True).clamp_min(1.0)
        floor_total = float(floor) * n_active
        blend = torch.clamp(1.0 - floor_total, min=0.0)
        view_w = float(floor) * active + blend * view_w
        view_w = view_w * active
        return view_w / view_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _modality_mask_values(sim_mask: torch.Tensor | None, device: torch.device, dtype: torch.dtype):
        if sim_mask is None:
            return 1.0, 1.0, 1.0, 1.0
        mask = sim_mask if sim_mask.dim() == 1 else sim_mask[0]
        mask = mask.to(device=device, dtype=dtype)
        return float(mask[0].item()), float(mask[1].item()), float(mask[2].item()), float(mask[3].item())

    @staticmethod
    def _screen_stv_views(stv: torch.Tensor, sat_anchor: torch.Tensor, sem_anchor: torch.Tensor, topm: int = 4):
        v = F.normalize(stv, dim=-1)
        sat_anchor = F.normalize(sat_anchor, dim=-1)
        sem_anchor = F.normalize(sem_anchor, dim=-1)
        view_mean = F.normalize(v.mean(dim=1), dim=-1)
        sim_mean = torch.einsum("bvd,bd->bv", v, view_mean)
        sim_sat = torch.einsum("bvd,bd->bv", v, sat_anchor)
        sim_sem = torch.einsum("bvd,bd->bv", v, sem_anchor)
        score = sim_mean + 0.5 * sim_sat + 0.5 * sim_sem
        k = min(int(topm), int(stv.size(1)))
        vals, inds = torch.topk(score, k=k, dim=1)
        w = torch.softmax(vals, dim=1)
        chosen = torch.gather(stv, 1, inds.unsqueeze(-1).expand(-1, -1, stv.size(-1)))
        pooled = torch.sum(w.unsqueeze(-1) * chosen, dim=1)
        quality = vals.mean(dim=1, keepdim=True)
        return pooled, quality

    @staticmethod
    def _screen_stv_views_no_sem(stv: torch.Tensor, sat_anchor: torch.Tensor, topm: int = 4):
        v = F.normalize(stv, dim=-1)
        sat_anchor = F.normalize(sat_anchor, dim=-1)
        view_mean = F.normalize(v.mean(dim=1), dim=-1)
        sim_mean = torch.einsum("bvd,bd->bv", v, view_mean)
        sim_sat = torch.einsum("bvd,bd->bv", v, sat_anchor)
        score = sim_mean + sim_sat
        k = min(int(topm), int(stv.size(1)))
        vals, inds = torch.topk(score, k=k, dim=1)
        w = torch.softmax(vals, dim=1)
        chosen = torch.gather(stv, 1, inds.unsqueeze(-1).expand(-1, -1, stv.size(-1)))
        pooled = torch.sum(w.unsqueeze(-1) * chosen, dim=1)
        quality = vals.mean(dim=1, keepdim=True)
        return pooled, quality

    def _instance_match(self, q_stv: torch.Tensor, bank_stv: torch.Tensor) -> torch.Tensor:
        sim = self._pairwise_cos(q_stv, bank_stv)
        q2c = sim.max(dim=3).values.mean(dim=2)
        c2q = sim.max(dim=2).values.mean(dim=2)
        return 0.5 * (q2c + c2q)

    @staticmethod
    def _retrieve(sim: torch.Tensor, bank_y: torch.Tensor, topk: int, tau: float, bank_self_idx: torch.Tensor | None):
        if bank_self_idx is not None:
            sim = sim.scatter(1, bank_self_idx.unsqueeze(1), -1e9)
        k = min(int(topk), sim.size(1))
        vals, inds = torch.topk(sim, k=k, dim=1)
        logits = vals / max(float(tau), 1e-6)
        logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
        alpha = torch.softmax(logits, dim=1)
        alpha = torch.nan_to_num(alpha, nan=1.0 / max(k, 1), posinf=1.0, neginf=0.0)
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
        neigh_y = bank_y[inds]
        y0 = torch.sum(alpha * neigh_y, dim=1)
        y_mu = torch.sum(alpha * neigh_y, dim=1, keepdim=True)
        y_var = torch.sum(alpha * (neigh_y - y_mu) ** 2, dim=1, keepdim=True)
        y_sd = torch.sqrt(y_var).clamp_min(1e-6)
        p = alpha.clamp_min(1e-9)
        ent = -(p * p.log()).sum(dim=1, keepdim=True) / math.log(float(max(k, 2)))
        margin = (vals[:, :1] - vals[:, 1:2]) if k >= 2 else torch.zeros_like(ent)
        disp = vals.std(dim=1, keepdim=True, unbiased=False)
        y_q25 = torch.quantile(neigh_y, 0.25, dim=1, keepdim=True)
        y_q75 = torch.quantile(neigh_y, 0.75, dim=1, keepdim=True)
        return {
            "y": torch.nan_to_num(y0, nan=0.0, posinf=0.0, neginf=0.0),
            "vals": vals,
            "inds": inds,
            "alpha": alpha,
            "ent": torch.nan_to_num(ent, nan=1.0, posinf=1.0, neginf=1.0),
            "margin": torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0),
            "disp": torch.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0),
            "y_mu": torch.nan_to_num(y_mu, nan=0.0, posinf=0.0, neginf=0.0),
            "y_var": torch.nan_to_num(y_var, nan=0.0, posinf=0.0, neginf=0.0),
            "y_sd": torch.nan_to_num(y_sd, nan=1.0, posinf=1.0, neginf=1.0),
            "y_iqr": torch.nan_to_num(y_q75 - y_q25, nan=0.0, posinf=0.0, neginf=0.0),
        }

    @staticmethod
    def _candidate_topk(topk: int, bank_size: int, multiplier: int = 3, min_extra: int = 5) -> int:
        topk = int(topk)
        bank_size = int(bank_size)
        return max(topk, min(bank_size, max(topk * int(multiplier), topk + int(min_extra))))


class SemEnhancedRetrieverReasoner(nn.Module, _UrbanICLRetrievalMixin):
    def __init__(
        self,
        dim,
        sem_dim,
        mech_dim,
        hidden,
        dropout,
        use_query_adaptive_lambda,
        reason_arch="retr_residual",
        residual_gate_min=0.0,
        residual_gate_max=0.50,
    ):
        super().__init__()
        self.dim = int(dim)
        self.dropout = nn.Dropout(dropout)
        self.reason_arch = str(reason_arch)
        self.residual_gate_min = float(residual_gate_min)
        self.residual_gate_max = float(residual_gate_max)

        self.sem_proj = nn.Sequential(
            nn.Linear(sem_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.view_prior_logits = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        if use_query_adaptive_lambda:
            self.view_gate_mlp = nn.Sequential(
                nn.Linear(dim * 3, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 4),
            )
        else:
            self.view_gate_mlp = None
        self.stv_quality_gate = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
        self.inst_gate_mlp = nn.Sequential(
            nn.Linear(dim * 3 + 1, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )
        self.inst_consistency_gate = nn.Sequential(
            nn.Linear(2, hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, 1),
            nn.Sigmoid(),
        )
        self.mech_pair_mlp = nn.Sequential(
            nn.Linear(mech_dim * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.mech_view_mlp = nn.Sequential(
            nn.Linear(mech_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 4),
        )
        self.mech_gamma_mlp = nn.Sequential(
            nn.Linear(4, hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, 1),
            nn.Sigmoid(),
        )
        self.mech_delta_proj = nn.Sequential(
            nn.Linear(mech_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, max(16, hidden // 8)),
            nn.ReLU(),
        )
        self.candidate_multiplier = 3
        self.candidate_extra = 5
        self.rerank_scale_logit = nn.Parameter(torch.tensor(-1.3862944, dtype=torch.float32))
        final_dim = 7 + max(16, hidden // 8)
        self.correction_head = nn.Sequential(
            nn.Linear(final_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.residual_scale_logit = nn.Parameter(torch.tensor(-2.1972246, dtype=torch.float32))

    def forward(
        self,
        q_sat,
        q_stv,
        q_sem_raw,
        q_mech,
        bank_sat,
        bank_stv,
        bank_sem_raw,
        bank_mech,
        bank_y_norm,
        topk,
        tau,
        bank_self_idx,
        beta,
        sim_mask,
        use_consensus=True,
        use_sem_anchor=True,
        use_relative_evidence=True,
        use_conf_gate=True,
    ):
        del beta, use_consensus, use_sem_anchor
        bsz = q_sat.size(0)
        sat_on, stv_on, inst_on, sem_on = self._modality_mask_values(sim_mask, q_sat.device, q_sat.dtype)

        q_sem = self.sem_proj(q_sem_raw) * sem_on
        bank_sem = self.sem_proj(bank_sem_raw) * sem_on
        q_sat = q_sat * sat_on
        bank_sat = bank_sat * sat_on
        q_stv = q_stv * stv_on
        bank_stv = bank_stv * stv_on

        q_stv_pool, q_stv_quality = self._screen_stv_views(q_stv, q_sat, q_sem, topm=4)
        bank_stv_pool, _ = self._screen_stv_views(bank_stv, bank_sat, bank_sem, topm=4)

        sim_sat = self._standardize(self._cos(q_sat, bank_sat))
        sim_stv = self._standardize(self._cos(q_stv_pool, bank_stv_pool))
        sim_inst = self._standardize(self._instance_match(q_stv, bank_stv)) * inst_on
        sim_sem = self._standardize(self._cos(q_sem, bank_sem))

        view_logits = self.view_prior_logits.view(1, 4).expand(bsz, -1)
        if self.view_gate_mlp is not None:
            query_feat = torch.cat([q_sat, q_stv_pool, q_sem], dim=-1)
            view_logits = view_logits + self.view_gate_mlp(self.dropout(query_feat))
        view_logits = self._mask_logits(view_logits, sim_mask)
        base_view_w = torch.softmax(view_logits, dim=-1)
        base_view_w = self._stabilize_view_weights(base_view_w, sim_mask, floor=0.05)

        stv_rel = 0.3 + 0.7 * self.stv_quality_gate(q_stv_quality).squeeze(-1)
        inst_gate_input = torch.cat([q_sat, q_stv_pool, q_sem, q_stv_quality], dim=-1)
        inst_gate = self.inst_gate_mlp(self.dropout(inst_gate_input)).squeeze(-1) * inst_on
        inst_consistency = self.inst_consistency_gate(
            torch.cat([
                sim_sat.mean(dim=1, keepdim=True),
                sim_stv.mean(dim=1, keepdim=True),
            ], dim=-1)
        ).squeeze(-1)
        inst_gate = inst_gate * inst_consistency
        sim_base = (
            base_view_w[:, 0:1] * sim_sat
            + base_view_w[:, 1:2] * stv_rel.unsqueeze(-1) * sim_stv
            + base_view_w[:, 2:3] * inst_gate.unsqueeze(-1) * sim_inst
            + base_view_w[:, 3:4] * sim_sem
        )
        retrieved_base = self._retrieve(sim_base, bank_y_norm, topk, tau, bank_self_idx)
        mech_present = (q_mech.abs().sum(dim=-1, keepdim=True) > 1e-8).to(q_sat.dtype)
        mech_view_logits = self.mech_view_mlp(self.dropout(q_mech))
        mech_view_logits = self._mask_logits(mech_view_logits, sim_mask)
        mech_view_w = torch.softmax(mech_view_logits, dim=-1)
        mech_view_w = self._stabilize_view_weights(mech_view_w, sim_mask, floor=0.05)
        mech_gamma_input = torch.cat(
            [
                retrieved_base["ent"],
                retrieved_base["margin"],
                retrieved_base["disp"],
                q_stv_quality,
            ],
            dim=-1,
        )
        mech_gamma = self.mech_gamma_mlp(self.dropout(mech_gamma_input)) * mech_present
        view_w = (1.0 - mech_gamma) * base_view_w + mech_gamma * mech_view_w
        view_w = view_w / view_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        sim_joint = (
            view_w[:, 0:1] * sim_sat
            + view_w[:, 1:2] * stv_rel.unsqueeze(-1) * sim_stv
            + view_w[:, 2:3] * inst_gate.unsqueeze(-1) * sim_inst
            + view_w[:, 3:4] * sim_sem
        )
        retrieved = self._retrieve(sim_joint, bank_y_norm, topk, tau, bank_self_idx)
        y0 = retrieved["y"]

        candidate_k = self._candidate_topk(topk, sim_joint.size(1), self.candidate_multiplier, self.candidate_extra)
        retrieved_cand = self._retrieve(sim_joint, bank_y_norm, candidate_k, tau, bank_self_idx)
        cand_inds = retrieved_cand["inds"]
        cand_alpha = retrieved_cand["alpha"]
        neigh_y = bank_y_norm[cand_inds]
        neigh_mech = bank_mech[cand_inds]
        q_mech_expand = q_mech.unsqueeze(1).expand_as(neigh_mech)
        delta_mech = q_mech_expand - neigh_mech
        mech_pair_feat = torch.cat([q_mech_expand, neigh_mech, delta_mech, torch.abs(delta_mech)], dim=-1)
        mech_pair_logits = self.mech_pair_mlp(self.dropout(mech_pair_feat)).squeeze(-1)
        rerank_scale = torch.sigmoid(self.rerank_scale_logit) * mech_present.squeeze(-1)
        rerank_logits = (retrieved_cand["vals"] / max(float(tau), 1e-6)) + rerank_scale.unsqueeze(-1) * mech_pair_logits
        refine_k = min(int(topk), int(rerank_logits.size(1)))
        rerank_vals, rerank_pos = torch.topk(rerank_logits, k=refine_k, dim=1)
        inds = torch.gather(cand_inds, 1, rerank_pos)
        alpha_mech = torch.softmax(rerank_vals, dim=1)
        alpha_mech = torch.nan_to_num(alpha_mech, nan=1.0 / max(int(refine_k), 1), posinf=1.0, neginf=0.0)
        alpha_mech = alpha_mech / alpha_mech.sum(dim=1, keepdim=True).clamp_min(1e-8)
        neigh_y_refined = bank_y_norm[inds]
        y_mech = torch.sum(alpha_mech * neigh_y_refined, dim=1)
        local_mu = torch.sum(alpha_mech * neigh_y_refined, dim=1)
        local_var = torch.sum(alpha_mech * (neigh_y_refined - local_mu.unsqueeze(-1)) ** 2, dim=1)
        local_sigma = torch.sqrt(local_var.clamp_min(1e-8))
        neigh_mech_refined = bank_mech[inds]
        ctx_sat = torch.sum(alpha_mech.unsqueeze(-1) * bank_sat[inds], dim=1)
        ctx_stv = torch.sum(alpha_mech.unsqueeze(-1) * bank_stv_pool[inds], dim=1)
        ctx_sem = torch.sum(alpha_mech.unsqueeze(-1) * bank_sem[inds], dim=1)
        delta_sat = q_sat - ctx_sat
        delta_stv = q_stv_pool - ctx_stv
        delta_sem = q_sem - ctx_sem

        mech_ctx = torch.sum(alpha_mech.unsqueeze(-1) * neigh_mech_refined, dim=1)
        mech_delta_global = q_mech - mech_ctx
        mech_delta_small = self.mech_delta_proj(self.dropout(mech_delta_global))
        corr_input = torch.cat(
            [
                y0.unsqueeze(-1),
                y_mech.unsqueeze(-1),
                local_sigma.unsqueeze(-1),
                retrieved["disp"],
                retrieved["ent"],
                retrieved["margin"],
                q_stv_quality,
                mech_delta_small,
            ],
            dim=-1,
        )
        if not use_relative_evidence:
            corr_input = torch.cat(
                [
                    y0.unsqueeze(-1),
                    y_mech.unsqueeze(-1),
                    local_sigma.unsqueeze(-1),
                    torch.zeros_like(retrieved["disp"]),
                    torch.zeros_like(retrieved["ent"]),
                    torch.zeros_like(retrieved["margin"]),
                    q_stv_quality,
                    mech_delta_small,
                ],
                dim=-1,
        )
        residual_scale = torch.sigmoid(self.residual_scale_logit)
        correction = residual_scale * local_sigma * torch.tanh(self.correction_head(self.dropout(corr_input)).squeeze(-1))
        y_pred = y_mech + correction
        correction = y_pred - y0
        alpha_base_refined = torch.gather(cand_alpha, 1, rerank_pos)
        gate = torch.mean(torch.abs(alpha_mech - alpha_base_refined), dim=-1)

        return {
            "y_pred": torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0),
            "y0": y0,
            "local_mu": local_mu,
            "local_sigma": local_sigma,
            "lambda": view_w,
            "lambda_calib": base_view_w,
            "inst_gate": torch.nan_to_num(inst_gate, nan=0.0, posinf=0.0, neginf=0.0),
            "alpha": alpha_mech,
            "inds": inds,
            "retrieval_conf": torch.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0),
            "delta_y": torch.mean(torch.gather(mech_pair_logits, 1, rerank_pos), dim=-1),
            "correction": torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0),
            "calibration_scale": residual_scale.expand_as(y0),
            "mech_gamma": mech_gamma.squeeze(-1),
        }


class SemEnhancedRetrieverNoReason(nn.Module, _UrbanICLRetrievalMixin):
    def __init__(self, dim, sem_dim, mech_dim, hidden, dropout, use_query_adaptive_lambda):
        del mech_dim
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.sem_proj = nn.Sequential(
            nn.Linear(sem_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.view_prior_logits = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        if use_query_adaptive_lambda:
            self.view_gate_mlp = nn.Sequential(
                nn.Linear(dim * 3, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 4),
            )
        else:
            self.view_gate_mlp = None
        self.stv_quality_gate = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
        self.inst_gate_mlp = nn.Sequential(
            nn.Linear(dim * 3 + 1, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )
        self.inst_consistency_gate = nn.Sequential(
            nn.Linear(2, hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        q_sat,
        q_stv,
        q_sem_raw,
        q_mech,
        bank_sat,
        bank_stv,
        bank_sem_raw,
        bank_mech,
        bank_y_norm,
        topk,
        tau,
        bank_self_idx,
        sim_mask,
        use_consensus=True,
        use_sem_anchor=True,
    ):
        del q_mech, bank_mech
        del use_consensus, use_sem_anchor
        bsz = q_sat.size(0)
        sat_on, stv_on, inst_on, sem_on = self._modality_mask_values(sim_mask, q_sat.device, q_sat.dtype)

        q_sem = self.sem_proj(q_sem_raw) * sem_on
        bank_sem = self.sem_proj(bank_sem_raw) * sem_on
        q_sat = q_sat * sat_on
        bank_sat = bank_sat * sat_on
        q_stv = q_stv * stv_on
        bank_stv = bank_stv * stv_on

        q_stv_pool, q_stv_quality = self._screen_stv_views(q_stv, q_sat, q_sem, topm=4)
        bank_stv_pool, _ = self._screen_stv_views(bank_stv, bank_sat, bank_sem, topm=4)
        sim_sat = self._standardize(self._cos(q_sat, bank_sat))
        sim_stv = self._standardize(self._cos(q_stv_pool, bank_stv_pool))
        sim_inst = self._standardize(self._instance_match(q_stv, bank_stv)) * inst_on
        sim_sem = self._standardize(self._cos(q_sem, bank_sem))

        view_logits = self.view_prior_logits.view(1, 4).expand(bsz, -1)
        if self.view_gate_mlp is not None:
            query_feat = torch.cat([q_sat, q_stv_pool, q_sem], dim=-1)
            view_logits = view_logits + self.view_gate_mlp(self.dropout(query_feat))
        view_logits = self._mask_logits(view_logits, sim_mask)
        view_w = torch.softmax(view_logits, dim=-1)
        view_w = self._stabilize_view_weights(view_w, sim_mask, floor=0.05)
        stv_rel = 0.3 + 0.7 * self.stv_quality_gate(q_stv_quality).squeeze(-1)
        inst_gate_input = torch.cat([q_sat, q_stv_pool, q_sem, q_stv_quality], dim=-1)
        inst_gate = self.inst_gate_mlp(self.dropout(inst_gate_input)).squeeze(-1) * inst_on
        inst_consistency = self.inst_consistency_gate(
            torch.cat([
                sim_sat.mean(dim=1, keepdim=True),
                sim_stv.mean(dim=1, keepdim=True),
            ], dim=-1)
        ).squeeze(-1)
        inst_gate = inst_gate * inst_consistency

        sim_joint = (
            view_w[:, 0:1] * sim_sat
            + view_w[:, 1:2] * stv_rel.unsqueeze(-1) * sim_stv
            + view_w[:, 2:3] * inst_gate.unsqueeze(-1) * sim_inst
            + view_w[:, 3:4] * sim_sem
        )
        retrieved = self._retrieve(sim_joint, bank_y_norm, topk, tau, bank_self_idx)
        retrieved_sat = self._retrieve(sim_sat, bank_y_norm, topk, tau, bank_self_idx)
        retrieved_stv = self._retrieve(stv_rel.unsqueeze(-1) * sim_stv, bank_y_norm, topk, tau, bank_self_idx)
        retrieved_inst = self._retrieve(inst_gate.unsqueeze(-1) * sim_inst, bank_y_norm, topk, tau, bank_self_idx)
        retrieved_sem = self._retrieve(sim_sem, bank_y_norm, topk, tau, bank_self_idx)
        y0 = (
            view_w[:, 0] * retrieved_sat["y"]
            + view_w[:, 1] * retrieved_stv["y"]
            + view_w[:, 2] * retrieved_inst["y"]
            + view_w[:, 3] * retrieved_sem["y"]
        )
        return {"y_pred": y0, "y0": y0, "lambda": view_w, "inst_gate": torch.nan_to_num(inst_gate, nan=0.0, posinf=0.0, neginf=0.0), "alpha": retrieved["alpha"], "inds": retrieved["inds"]}


class SemEnhancedRetrieverReasonerNoSem(nn.Module, _UrbanICLRetrievalMixin):
    def __init__(
        self,
        dim,
        sem_dim,
        mech_dim,
        hidden,
        dropout,
        use_query_adaptive_lambda,
        reason_arch="retr_residual",
        residual_gate_min=0.0,
        residual_gate_max=0.50,
    ):
        del sem_dim, reason_arch
        super().__init__()
        self.dim = int(dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_gate_min = float(residual_gate_min)
        self.residual_gate_max = float(residual_gate_max)
        self.view_prior_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        if use_query_adaptive_lambda:
            self.view_gate_mlp = nn.Sequential(
                nn.Linear(dim * 2, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 3),
            )
        else:
            self.view_gate_mlp = None
        self.stv_quality_gate = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
        self.inst_gate_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )
        self.inst_consistency_gate = nn.Sequential(
            nn.Linear(2, hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, 1),
            nn.Sigmoid(),
        )
        self.mech_pair_mlp = nn.Sequential(
            nn.Linear(mech_dim * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.mech_view_mlp = nn.Sequential(
            nn.Linear(mech_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 3),
        )
        self.mech_gamma_mlp = nn.Sequential(
            nn.Linear(4, hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, 1),
            nn.Sigmoid(),
        )
        self.mech_delta_proj = nn.Sequential(
            nn.Linear(mech_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, max(16, hidden // 8)),
            nn.ReLU(),
        )
        self.candidate_multiplier = 3
        self.candidate_extra = 5
        self.rerank_scale_logit = nn.Parameter(torch.tensor(-1.3862944, dtype=torch.float32))
        final_dim = 7 + max(16, hidden // 8)
        self.correction_head = nn.Sequential(
            nn.Linear(final_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.residual_scale_logit = nn.Parameter(torch.tensor(-2.1972246, dtype=torch.float32))

    def forward(
        self,
        q_sat,
        q_stv,
        q_sem_raw,
        q_mech,
        bank_sat,
        bank_stv,
        bank_sem_raw,
        bank_mech,
        bank_y_norm,
        topk,
        tau,
        bank_self_idx,
        beta,
        sim_mask,
        use_consensus=True,
        use_sem_anchor=True,
        use_relative_evidence=True,
        use_conf_gate=True,
    ):
        del q_sem_raw, bank_sem_raw, beta, use_consensus, use_sem_anchor
        bsz = q_sat.size(0)
        sat_on, stv_on, inst_on, _ = self._modality_mask_values(sim_mask, q_sat.device, q_sat.dtype)
        q_sat = q_sat * sat_on
        bank_sat = bank_sat * sat_on
        q_stv = q_stv * stv_on
        bank_stv = bank_stv * stv_on

        q_stv_pool, q_stv_quality = self._screen_stv_views_no_sem(q_stv, q_sat, topm=4)
        bank_stv_pool, _ = self._screen_stv_views_no_sem(bank_stv, bank_sat, topm=4)

        sim_sat = self._standardize(self._cos(q_sat, bank_sat))
        sim_stv = self._standardize(self._cos(q_stv_pool, bank_stv_pool))
        sim_inst = self._standardize(self._instance_match(q_stv, bank_stv)) * inst_on

        view_logits = self.view_prior_logits.view(1, 3).expand(bsz, -1)
        if self.view_gate_mlp is not None:
            query_feat = torch.cat([q_sat, q_stv_pool], dim=-1)
            view_logits = view_logits + self.view_gate_mlp(self.dropout(query_feat))
        if sim_mask is not None:
            if sim_mask.dim() == 1:
                sim_mask_3 = torch.stack([sim_mask[0], sim_mask[1], sim_mask[2]])
            else:
                sim_mask_3 = torch.stack([sim_mask[:, 0], sim_mask[:, 1], sim_mask[:, 2]], dim=1)
        else:
            sim_mask_3 = None
        view_logits = self._mask_logits(view_logits, sim_mask_3)
        base_view_w = torch.softmax(view_logits, dim=-1)
        base_view_w = self._stabilize_view_weights(base_view_w, sim_mask_3, floor=0.05)

        stv_rel = 0.3 + 0.7 * self.stv_quality_gate(q_stv_quality).squeeze(-1)
        inst_gate_input = torch.cat([q_sat, q_stv_pool, q_stv_quality], dim=-1)
        inst_gate = self.inst_gate_mlp(self.dropout(inst_gate_input)).squeeze(-1) * inst_on
        inst_consistency = self.inst_consistency_gate(
            torch.cat([
                sim_sat.mean(dim=1, keepdim=True),
                sim_stv.mean(dim=1, keepdim=True),
            ], dim=-1)
        ).squeeze(-1)
        inst_gate = inst_gate * inst_consistency
        sim_base = (
            base_view_w[:, 0:1] * sim_sat
            + base_view_w[:, 1:2] * stv_rel.unsqueeze(-1) * sim_stv
            + base_view_w[:, 2:3] * inst_gate.unsqueeze(-1) * sim_inst
        )
        retrieved_base = self._retrieve(sim_base, bank_y_norm, topk, tau, bank_self_idx)
        mech_present = (q_mech.abs().sum(dim=-1, keepdim=True) > 1e-8).to(q_sat.dtype)
        mech_view_logits = self.mech_view_mlp(self.dropout(q_mech))
        mech_view_logits = self._mask_logits(mech_view_logits, sim_mask_3)
        mech_view_w = torch.softmax(mech_view_logits, dim=-1)
        mech_view_w = self._stabilize_view_weights(mech_view_w, sim_mask_3, floor=0.05)
        mech_gamma_input = torch.cat(
            [
                retrieved_base["ent"],
                retrieved_base["margin"],
                retrieved_base["disp"],
                q_stv_quality,
            ],
            dim=-1,
        )
        mech_gamma = self.mech_gamma_mlp(self.dropout(mech_gamma_input)) * mech_present
        view_w = (1.0 - mech_gamma) * base_view_w + mech_gamma * mech_view_w
        view_w = view_w / view_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        sim_joint = (
            view_w[:, 0:1] * sim_sat
            + view_w[:, 1:2] * stv_rel.unsqueeze(-1) * sim_stv
            + view_w[:, 2:3] * inst_gate.unsqueeze(-1) * sim_inst
        )
        retrieved = self._retrieve(sim_joint, bank_y_norm, topk, tau, bank_self_idx)
        y0 = retrieved["y"]

        candidate_k = self._candidate_topk(topk, sim_joint.size(1), self.candidate_multiplier, self.candidate_extra)
        retrieved_cand = self._retrieve(sim_joint, bank_y_norm, candidate_k, tau, bank_self_idx)
        cand_inds = retrieved_cand["inds"]
        cand_alpha = retrieved_cand["alpha"]
        neigh_y = bank_y_norm[cand_inds]
        neigh_mech = bank_mech[cand_inds]
        q_mech_expand = q_mech.unsqueeze(1).expand_as(neigh_mech)
        delta_mech = q_mech_expand - neigh_mech
        mech_pair_feat = torch.cat([q_mech_expand, neigh_mech, delta_mech, torch.abs(delta_mech)], dim=-1)
        mech_pair_logits = self.mech_pair_mlp(self.dropout(mech_pair_feat)).squeeze(-1)
        rerank_scale = torch.sigmoid(self.rerank_scale_logit) * mech_present.squeeze(-1)
        rerank_logits = (retrieved_cand["vals"] / max(float(tau), 1e-6)) + rerank_scale.unsqueeze(-1) * mech_pair_logits
        refine_k = min(int(topk), int(rerank_logits.size(1)))
        rerank_vals, rerank_pos = torch.topk(rerank_logits, k=refine_k, dim=1)
        inds = torch.gather(cand_inds, 1, rerank_pos)
        alpha_mech = torch.softmax(rerank_vals, dim=1)
        alpha_mech = torch.nan_to_num(alpha_mech, nan=1.0 / max(int(refine_k), 1), posinf=1.0, neginf=0.0)
        alpha_mech = alpha_mech / alpha_mech.sum(dim=1, keepdim=True).clamp_min(1e-8)
        neigh_y_refined = bank_y_norm[inds]
        y_mech = torch.sum(alpha_mech * neigh_y_refined, dim=1)
        local_mu = torch.sum(alpha_mech * neigh_y_refined, dim=1)
        local_var = torch.sum(alpha_mech * (neigh_y_refined - local_mu.unsqueeze(-1)) ** 2, dim=1)
        local_sigma = torch.sqrt(local_var.clamp_min(1e-8))
        neigh_mech_refined = bank_mech[inds]
        ctx_sat = torch.sum(alpha_mech.unsqueeze(-1) * bank_sat[inds], dim=1)
        ctx_stv = torch.sum(alpha_mech.unsqueeze(-1) * bank_stv_pool[inds], dim=1)
        delta_sat = q_sat - ctx_sat
        delta_stv = q_stv_pool - ctx_stv

        mech_ctx = torch.sum(alpha_mech.unsqueeze(-1) * neigh_mech_refined, dim=1)
        mech_delta_global = q_mech - mech_ctx
        mech_delta_small = self.mech_delta_proj(self.dropout(mech_delta_global))
        corr_input = torch.cat(
            [
                y0.unsqueeze(-1),
                y_mech.unsqueeze(-1),
                local_sigma.unsqueeze(-1),
                retrieved["disp"],
                retrieved["ent"],
                retrieved["margin"],
                q_stv_quality,
                mech_delta_small,
            ],
            dim=-1,
        )
        if not use_relative_evidence:
            corr_input = torch.cat(
                [
                    y0.unsqueeze(-1),
                    y_mech.unsqueeze(-1),
                    local_sigma.unsqueeze(-1),
                    torch.zeros_like(retrieved["disp"]),
                    torch.zeros_like(retrieved["ent"]),
                    torch.zeros_like(retrieved["margin"]),
                    q_stv_quality,
                    mech_delta_small,
                ],
                dim=-1,
        )
        residual_scale = torch.sigmoid(self.residual_scale_logit)
        correction = residual_scale * local_sigma * torch.tanh(self.correction_head(self.dropout(corr_input)).squeeze(-1))
        y_pred = y_mech + correction
        correction = y_pred - y0
        alpha_base_refined = torch.gather(cand_alpha, 1, rerank_pos)
        gate = torch.mean(torch.abs(alpha_mech - alpha_base_refined), dim=-1)
        return {
            "y_pred": torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0),
            "y0": y0,
            "local_mu": local_mu,
            "local_sigma": local_sigma,
            "lambda": view_w,
            "lambda_calib": base_view_w,
            "inst_gate": torch.nan_to_num(inst_gate, nan=0.0, posinf=0.0, neginf=0.0),
            "alpha": alpha_mech,
            "inds": inds,
            "retrieval_conf": torch.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0),
            "delta_y": torch.mean(torch.gather(mech_pair_logits, 1, rerank_pos), dim=-1),
            "correction": torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0),
            "calibration_scale": residual_scale.expand_as(y0),
            "mech_gamma": mech_gamma.squeeze(-1),
        }


class SemEnhancedRetrieverNoReasonNoSem(nn.Module, _UrbanICLRetrievalMixin):
    def __init__(self, dim, sem_dim, mech_dim, hidden, dropout, use_query_adaptive_lambda):
        del sem_dim, mech_dim
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.view_prior_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        if use_query_adaptive_lambda:
            self.view_gate_mlp = nn.Sequential(
                nn.Linear(dim * 2, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 3),
            )
        else:
            self.view_gate_mlp = None
        self.stv_quality_gate = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
        self.inst_gate_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )
        self.inst_consistency_gate = nn.Sequential(
            nn.Linear(2, hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        q_sat,
        q_stv,
        q_sem_raw,
        q_mech,
        bank_sat,
        bank_stv,
        bank_sem_raw,
        bank_mech,
        bank_y_norm,
        topk,
        tau,
        bank_self_idx,
        sim_mask,
        use_consensus=True,
        use_sem_anchor=True,
    ):
        del q_sem_raw, q_mech, bank_sem_raw, bank_mech, use_consensus, use_sem_anchor
        bsz = q_sat.size(0)
        sat_on, stv_on, inst_on, _ = self._modality_mask_values(sim_mask, q_sat.device, q_sat.dtype)
        q_sat = q_sat * sat_on
        bank_sat = bank_sat * sat_on
        q_stv = q_stv * stv_on
        bank_stv = bank_stv * stv_on

        q_stv_pool, q_stv_quality = self._screen_stv_views_no_sem(q_stv, q_sat, topm=4)
        bank_stv_pool, _ = self._screen_stv_views_no_sem(bank_stv, bank_sat, topm=4)
        sim_sat = self._standardize(self._cos(q_sat, bank_sat))
        sim_stv = self._standardize(self._cos(q_stv_pool, bank_stv_pool))
        sim_inst = self._standardize(self._instance_match(q_stv, bank_stv)) * inst_on

        view_logits = self.view_prior_logits.view(1, 3).expand(bsz, -1)
        if self.view_gate_mlp is not None:
            query_feat = torch.cat([q_sat, q_stv_pool], dim=-1)
            view_logits = view_logits + self.view_gate_mlp(self.dropout(query_feat))
        if sim_mask is not None:
            if sim_mask.dim() == 1:
                sim_mask_3 = torch.stack([sim_mask[0], sim_mask[1], sim_mask[2]])
            else:
                sim_mask_3 = torch.stack([sim_mask[:, 0], sim_mask[:, 1], sim_mask[:, 2]], dim=1)
        else:
            sim_mask_3 = None
        view_logits = self._mask_logits(view_logits, sim_mask_3)
        view_w = torch.softmax(view_logits, dim=-1)
        view_w = self._stabilize_view_weights(view_w, sim_mask_3, floor=0.05)
        stv_rel = 0.3 + 0.7 * self.stv_quality_gate(q_stv_quality).squeeze(-1)
        inst_gate_input = torch.cat([q_sat, q_stv_pool, q_stv_quality], dim=-1)
        inst_gate = self.inst_gate_mlp(self.dropout(inst_gate_input)).squeeze(-1) * inst_on
        inst_consistency = self.inst_consistency_gate(
            torch.cat([
                sim_sat.mean(dim=1, keepdim=True),
                sim_stv.mean(dim=1, keepdim=True),
            ], dim=-1)
        ).squeeze(-1)
        inst_gate = inst_gate * inst_consistency

        sim_joint = (
            view_w[:, 0:1] * sim_sat
            + view_w[:, 1:2] * stv_rel.unsqueeze(-1) * sim_stv
            + view_w[:, 2:3] * inst_gate.unsqueeze(-1) * sim_inst
        )
        retrieved = self._retrieve(sim_joint, bank_y_norm, topk, tau, bank_self_idx)
        retrieved_sat = self._retrieve(sim_sat, bank_y_norm, topk, tau, bank_self_idx)
        retrieved_stv = self._retrieve(stv_rel.unsqueeze(-1) * sim_stv, bank_y_norm, topk, tau, bank_self_idx)
        retrieved_inst = self._retrieve(inst_gate.unsqueeze(-1) * sim_inst, bank_y_norm, topk, tau, bank_self_idx)
        y0 = (
            view_w[:, 0] * retrieved_sat["y"]
            + view_w[:, 1] * retrieved_stv["y"]
            + view_w[:, 2] * retrieved_inst["y"]
        )
        return {"y_pred": y0, "y0": y0, "lambda": view_w, "inst_gate": torch.nan_to_num(inst_gate, nan=0.0, posinf=0.0, neginf=0.0), "alpha": retrieved["alpha"], "inds": retrieved["inds"]}


@torch.no_grad()
def evaluate(
    model,
    loader,
    bank,
    y_mean,
    y_std,
    device,
    topk,
    tau,
    with_reasoning,
    sem_lookup=None,
    use_consensus=True,
    use_sem_anchor=True,
    use_relative_evidence=True,
    use_conf_gate=True,
):
    model.eval()
    ys_true, ys_pred = [], []
    for batch in loader:
        q_sat = batch["sat"].to(device)
        q_stv = batch["stv"].to(device)
        q_sem = batch["sem"].to(device) if sem_lookup is None else torch.stack([sem_lookup[rid] for rid in batch["region_id"]], dim=0).to(device)
        q_mech = batch["mech"].to(device)
        y_true_norm = batch["y_norm"].to(device)
        bank_self_idx = None
        if bank.get("id2idx") is not None:
            region_ids = list(batch["region_id"])
            if all(rid in bank["id2idx"] for rid in region_ids):
                bank_self_idx = torch.tensor([bank["id2idx"][rid] for rid in region_ids], device=device, dtype=torch.long)
        if with_reasoning:
            out = model(q_sat, q_stv, q_sem, q_mech, bank["sat"], bank["stv"], bank["sem"], bank["mech"], bank["y_norm"], topk, tau, bank_self_idx, 0.0, bank.get("sim_mask", None), use_consensus, use_sem_anchor, use_relative_evidence, use_conf_gate)
        else:
            out = model(q_sat, q_stv, q_sem, q_mech, bank["sat"], bank["stv"], bank["sem"], bank["mech"], bank["y_norm"], topk, tau, bank_self_idx, bank.get("sim_mask", None), use_consensus, use_sem_anchor)
        y_pred = (out["y_pred"] * y_std) + y_mean
        y_true = (y_true_norm * y_std) + y_mean
        ys_true.extend(y_true.detach().cpu().numpy().tolist())
        ys_pred.extend(y_pred.detach().cpu().numpy().tolist())
    return metrics_from_log_arrays(ys_true, ys_pred)


@torch.no_grad()
def evaluate_detailed(
    model,
    loader,
    bank,
    y_mean,
    y_std,
    device,
    topk,
    tau,
    sem_lookup=None,
    use_consensus=True,
    use_sem_anchor=True,
    use_relative_evidence=True,
    use_conf_gate=True,
):
    model.eval()
    y_true_all, y0_all, y_pred_all = [], [], []
    for batch in loader:
        q_sat = batch["sat"].to(device)
        q_stv = batch["stv"].to(device)
        q_sem = batch["sem"].to(device) if sem_lookup is None else torch.stack([sem_lookup[rid] for rid in batch["region_id"]], dim=0).to(device)
        q_mech = batch["mech"].to(device)
        y_true_norm = batch["y_norm"].to(device)
        bank_self_idx = None
        if bank.get("id2idx") is not None:
            region_ids = list(batch["region_id"])
            if all(rid in bank["id2idx"] for rid in region_ids):
                bank_self_idx = torch.tensor([bank["id2idx"][rid] for rid in region_ids], device=device, dtype=torch.long)
        out = model(q_sat, q_stv, q_sem, q_mech, bank["sat"], bank["stv"], bank["sem"], bank["mech"], bank["y_norm"], topk, tau, bank_self_idx, 0.0, bank.get("sim_mask", None), use_consensus, use_sem_anchor, use_relative_evidence, use_conf_gate)
        y_true = (y_true_norm * y_std) + y_mean
        y0 = (out["y0"] * y_std) + y_mean
        y_pred = (out["y_pred"] * y_std) + y_mean
        y_true_all.extend(y_true.detach().cpu().numpy().tolist())
        y0_all.extend(y0.detach().cpu().numpy().tolist())
        y_pred_all.extend(y_pred.detach().cpu().numpy().tolist())
    return np.asarray(y_true_all), np.asarray(y0_all), np.asarray(y_pred_all)


@torch.no_grad()
def collect_retrieval_examples(
    model,
    loader,
    bank,
    y_mean,
    y_std,
    device,
    topk,
    tau,
    sem_lookup=None,
    with_reasoning=True,
    use_consensus=True,
    use_sem_anchor=True,
    use_relative_evidence=True,
    use_conf_gate=True,
):
    model.eval()
    examples = []
    bank_ids = list(bank.get("ids", []))
    bank_y = ((bank["y_norm"].detach().cpu() * y_std) + y_mean)
    for batch in loader:
        q_sat = batch["sat"].to(device)
        q_stv = batch["stv"].to(device)
        q_sem = batch["sem"].to(device) if sem_lookup is None else torch.stack([sem_lookup[rid] for rid in batch["region_id"]], dim=0).to(device)
        q_mech = batch["mech"].to(device)
        y_true_norm = batch["y_norm"].to(device)
        batch_ids = list(batch["region_id"])
        bank_self_idx = None
        if bank.get("id2idx") is not None and all(rid in bank["id2idx"] for rid in batch_ids):
            bank_self_idx = torch.tensor([bank["id2idx"][rid] for rid in batch_ids], device=device, dtype=torch.long)
        if with_reasoning:
            out = model(q_sat, q_stv, q_sem, q_mech, bank["sat"], bank["stv"], bank["sem"], bank["mech"], bank["y_norm"], topk, tau, bank_self_idx, 0.0, bank.get("sim_mask", None), use_consensus, use_sem_anchor, use_relative_evidence, use_conf_gate)
        else:
            out = model(q_sat, q_stv, q_sem, q_mech, bank["sat"], bank["stv"], bank["sem"], bank["mech"], bank["y_norm"], topk, tau, bank_self_idx, bank.get("sim_mask", None), use_consensus, use_sem_anchor)

        y_true = ((y_true_norm * y_std) + y_mean).detach().cpu()
        y0 = ((out["y0"] * y_std) + y_mean).detach().cpu()
        y_pred = ((out["y_pred"] * y_std) + y_mean).detach().cpu()
        inds = out["inds"].detach().cpu()
        alpha = out["alpha"].detach().cpu()

        for row_idx, rid in enumerate(batch_ids):
            neigh = []
            for k_idx in range(int(inds.size(1))):
                bank_idx = int(inds[row_idx, k_idx].item())
                neigh_id = bank_ids[bank_idx] if bank_idx < len(bank_ids) else str(bank_idx)
                neigh.append(
                    {
                        "rank": int(k_idx + 1),
                        "region_id": neigh_id,
                        "alpha": float(alpha[row_idx, k_idx].item()),
                        "y_value": float(bank_y[bank_idx].item()),
                    }
                )
            examples.append(
                {
                    "region_id": rid,
                    "y_true_value": float(y_true[row_idx].item()),
                    "y0_value": float(y0[row_idx].item()),
                    "y_pred_value": float(y_pred[row_idx].item()),
                    "retrieval_abs_err": float(abs(y0[row_idx].item() - y_true[row_idx].item())),
                    "final_abs_err": float(abs(y_pred[row_idx].item() - y_true[row_idx].item())),
                    "neighbors": neigh,
                }
            )
    return examples


@torch.no_grad()
def evaluate_reasoning_stats(
    model,
    loader,
    bank,
    device,
    topk,
    tau,
    sem_lookup=None,
    use_consensus=True,
    use_sem_anchor=True,
    use_relative_evidence=True,
    use_conf_gate=True,
):
    model.eval()
    corr_all, gate_all, delta_all, sigma_all, scale_all = [], [], [], [], []
    for batch in loader:
        q_sat = batch["sat"].to(device)
        q_stv = batch["stv"].to(device)
        q_sem = batch["sem"].to(device) if sem_lookup is None else torch.stack([sem_lookup[rid] for rid in batch["region_id"]], dim=0).to(device)
        q_mech = batch["mech"].to(device)
        bank_self_idx = None
        if bank.get("id2idx") is not None:
            region_ids = list(batch["region_id"])
            if all(rid in bank["id2idx"] for rid in region_ids):
                bank_self_idx = torch.tensor([bank["id2idx"][rid] for rid in region_ids], device=device, dtype=torch.long)
        out = model(
            q_sat, q_stv, q_sem, q_mech, bank["sat"], bank["stv"], bank["sem"], bank["mech"], bank["y_norm"],
            topk, tau, bank_self_idx, 0.0, bank.get("sim_mask", None),
            use_consensus, use_sem_anchor, use_relative_evidence, use_conf_gate,
        )
        if "correction" in out:
            corr_all.append(out["correction"].detach().float().cpu())
        if "retrieval_conf" in out:
            gate_all.append(out["retrieval_conf"].detach().float().cpu())
        if "delta_y" in out:
            delta_all.append(out["delta_y"].detach().float().cpu())
        if "local_sigma" in out:
            sigma_all.append(out["local_sigma"].detach().float().cpu())
        if "calibration_scale" in out:
            scale_all.append(out["calibration_scale"].detach().float().cpu())

    def _summ(x_list):
        if not x_list:
            return None
        x = torch.cat([t.reshape(-1) for t in x_list], dim=0).numpy()
        return {
            "mean": float(np.mean(x)),
            "abs_mean": float(np.mean(np.abs(x))),
            "p50_abs": float(np.quantile(np.abs(x), 0.50)),
            "p95_abs": float(np.quantile(np.abs(x), 0.95)),
        }

    return {
        "correction": _summ(corr_all),
        "gate": _summ(gate_all),
        "delta": _summ(delta_all),
        "sigma": _summ(sigma_all),
        "scale": _summ(scale_all),
    }


@torch.no_grad()
def evaluate_mean_view_weights(
    model,
    loader,
    bank,
    device,
    topk,
    tau,
    sem_lookup=None,
    with_reasoning=True,
    use_consensus=True,
    use_sem_anchor=True,
    use_relative_evidence=True,
    use_conf_gate=True,
):
    model.eval()
    sum_w = None
    count = 0
    for batch in loader:
        q_sat = batch["sat"].to(device)
        q_stv = batch["stv"].to(device)
        q_sem = batch["sem"].to(device) if sem_lookup is None else torch.stack([sem_lookup[rid] for rid in batch["region_id"]], dim=0).to(device)
        q_mech = batch["mech"].to(device)
        bank_self_idx = None
        if bank.get("id2idx") is not None:
            region_ids = list(batch["region_id"])
            if all(rid in bank["id2idx"] for rid in region_ids):
                bank_self_idx = torch.tensor([bank["id2idx"][rid] for rid in region_ids], device=device, dtype=torch.long)
        if with_reasoning:
            out = model(q_sat, q_stv, q_sem, q_mech, bank["sat"], bank["stv"], bank["sem"], bank["mech"], bank["y_norm"], topk, tau, bank_self_idx, 0.0, bank.get("sim_mask", None), use_consensus, use_sem_anchor, use_relative_evidence, use_conf_gate)
        else:
            out = model(q_sat, q_stv, q_sem, q_mech, bank["sat"], bank["stv"], bank["sem"], bank["mech"], bank["y_norm"], topk, tau, bank_self_idx, bank.get("sim_mask", None), use_consensus, use_sem_anchor)
        w = out.get("lambda", None)
        if w is None:
            continue
        w = w.detach().float().cpu()
        sum_w = w.sum(dim=0) if sum_w is None else sum_w + w.sum(dim=0)
        count += int(w.size(0))
    if sum_w is None or count == 0:
        return None
    return (sum_w / float(count)).numpy()


def get_default_ablations():
    return [
        ("full", "sat,stv,inst,sem", True, "normal", True, True, True, True),
        ("no_reason", "sat,stv,inst,sem", False, "normal", True, True, True, True),
        ("no_inst", "sat,stv,sem", True, "normal", True, True, True, True),
        # ("no_sem", "sat,stv,inst", True, "normal", True, False, True, True),
        ("no_sat", "stv,inst,sem", True, "normal", True, True, True, True),
        ("no_stv", "sat,sem", True, "normal", True, True, True, True),
        # ("rand_sem", "sat,stv,inst,sem", True, "random", True, True, True, True),
    ]
