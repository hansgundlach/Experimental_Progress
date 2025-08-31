import math, copy




def chinchilla_scale(base_cfg, hidden_dims):
    """
    Return a list of configs that satisfy:
      • tokens ≈ 20 × parameters
      • per-step compute budget unchanged vs. baseline
      • depth/width ratio fixed (layers ∝ hidden_dim)
    """

    def param_count(d, L):
        # crude but width-dominant: 12·L·d²  (ignores embeddings/out-proj)
        return 12 * L * d**2

    base_d = base_cfg["hidden_dim"]
    base_L = base_cfg["num_layers"]
    base_bsz = base_cfg["batch_size"]
    base_lr = base_cfg["learning_rate"]
    base_clip = base_cfg["gradient_clip_val"]
    seq_len = base_cfg["seq_length"]

    out = []
    for d in hidden_dims:
        width_scale = d / base_d

        # 1) Depth: keep L ∝ d   (so aspect-ratio is preserved)
        L = max(1, int(round(base_L * width_scale)))

        # 2) Keep per-step FLOPs ≈ const ⇒ batch ∝ 1 / (width² · depth/base_depth)
        flops_scale = (width_scale**2) * (L / base_L)
        bsz = max(1, int(round(base_bsz / flops_scale)))

        # 3) LR & grad-clip heuristics
        lr = base_lr * (base_d / d) ** 0.5
        clip = base_clip * math.sqrt(width_scale)

        # 4) Chinchilla target tokens  (≈ 20 × parameters)
        params = param_count(d, L)
        tgt_tok = int(20 * params)

        # 5) Convert token target into epochs
        tokens_per_step = bsz * seq_len
        est_steps = math.ceil(tgt_tok / tokens_per_step)
        max_epochs = math.ceil(
            est_steps / (len(base_cfg.get("dataset", [])) or 1)
        )  # adjust as needed

        cfg = copy.deepcopy(base_cfg)
        cfg.update(
            {
                "hidden_dim": d,
                "num_layers": L,
                "num_heads": max(1, d // 16),
                "batch_size": bsz,
                "learning_rate": lr,
                "gradient_clip_val": clip,
                "target_tokens": tgt_tok,
                "max_epochs": max(max_epochs, cfg.get("min_epochs", 1)),
            }
        )
        out.append(cfg)
    return out
