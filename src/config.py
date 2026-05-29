# src/config.py

class Config:
    # --- Data ---
    dataset = "pems04"                  # or "pems08"
    data_path = {
        "pems04": "data/raw/pems04.npz",
        "pems08": "data/raw/pems08.npz",
    }
    num_sensors = {"pems04": 307, "pems08": 170}
    feature_channel = 0                 # 0 = traffic flow (paper uses this only)
    input_len = 12                      # 12 time steps = 1 hour of history
    horizons = [3, 6, 12]              # predict at 15min, 30min, 60min

    # --- Data split (paper: 70/20/10) ---
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    # --- Model ---
    d_model = 512                       # 8 heads * 64 per head = 512 total
    n_heads = 8                         # explicitly stated in paper
    d_ff = 1024                         # feedforward dim, 2x d_model (standard convention)
    dropout = 0.2                       # explicitly stated in paper
    n_transformer_layers = 1            # ASSUMED: paper implies single block
    tcn_kernel_size = 3                 # explicitly stated
    tcn_layers = 4                      # explicitly stated
    tcn_channels = 512                  

    # --- Training ---
    batch_size = 64                     # ASSUMED: common default
    learning_rate = 0.001               # explicitly stated
    weight_decay = 1e-4                 # ASSUMED: standard L2 lambda
    max_epochs = 200
    patience = 25                       # ASSUMED: early stopping patience
    grad_clip = 5.0                     # ASSUMED: standard for transformer training

    # --- System ---
    seed = 42
    device = "cuda"                     # falls back to cpu if unavailable
    checkpoint_dir = "checkpoints"
    results_dir = "results"