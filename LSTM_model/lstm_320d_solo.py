from lstm_experiment_utils import gen_lstm_experim
GRAND_EXPERIMENT = (
    gen_lstm_experim(320, label="320d", folder_name="x1_lstm_layer1_320d_solo",
                     learning_rate=0.01413933, token_to_param_ratio=40,
                     num_layers=1, csv_log_interval=200)
    + gen_lstm_experim(320, label="320d", folder_name="x2_lstm_layer2_320d_solo",
                       learning_rate=0.01956473, token_to_param_ratio=40,
                       num_layers=2, csv_log_interval=200)
)
