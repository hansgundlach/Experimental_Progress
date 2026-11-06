from lstm_experiment_utils import gen_lstm_experim
GRAND_EXPERIMENT = (
    gen_lstm_experim(416, label="416d", folder_name="x1_lstm_layer1_lr_trend",
                     learning_rate=0.01144552, token_to_param_ratio=40,
                     num_layers=1, csv_log_interval=200)
    + gen_lstm_experim(416, label="416d", folder_name="x2_lstm_layer2_lr_trend",
                       learning_rate=0.01695679, token_to_param_ratio=40,
                       num_layers=2, csv_log_interval=200)
)
