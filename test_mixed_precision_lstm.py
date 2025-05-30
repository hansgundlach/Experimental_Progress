import torch
import torch.nn as nn
import time


# Simple LSTM test
class TestLSTM(nn.Module):
    def __init__(self, vocab_size=50000, hidden_size=512, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.linear(lstm_out)
        return output


def test_mixed_precision():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestLSTM().to(device)

    # Test data
    batch_size, seq_len = 64, 128
    x = torch.randint(0, 50000, (batch_size, seq_len)).to(device)

    # Test without mixed precision
    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    start_time = time.time()
    for i in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = output.mean()  # Dummy loss
        loss.backward()
        optimizer.step()
    fp32_time = time.time() - start_time

    # Test with mixed precision
    scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()
    for i in range(10):
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda"):
            output = model(x)
            loss = output.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    fp16_time = time.time() - start_time

    print(f"FP32 time: {fp32_time:.3f}s")
    print(f"Mixed precision time: {fp16_time:.3f}s")
    print(f"Speedup: {fp32_time/fp16_time:.2f}x")

    # Check memory usage
    print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    test_mixed_precision()
