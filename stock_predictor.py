# Standalone LSTM prediction example
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

ticker = "RELIANCE.NS"
data = yf.download(
    ticker,
    period="1y",
    progress=False,
    auto_adjust=False,
)["Close"]

if hasattr(data, "columns"):
    data = data.iloc[:, 0]

data = data.dropna().values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

time_step = 60
X, y = [], []

for i in range(time_step, len(scaled)):
    X.append(scaled[i-time_step:i, 0])
    y.append(scaled[i, 0])

X = np.array(X).reshape(-1, time_step, 1)
y = np.array(y)


class StockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(1, 50, batch_first=True)
        self.lstm2 = nn.LSTM(50, 50, batch_first=True)
        self.fc1 = nn.Linear(50, 25)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, inputs):
        outputs, _ = self.lstm1(inputs)
        outputs, _ = self.lstm2(outputs)
        outputs = torch.relu(self.fc1(outputs[:, -1, :]))
        return self.fc2(outputs)


model = StockModel()
features = torch.tensor(X, dtype=torch.float32)
targets = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.MSELoss()

model.train()
for epoch in range(10):
    optimizer.zero_grad()
    loss = loss_function(model(features), targets)
    loss.backward()
    optimizer.step()

last_60 = scaled[-time_step:].reshape(1, time_step, 1)
model.eval()
with torch.no_grad():
    prediction = model(
        torch.tensor(last_60, dtype=torch.float32)
    ).numpy()

prediction = scaler.inverse_transform(prediction)

print("Stock:", ticker)
print("Current Price:", float(data[-1]))
print("Predicted Next Price:", float(prediction[0][0]))
