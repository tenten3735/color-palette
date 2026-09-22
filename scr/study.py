import os
import random
import colorsys

import numpy as np
import torch
import torch.nn as nn


# ランダムでデータを作成
def color_data():
    r = round(random.random(), 2)
    g = round(random.random(), 2)
    b = round(random.random(), 2)

    # パラメーターランダム生成
    cute = round(random.random(), 2)      # かわいい
    calm = round(random.random(), 2)      # おちついた
    dark = round(random.random(), 2)      # 暗め
    vivid = round(random.random(), 2)     # ビビッド
    fantasy = round(random.random(), 2)   # 幻想的

    # RGB -> HSV
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # ランダム生成のパラメーターをもとに調整を入れる
    if cute > 0.5: # 彩度低め・明度高め
        s = min(0.4, s)
        v = max(0.8, v)

    if calm > 0.5: # 彩度低め・明度中くらい
        s = min(0.3, s)
        v = max(0.4, min(0.7, v))

    if dark > 0.5: # 明度低め
        v = min(0.3, v)

    if vivid > 0.5: # 彩度高め・明度高め
        s = max(0.8, s)
        v = max(0.8, v)

    colors = []
    for i in range(5):
        hue_step = 0.2 if fantasy > 0.5 else 0.08
        new_h = (h + i * hue_step) % 1.0
        
        r, g, b = colorsys.hsv_to_rgb(new_h, s, v)
        colors.extend([round(r, 2), round(g, 2), round(b, 2)])

    input_data = [r, g, b, cute, calm, dark, vivid, fantasy]
    
    return {"input": input_data, "output": colors}



train_data_list = []
samples_N = 10000
for _ in range(samples_N):
    data_pair = color_data()
    train_data_list.append(data_pair)

random.seed(10)
random.shuffle(train_data_list)

print("学習用データの数:", len(train_data_list))
print("最初の5個のデータペア（入力と模範解答）:")
for i in range(5):
    print(f"データ {i+1}:")
    print("  入力 (ベース色+パラメータ) :", train_data_list[i]["input"])
    print("  出力 (生成された5色RGB)  :", train_data_list[i]["output"])



# -------------------------- 学習準備 --------------------------
trainX_list = []
trainY_list = []

for item in train_data_list:
    trainX_list.append(item["input"])
    
    trainY_list.append(item["output"])


# 正規化
trainX = np.array(trainX_list, dtype = np.float32)  # (N, 8)
trainY = np.array(trainY_list, dtype = np.float32)  # (N, 15)

# Tensor化
trainX = torch.tensor(trainX)  # (N, 8)
trainY = torch.tensor(trainY)  # (N, 15)

# 確認
print("trainX shape (入力データ):", trainX.shape)
print("trainY shape (模範解答)  :", trainY.shape)

print("完了")



# -------------------------- 表示 --------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

for i in range(5):
    # trainXからベース色を取得
    base_rgb = trainX[i][:3].numpy()
    
    # trainYから出力の5色を取得して 5x3 に整形
    colors_rgb = trainY[i].numpy().reshape(5, 3)
    
    # ベース色表示
    plt.subplot(5, 6, i * 6 + 1)
    plt.imshow([[base_rgb]])
    plt.title(f"Data {i+1}\nBase")
    plt.axis("off")
    
    # 生成された5色を横に表示
    for j in range(5):
        plt.subplot(5, 6, i * 6 + + 2 + j)
        plt.imshow([[colors_rgb[j]]])
        plt.title(f"Color {j+1}")
        plt.axis("off")

plt.tight_layout()
plt.show()



# -------------------------- テスト用データの作成 --------------------------
test_data_list = []
test_N = 2000  # テストデータの数

for _ in range(test_N):
    data_pair = color_data()
    test_data_list.append(data_pair)

# 入力 (testX) と 出力 (testY) に分離
testX_list = [item["input"] for item in test_data_list]
testY_list = [item["output"] for item in test_data_list]

# Tensor (テンソル) 化
testX = torch.tensor(testX_list, dtype=torch.float32)  # (2000, 8)
testY = torch.tensor(testY_list, dtype=torch.float32)  # (2000, 15)

# --- 確認 ---
print("テスト用データの数:", len(testX))
print("testX shape (入力):", testX.shape)
print("testY shape (出力):", testY.shape)



from sklearn.model_selection import train_test_split

#データ分割
trainX_split, validX_split, trainY_split, validY_split = train_test_split(
    trainX, trainY, test_size=0.2, random_state=42, shuffle=True
)

# --- Tensor型の直接確認（前段階までのtrainX, trainY） ---
print("trainX shape:", trainX_split.shape)
print("trainY shape:", trainY_split.shape)
print("validX shape:", validX_split.shape)
print("validY shape:", validY_split.shape)
# --- testXはそのままTensorで保持している ---
print("testX shape:", testX.shape)

# --- trainX 保存 ---
if os.path.exists("trainX.pt"):
    print("trainX.pt found!")
else:
    torch.save(trainX, "trainX.pt")
    print("trainX.pt saved.")

# --- trainY 保存 ---
if os.path.exists("trainY.pt"):
    print("trainY.pt found!")
else:
    torch.save(trainY, "trainY.pt")
    print("trainY.pt saved.")

# --- testX 保存 ---
if os.path.exists("testX.pt"):
    print("testX.pt found!")
else:
    torch.save(testX, "testX.pt")
    print("testX.pt saved.")



# -------------------------- 学習 --------------------------
from color_palette import ColorPaletteNet
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# === ハイパーパラメータ ===
# batchサイズ
batch_size = 32
# optimaizerの学習率
lr = 0.001

# DataLoader
train_dataset = TensorDataset(trainX_split, trainY_split)
valid_dataset = TensorDataset(validX_split, validY_split)
test_dataset = TensorDataset(testX, testY)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# --- モデルのインスタンスを作る
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = ColorPaletteNet().to(device)
model = ColorPaletteNet()
model.summary((8, ))

# --- オプティマイザーの設定 ---
optimizer = optim.Adam(model.parameters(), lr=lr)

criterion = nn.MSELoss()
# --- EarlyStopping付き学習ループ ---
best_val_loss = float('inf')
early_stop_count = 0
#patience = 3
# EarlyStopping OFF
patience = 999999  # めったに止まらない（実質EarlyStopping無効）


# エポック数
n_epochs = 30

# 記録用リスト（Lossのみ記録）
train_loss_list, valid_loss_list = [], []

# EarlyStopping 用の変数リセット
best_val_loss = float('inf')
early_stop_count = 0



# === 学習ループ ===
for epoch in range(n_epochs):
    model.train()
    total_loss, total_samples = 0.0, 0
    
    for x, t in train_loader:
        x, t = x.to(device), t.to(device)

        y = model(x)
        loss = criterion(y, t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(x)
        total_samples += len(x)

    epoch_train_loss = total_loss / total_samples
    train_loss_list.append(epoch_train_loss)



    model.eval()
    total_loss, total_samples = 0.0, 0
    
    with torch.no_grad():
        for x, t in valid_loader:
            x, t = x.to(device), t.to(device)
            
            y = model(x)
            loss = criterion(y, t)
            
            total_loss += loss.item() * len(x)
            total_samples += len(x)

    epoch_valid_loss = total_loss / total_samples
    valid_loss_list.append(epoch_valid_loss)

    print(f"Epoch {epoch+1:02d}/{n_epochs} | Train Loss: {epoch_train_loss:.6f} | Valid Loss: {epoch_valid_loss:.6f}")


    if epoch_valid_loss < best_val_loss:
        best_val_loss = epoch_valid_loss
        early_stop_count = 0

        torch.save(model.state_dict(), "color_palette_best_model.pth")
    else:
        early_stop_count += 1
        if early_stop_count >= patience:
            print("Early stopping triggered.")
            break



# グラフ描画
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

epochs = np.arange(1, len(train_loss_list) + 1)

# Loss (誤差) の推移のみをプロット
plt.plot(epochs, train_loss_list, label="Train Loss", color="#1f77b4")
plt.plot(epochs, valid_loss_list, label="Valid Loss", color="#ff7f0e", linestyle="--")

plt.legend()
plt.grid()

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

# PDF画像として保存（授業のコードを踏襲）
plt.savefig("color_loss_graph.pdf")
plt.show()


# --------------------------------------------------------------------------
# アプリとつなげる前の動作チェック用
# --------------------------------------------------------------------------

import torch
import numpy as np
import matplotlib.pyplot as plt
from color_palette import ColorPaletteNet

model = ColorPaletteNet().to(device)
model.load_state_dict(torch.load("color_palette_best_model.pth"))
model.eval()  # 推論モードに切り替え

# 2. テスト用の入力パラメータを作成
# [R, G, B, Cute, Calm, Dark, Vivid, Fantasy]
sample_input = [1.0, 0.2, 0.2,  0.9, 0.1, 0.0, 0.8, 0.2]

input_tensor = torch.tensor([sample_input], dtype=torch.float32).to(device)

with torch.no_grad():
    predicted_colors = model(input_tensor)

colors_rgb = predicted_colors.cpu().numpy().reshape(5, 3)

plt.figure(figsize=(10, 2))
plt.suptitle("Generated 5-Color Palette")

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow([[np.clip(colors_rgb[i], 0.0, 1.0)]])
    plt.title(f"Color {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()