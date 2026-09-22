import os
import colorsys
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, random_split


def extract_features(palette_rgb):
    r, g, b = palette_rgb[0]

    # HSV 変換
    hsv_list = [colorsys.rgb_to_hsv(r, g, b) for r, g, b in palette_rgb]
    saturations = [hsv[1] for hsv in hsv_list]
    values = [hsv[2] for hsv in hsv_list]
    hues = [hsv[0] for hsv in hsv_list]

    avg_s = np.mean(saturations)
    avg_v = np.mean(values)

    # 雰囲気パラメータの逆算
    dark = float(np.clip(1.0 - avg_v, 0.0, 1.0))
    vivid = float(np.clip(avg_s, 0.0, 1.0))
    cute = float(np.clip(avg_v * avg_s, 0.0, 1.0))
    calm = float(np.clip((1.0 - avg_s) * avg_v, 0.0, 1.0))
    fantasy = float(np.clip(np.std(hues) * 2.0, 0.0, 1.0))

    return [r, g, b, cute, calm, dark, vivid, fantasy]


# JSON ファイルの読み込みと変換処理
file_path = "data/dataset.jsonl"
X_list = []
Y_list = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not line:
            continue

        item = json.loads(line)

        rgb_array = np.array(item["rgbs"], dtype=np.float32) / 255.0

        # 不整データを除外
        if rgb_array.shape != (5, 3):
            continue

        x_8d = extract_features(rgb_array)
        y_15d = rgb_array.flatten()

        X_list.append(x_8d)
        Y_list.append(y_15d)


# PyTorch テンソルへ変換
X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
Y_tensor = torch.tensor(np.array(Y_list), dtype=torch.float32)

# 学習用80%・テスト用20%に分ける
total_size = len(X_tensor)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

generator = torch.Generator().manual_seed(42)

full_dataset = TensorDataset(X_tensor, Y_tensor)
train_dataset, test_dataset = random_split(
    full_dataset, [train_size, test_size], generator=generator
)

trainX = X_tensor[train_dataset.indices]
trainY = Y_tensor[train_dataset.indices]
testX = X_tensor[test_dataset.indices]
testY = Y_tensor[test_dataset.indices]

# data/ フォルダに保存
os.makedirs("data", exist_ok=True)
torch.save(trainX, "data/trainX.pt")
torch.save(trainY, "data/trainY.pt")
torch.save(testX, "data/testX.pt")
torch.save(testY, "data/testY.pt")

print(f"処理完了！ 全{total_size}件 -> 学習データ: {len(trainX)}件 / テストデータ: {len(testX)}件")



# -------------------------- 学習準備 --------------------------
X_tensor = torch.load("data/trainX.pt")
Y_tensor = torch.load("data/trainY.pt")

dataset = TensorDataset(X_tensor, Y_tensor)

train_loader = DataLoader(dataset, batch_size=64, shuffle=True)


# -------------------------- 表示 --------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

for i in range(5):
    # trainXからベース色を取得
    base_rgb = X_tensor[i][:3].numpy()
    
    # trainYから出力の5色を取得して 5x3 に整形
    colors_rgb = Y_tensor[i].numpy().reshape(5, 3)
    
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

os.makedirs("data", exist_ok=True)

# --- trainX 保存 ---
if os.path.exists("data/trainX.pt"):
    print("data/trainX.pt found!")
else:
    torch.save(trainX, "data/trainX.pt")
    print("data/trainX.pt saved.")

# --- trainY 保存 ---
if os.path.exists("data/trainY.pt"):
    print("data/trainY.pt found!")
else:
    torch.save(trainY, "data/trainY.pt")
    print("data/trainY.pt saved.")

# --- testX 保存 ---
if os.path.exists("data/testX.pt"):
    print("data/testX.pt found!")
else:
    torch.save(testX, "data/testX.pt")
    print("data/testX.pt saved.")



# -------------------------- 学習 --------------------------
from create_model import ColorPaletteNet
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

model = ColorPaletteNet().to(device)
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
n_epochs = 50

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

        torch.save(model.state_dict(), "scr/color_palette_best_model.pth")
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

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from create_model import ColorPaletteNet

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = ColorPaletteNet().to(device)
# model.load_state_dict(torch.load("scr/color_palette_best_model.pth"))
# model.eval()  # 推論モードに切り替え

# # テスト用の入力パラメータを作成
# # [R, G, B, Cute, Calm, Dark, Vivid, Fantasy]
# sample_input = [0.95, 0.4, 0.5,  0.9, 0.1, 0.0, 0.8, 0.3]

# input_tensor = torch.tensor([sample_input], dtype=torch.float32).to(device)

# with torch.no_grad():
#     predicted_colors = model(input_tensor)

# colors_rgb = predicted_colors.cpu().numpy().reshape(5, 3)

# plt.figure(figsize=(10, 2))
# plt.suptitle("Generated 5-Color Palette")

# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow([[np.clip(colors_rgb[i], 0.0, 1.0)]])
#     plt.title(f"Color {i+1}")
#     plt.axis("off")

# plt.tight_layout()
# plt.show()

