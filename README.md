# color-palette (からーぱれっど！)

ベース色と5つの雰囲気スライダー（かわいい、おちついた、暗め、ビビッド、幻想的）を入力することで、ニューラルネットワークが最適な5色のカラーパレットを自動生成するWebアプリケーションです。

## 特徴

- **ベースカラー＋パラメーター**: 好きな色と欲しい雰囲気設定
- **カラーパレッド生成**: PyTorchの多層パーセプトロンモデルが連続的なRGBの組み合わせを予測
- **直感的なWeb UI**: Gradioを採用し、直感的なスライダー操作とプレビュー表示を実現

## 使用技術（Tech Stack）

- **Language**: Python 3.13
- **Framework / Libraries**:
  - `PyTorch` (ニューラルネットワーク)
  - `Gradio` (Web UIの構築)
  - `NumPy` / `Matplotlib` (データ処理・学習ログの可視化)
  - `scikit-learn` (データ分割)

---

## ファイル構成

```text
.
├── README.md                   # プロジェクトの説明と実行方法（本ファイル）
├── requirements.txt            # ライブラリの依存関係ファイル
├── AGENTS.md
├── docs/                       # 詳細資料・スライドなど
├── data/                       # 使用データセット
│   ├── dataset.jsonl
│   ├── testX.pt
│   ├── testY.pt
│   ├── trainX.pt
│   └── trainY.pt
└── src/
    ├── main.py                 # Gradio アプリの起動メインスクリプト
    ├── create_model.py         # モデル定義 (ColorPaletteNet)
    ├── study.py                # 学習過程
    └── color_palette_best_model.pth # 学習済みモデルの重みファイル
```

## モデル構造と学習について

- **入力**: ベース色（RGB）＋ パラメータ（Cute, Calm, Dark, Vivid, Fantasy）
- **出力**: 提案される5色分のRGB値
- **損失関数**: `MSELoss` (平均二乗誤差)
- **最適化アルゴリズム**: `Adam` (Learning Rate: 0.001)

## 実行方法

### 1. 依存ライブラリのインストール

必要なライブラリをインストールします。

```bash
pip install -r requirements.txt
```

### 2. アプリの起動

以下のコマンドで Web アプリを立ち上げます。

```bash
python main.py
```

実行後、ターミナルに表示される URL（例: `http://127.0.0.1:7860`）にブラウザでアクセスしてください。
