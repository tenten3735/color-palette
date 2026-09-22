import gradio as gr
import colorsys
import numpy as np
import torch

from create_model import ColorPaletteNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ColorPaletteNet().to(device)
model.load_state_dict(torch.load("color_palette_best_model.pth", map_location=device))
model.eval()

# Hex -> RGB
def get_rgb_float(hex):
    r = int(hex[1:3], 16) / 255.0
    g = int(hex[3:5], 16) / 255.0
    b = int(hex[5:7], 16) / 255.0
    return r, g, b

# RGB -> Hex
def rgb_to_hex(r, g, b):
    r_int = int(np.clip(r * 255, 0, 255))
    g_int = int(np.clip(g * 255, 0, 255))
    b_int = int(np.clip(b * 255, 0, 255))
    return f"#{r_int:02x}{g_int:02x}{b_int:02x}"

# 予測
def predict(base, cute, calm, dark, vivid, fantasy):
    r, g, b = get_rgb_float(base)
    
    input_list = [r, g, b, cute, calm, dark, vivid, fantasy]
    input_tensor = torch.tensor([input_list], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        
    rgb = output.cpu().numpy().reshape(5, 3)
    
    hex = []
    for rgb in rgb:
        hex.append(rgb_to_hex(rgb[0], rgb[1], rgb[2]))
        

    html_preview = "<div style='display: flex; gap: 8px; margin-top: 10px;'>"
    for hex_c in hex:
        html_preview += f"<div style='background-color: {hex_c}; width: 60px; height: 60px; border-radius: 6px;'></div>"
    html_preview += "</div>"

    return html_preview, ", ".join(hex)




interface = gr.Interface(
    predict, 
    inputs=[
        gr.ColorPicker(label='元の色を選択：'),
        gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="かわいい"),
        gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="おちついた"),
        gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="暗め"),
        gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="ビビッド"),
        gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="幻想的"),
    ], 
    outputs=[
        gr.HTML(label="パレットプレビュー"),
        gr.Textbox(label="生成された5色のHEXコード")
    ],
    title="からーぱれっど！",
    flagging_mode='never'
)

if __name__ == "__main__":
    interface.launch()
