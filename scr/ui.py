import gradio as gr
import colorsys

def getColor(value):
    r = int(value[1:3], 16)
    g = int(value[3:5], 16)
    b = int(value[5:7], 16)

    h, s, v = colorsys.rgb_to_hsv(
        r / 255,
        g / 255,
        b / 255
    )

    return h, s, v

def predict(value: str | None):
    # process value from the ColorPicker component
    h, s, v = getColor(value)
    hsv = str(h)+', '+str(s)+', '+str(v)
    return value, hsv


interface = gr.Interface(
    predict, 
    gr.ColorPicker(label='元の色を選択：'), 
    [gr.Textbox(label="Hex"), gr.Textbox(label="HSV")],
    flagging_mode='auto'
)

interface.launch()