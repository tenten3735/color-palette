import gradio as gr

def predict(value: str | None):
    # process value from the ColorPicker component
    return value



interface = gr.Interface(
    predict, 
    gr.ColorPicker(label='元の色を選択：'), 
    gr.Textbox(visible=False),
    flagging_mode='auto'
)


interface.launch()