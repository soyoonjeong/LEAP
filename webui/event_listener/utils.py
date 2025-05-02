import gradio as gr
import pandas as pd


def create_args(elems):
    try:
        for key, value in elems["args"].items():
            if isinstance(value, list):
                elems["args"][key] = [",".join(value)]
        df = pd.DataFrame(elems["args"]).T
        df = df.reset_index()
        df.columns = ["index", "value"]
        return gr.DataFrame(
            value=df, label="Arguments", visible=True, interactive=False, max_height=250
        )
    except Exception as e:
        return gr.DataFrame(visible=False, interactive=False)
