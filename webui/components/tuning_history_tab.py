import gradio as gr

from event_listener import queue_listener


def create_tuning_history_tab():
    with gr.Tab("🧠 Tuning History") as tuning_history_tab:
        finished_queue_elems = gr.Dropdown(
            label="Finished Tasks",
            show_label=False,
            visible=True,
            interactive=True,
        )
        with gr.Row():
            finished_args = gr.DataFrame(visible=False, interactive=False)
            finished_metrics = gr.DataFrame(visible=False, interactive=False)
        with gr.Row():
            loss_graph = gr.Plot(label="Loss", visible=True)
            grad_norm_graph = gr.Plot(label="Grad Norm", visible=True)

        finished_queue_elems.change(
            fn=queue_listener.select_tuning_history_queue,
            inputs=[finished_queue_elems],
            outputs=[finished_args, finished_metrics, loss_graph, grad_norm_graph],
        )
    return {
        "tuning_history_tab": tuning_history_tab,
        "finished_queue_elems": finished_queue_elems,
        "finished_args": finished_args,
        "finished_metrics": finished_metrics,
    }
