import gradio as gr
from gradio_leaderboard import Leaderboard, SelectColumns, ColumnFilter
from event_listener import leaderboard_listener

from config.leaderboard import LEADERBOARD_ON_LOAD_COLUMNS, LEADERBOARD_TYPES


def create_leaderboard_tab():
    with gr.Tab("🏅 LLM LeaderBoard") as leaderboard_tab:
        df = leaderboard_listener.init_llm_leaderboard()
        leaderboard = Leaderboard(
            value=df,
            select_columns=SelectColumns(
                default_selection=LEADERBOARD_ON_LOAD_COLUMNS,
                cant_deselect=[],
                label="Select Columns to Display:",
            ),
            search_columns=["Model"],
            filter_columns=[
                ColumnFilter(
                    "#Params(B)",
                    default=[min(df["#Params(B)"]), max(df["#Params(B)"])],
                )
            ],
            datatype=LEADERBOARD_TYPES,
            interactive=False,
        )

    leaderboard_tab.select(
        fn=leaderboard_listener.update_leaderboard,
        inputs=[leaderboard],
        outputs=[leaderboard],
    )
