import aiohttp
import gradio as gr
import asyncio
import uuid
import asyncio
from vllm import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    SamplingParams,
)


async def chat(message, history):
    url = "http://0.0.0.0:8080/chat"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"message": message}) as response:
                async for line in response.content:
                    token = line.decode("utf-8").strip()
                    if token:  # 토큰이 비어 있지 않을 때만 처리
                        yield history + [{"role": "assistant", "content": token}]
    except Exception as e:
        yield history + [{"role": "assistant", "content": f"Error: {str(e)}"}]


async def sse_generator(message, history):
    model = "/home/llm_models/Qwen/Qwen2.5-0.5B-Instruct"
    engine_args = {
        "model": model,
        "trust_remote_code": True,
        "disable_log_stats": True,
        "disable_log_requests": True,
    }
    model = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))

    sampling_params = SamplingParams(temperature=0.5, min_tokens=100, max_tokens=200)

    # engine.generate
    multi_modal_data = None
    result_generator = model.generate(
        inputs={"prompt": message, "multi_modal_data": multi_modal_data},
        sampling_params=sampling_params,
        request_id=str(uuid.uuid4()),
    )
    try:
        async for result in result_generator:
            if not result.finished:
                yield f"data: {result.outputs[0].text}\n\n"
                print(f"data: {result.outputs[0].text}\n\n")
                await asyncio.sleep(0)
    except Exception as e:
        yield "data: Error processing your request\n\n"
        print(e)


gr.ChatInterface(fn=chat, type="messages").launch()
