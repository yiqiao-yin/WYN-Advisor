import streamlit as st
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]


def call_chatcompletion(
    messages: list, model: str = "gpt-4", temperature: int = 0
) -> str:
    """
    Get the completion response from a list of messages using OpenAI's ChatCompletion API.

    Parameters:
    - messages (list): A list of messages which includes role ("user" or "assistant") and content.
    - model (str): The name of the OpenAI model to use. Default is "gpt-4".
    - temperature (int): The temperature parameter for generating more random or deterministic responses. Default is 0.

    Returns:
    - str: The content of the first response choice in the completed message.

    """
    # Call OpenAI's ChatCompletion API with the specified parameters
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=4000,
    )

    # Get the content of the first response choice in the completed message
    return response.choices[0].message["content"]