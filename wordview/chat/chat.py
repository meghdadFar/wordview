import pandas as pd
from openai import OpenAI


class Datachat:
    def __init__(self, api_key: str = "", dataframe: pd.DataFrame = None):
        self.api_key = api_key
        self.dataframe = dataframe
        self.client = OpenAI(api_key=credentials.get("openai_api_key"))

    def wordview_chat(self):
        # history = ""
        while True:
            user_prompt = input("You: ")

            prompt_for_action = f"""Classify "user prompt" below into one of the following categories:
            [multiword_expressions, bias_detection, text_analysis, unknown, continuation].

            Only return the class name without any extra explanation, words or tokens in your response.

            Use the class label continuation if the user prompt is a follow up on a previous prompt.

            Use the class label unknown if you cannot make a suggestion with high confidence, or the prompt is not a continuation.

            "user prompt":
            {user_prompt}
            """
            gpt_action_suggestion = (
                self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt_for_action},
                    ],
                )
                .choices[0]
                .message.content
            )

            if gpt_action_suggestion == "unknown":
                print(
                    "ChatGPT: I'm not sure what you're asking. Please provide more context or try a different prompt."
                )
                continue

            # if gpt_action_suggestion == "continuation":
            #     pass
            #     # gpt_response = (
            #     #     openai.Completion.create(
            #     #         engine="davinci", prompt=user_prompt, max_tokens=100
            #     #     )
            #     #     .choices[0]
            #     #     .text.strip()
            #     # )
            #     # history += f"\nUser: {user_prompt}\nChatGPT: {gpt_response}\n"

            # if gpt_action_suggestion in self.wordview_functions.keys():
            #     history += f"\nUser: {user_prompt}\nChatGPT: {gpt_action_suggestion}\n"
            #     if gpt_action_suggestion == "multiword_expressions":
            #         pass


if __name__ == "__main__":
    import json

    with open("wordview/chat/secrets/openai_api_key.json", "r") as f:
        credentials = json.load(f)
    datachat = Datachat(api_key=credentials["openai_api_key"])
    datachat.wordview_chat()
