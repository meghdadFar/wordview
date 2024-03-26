import openai
import pandas as pd


class Datachat:
    def __init__(self, api_key: str = "", dataframe: pd.DataFrame = None):
        self.api_key = api_key
        self.dataframe = dataframe
        self.wordview_functions = {
            "multiword_expressions": "This function extracts different types of multiword expressions (MWEs) from a text corpus. \
            It can identify MWEs such as Light Verb Constructions, Noun Compounds, Adjective Noun Compounds, and Verb Particles.",
            "bias_detection": "This function detects bias in a given text by analyzing the sentiment and emotional tone of the text.\
            It can identify gender, racial and religious bias in the text.",
        }

    def wordview_chat(self):
        history = ""
        while True:
            user_prompt = input("You: ")
            prompt_for_action = f"""
            This is a classification task. Return either of the following three values, with respect to the following user prompt and wordview_functions below:\n\
                1. A function suggestion from among wordview_functions keys by reading their description and relating to the user_prompt. \n\
                2. The term "continuation" if user prompt is a follow up on a previous prompt. \n\
                3. The term "unknown" if you cannot make a suggestion with high confidence, or the prompt is not a continuation.\n\
            \n
            ------------------
            user_prompt:
            {user_prompt}
            \n
            ------------------
            wordview_functions:
            {self.wordview_functions}
            """

            gpt_action_suggestion = (
                openai.Completion.create(
                    engine="davinci", prompt=prompt_for_action, max_tokens=100
                )
                .choices[0]
                .text.strip()
            )

            if gpt_action_suggestion == "unknown":
                print(
                    "ChatGPT: I'm not sure what you're asking. Please provide more context or try a different prompt."
                )
                continue

            if gpt_action_suggestion == "continuation":
                gpt_response = (
                    openai.Completion.create(
                        engine="davinci", prompt=user_prompt, max_tokens=100
                    )
                    .choices[0]
                    .text.strip()
                )
                history += f"\nUser: {user_prompt}\nChatGPT: {gpt_response}\n"

            if gpt_action_suggestion in self.wordview_functions.keys():
                history += f"\nUser: {user_prompt}\nChatGPT: {gpt_action_suggestion}\n"
                if gpt_action_suggestion == "multiword_expressions":
                    pass
