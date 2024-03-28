import pandas as pd
from openai import OpenAI

from wordview.text_analysis import TextStatsPlots


class Datachat:
    def __init__(
        self,
        api_key: str = "",
        dataframe: pd.DataFrame = pd.DataFrame([]),
        text_column: str = "",
    ):
        self.api_key = api_key
        self.dataframe = dataframe
        self.text_column = text_column
        self.classifier_client = OpenAI(api_key=credentials.get("openai_api_key"))
        self.chat_client = OpenAI(api_key=credentials.get("openai_api_key"))

    def wordview_chat(self):
        chat_history = []
        while True:
            response = ""
            user_prompt = input("You: ")
            chat_history.append({"role": "user", "content": user_prompt})

            prompt_for_action_classification = f"""Classify "user prompt" below into one of the following categories:
            [multiword_expressions, bias_detection, text_analysis, unknown, continuation].

            Only return the class name without any extra explanation, words or tokens in your response.

            Use the class label continuation if the user prompt is a follow up on a previous prompt.

            Use the class label unknown if you cannot make a suggestion with high confidence, or the prompt is not a continuation.

            "user prompt":
            {user_prompt}
            """
            gpt_action_suggestion = (
                self.classifier_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt_for_action_classification},
                    ],
                )
                .choices[0]
                .message.content
            )

            if gpt_action_suggestion == "unknown":
                print(
                    "Datachat: I'm not sure what you're asking. Please provide more context or try a different prompt."
                )
                continue

            if gpt_action_suggestion == "continuation":
                response = (
                    self.chat_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=chat_history,
                    )
                    .choices[0]
                    .message.content
                )
                chat_history.append({"role": "assistant", "content": response})
                print(f"Datachat: {response}")
                continue

            if gpt_action_suggestion == "multiword_expressions":
                print("Datachat: I think you're asking about multiword expressions.")
                continue

            if gpt_action_suggestion == "bias_detection":
                print("Datachat: I think you're asking about bias detection.")
                continue

            if gpt_action_suggestion == "text_analysis":
                tsp = TextStatsPlots(df=self.dataframe, text_column=self.text_column)
                corpus_stats = tsp.return_stats()
                prompt = f"""Explain the following dictionary of text analysis statistics extracted from corpus in Natural Language:

                ---------------------------------------
                dictionary of text analysis statistics:
                {corpus_stats}
                """
                response = (
                    self.chat_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                    )
                    .choices[0]
                    .message.content
                )
                print(response)
                continue


if __name__ == "__main__":
    import json

    imdb_df = pd.read_csv("data/IMDB_Dataset_sample_5k.csv")
    with open("wordview/chat/secrets/openai_api_key.json", "r") as f:
        credentials = json.load(f)
    datachat = Datachat(
        api_key=credentials["openai_api_key"], dataframe=imdb_df, text_column="review"
    )
    datachat.wordview_chat()
