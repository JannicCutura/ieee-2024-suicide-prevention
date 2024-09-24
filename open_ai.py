import os

import openai
import pandas as pd
from loguru import logger
from openai import OpenAI
from pathlib import Path

client = OpenAI(api_key='myjj8yogHDAurY2rgnkQJFkblB3TMTNhkS4uM8Nq0vSnlzLn-jorp-ks'[::-1])


class Paths:
    _current_file_path = Path(os.path.abspath(__name__))
    ROOT_PATH = _current_file_path.parent


def prompt_engineering(data: pd.DataFrame, client, model_name: str, system_prompt: str):
    logger.info("Submitting to OpenAI")
    posts = data.reset_index(names="original_index")

    responses = dict()
    for index, post in zip(posts['original_index'], posts['post']):
        logger.info(f"processing {index + 1}/{len(posts)}")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": post}],
                temperature=0,
                n=1,
                max_tokens=50
            )
            responses[index]: openai.ChatCompletion = response
        except:
            pass

    predictions = dict()

    logger.info("Parsing responses")
    for index, response in responses.items():
        try:
            prediction = response.choices[0].message.content
            predictions[index] = prediction
        except:
            pass

    posts['pred'] = posts['original_index'].map(predictions)

    return posts


def clean_result(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("cleaning responses")
    labels = ['indicator', 'behavior', 'attempt', 'ideation']
    df['pred_original'] = [x for x in df['pred'].values]
    df['pred'] = df['pred'].str.lower()
    df['pred'] = df['pred'].str.replace(r'[^a-zA-Z0-9]', '', regex=True)
    replacements = {'behaviour': 'behavior',
                    ",": "",
                    "'": ""}

    df['pred'] = df['pred'].replace(replacements)
    pred_labels = list(set(df['pred'].to_list()))
    unknowns = [label for label in pred_labels if label not in labels]
    if unknowns:
        logger.warning(f"There were unknowns responses: {unknowns}, replacing with behavior")
        def replace_incorrect_ones(category: str) -> str:
            """last resort defense against incorrect open ai replies"""
            if category not in labels:
                return "behaviour"
            return category

        df['pred'] = df['pred'].apply(replace_incorrect_ones)

    return df


def add_probs(df: pd.DataFrame) -> pd.DataFrame:
    def add_prob(category: str) -> list[int]:
        if category == "indicator":
            return [1, 0, 0, 0]
        elif category == "ideation":
            return [0, 1, 0, 0]
        elif category == "behavior":
            return [0, 0, 1, 0]
        elif category == "attempt":
            return [0, 0, 0, 1]

    df['probs'] = df['pred'].apply(add_prob)

    return df


def validate_input_data(file_path: Path) -> pd.DataFrame:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Please put the holdout/evaluation dataset to this path: {file_path}")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise RuntimeError(f"File is available at{file_path} but could not be read. ")

    if 'index' in df.columns:
        del df['index']

    if 'post' not in df.columns:
        raise ValueError(f"The data is missing the column 'post'. It only contains the columns {list(df.columns)}. "
                         f"Please rename the respective colum, as it was named in the training data")


    return df


PROMPT = """
       Based on the reddit post provided in the user prompt, return the category that the post belongs to of the following four categories and their definitions.
    ---
     Indicator:
    Definition: The post or content lacks any explicit expression concerning suicide and at best hints at it. There may be general expressions of extreme distress or sadness, but no direct mention or indication of suicidal thoughts or feelings. May include also a vague feeling about dying or death but no explicit mentioning of suicide. Note that the mentioning of suicide needs to be very explicit in order to not be an ‘indicator'. Very large general dissatisfaction about life and hopefulness is still 'indicator' when suicide/killing oneself is not mentioned. When in doubt whether the suicide was explicitly mentioned or not choose 'indicator' over 'ideation'.

    Ideation:
    Definition: The content includes explicit expressions of suicidal thoughts or feelings, i.e. a desire to not be alive any longer, but without any specific plan to commit suicide. This can range from vague thoughts about not wanting to live to stronger, more explicit desires to end one’s life (albeit without a specific plan how to do so). Posts where an explicit wish to be dead is described, should be  If no desire to die or to commit suicide is expressed, consider it 'indicator'. Also, statements denying the intention to commit suicide ('I won’t commit suicide/do it') should be considered 'indicator'. If the post contains specific ideas of how to commit suicide (ways/locations/means/methods) consider it 'behavior'.  

    Behavior:
    Definition: The post includes explicit ideas / refined plans how to commit suicide. It must include some form of explicit planning like a specific method, or preparations taken (e.g. suicide note, lethal medication/drugs, tools/weapons (e.g. knifes/guns/ropes) suitable to end one’s life, suitable locations (e.g. bridges/cliffs/buildings to jump off from, train lines to get run over by). Otherwise, unhealthy or depressed behavior not suitable to ends one life (cutting, starving, etc.) does not classify as 'behavior'.  If you are unsure whether there is an explicit plan or not choose 'ideation' over 'behavior'. 

    Attempt:
    Definition: The content describes past attempts at suicide. This category is focused on historic actions rather than current plans. A concrete suicide attempt needs to have happened in the past (e.g. overdose). When someone merely thought of an attempt in the past this classifies not as an 'attempt', but as 'behavior'.
    Note that when a post refers to past attempts (e.g. I tried again) but also mentions current plans it should be labeled as 'attempt'.
    ---
    Note that the suicide risk only corresponds to the person writing the post not of other people potentially being mentioned.
       Only answer with one word. It should be always one of the following 'indicator', 'ideation', 'behavior', 'attempt'   
       Never answer with something different than one of the four options. Never answer with NaN or empty answer
       """

if __name__ == "__main__":
    file_path = Paths.ROOT_PATH / "evaluation_data.xlsx"

    model_name = "gpt-4-turbo"

    data = validate_input_data(file_path).head(5)

    df = prompt_engineering(
        data=data,
        client=client,
        model_name=model_name,
        system_prompt=PROMPT)

    df = clean_result(df)
    df = add_probs(df)
    submission_file = Paths.ROOT_PATH / "Calculators.xlsx"
    logger.info(f"Persisting output file to {submission_file}")
    df.filter(['pred', 'probs']).rename(columns={
        'pred': 'suicide risk',
        'probs': 'probability distribution'
    }).to_excel(submission_file, index=True, index_label="index")
