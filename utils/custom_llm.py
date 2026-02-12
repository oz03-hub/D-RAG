from openai import OpenAI, RateLimitError
import backoff
import tqdm
import concurrent.futures as futures

def batchify(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


class TextHelper:
    def __init__(self, text):
        self.text = text
        self.cumulative_logprob = 0

class OutputHelper:
    def __init__(self, output):
        self.outputs = [TextHelper(output)]

@backoff.on_exception(backoff.expo, RateLimitError)
def get_completion_from_gpt(messages, model_name, max_tokens, api_key, temperature):
    client = OpenAI(api_key=api_key)
    retries = 0
    while True:
        retries += 1    
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

class OpenAILLM:
    def __init__(self, model_name, api_key) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.n_threads=8
    
    def generate(self, prompts, sampling_param):
        results = []
        barches = batchify(prompts, self.n_threads)
        for batch in tqdm.tqdm(barches):
            temp_results = []
            with futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                for data in batch:
                    temp_results.append(executor.submit(get_completion_from_gpt, data, self.model_name, sampling_param.max_tokens, self.api_key, sampling_param.temperature))
            for temp_result in temp_results:
                results.append(OutputHelper(temp_result.result()))
        return results