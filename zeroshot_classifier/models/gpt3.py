import requests
from os.path import join as os_join

from stefutil import *
from zeroshot_classifier.util import *


if __name__ == '__main__':
    import json
    with open(os_join(u.proj_path, 'auth', 'open-ai.json')) as f:
        d = json.load(f)
        api_key, org = d['api-key'], d['organization']

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'OpenAI-Organization': org
    }

    def check_api():
        res = requests.get(url='https://api.openai.com/v1/models', headers=headers)
        mic(res)
        res = json.loads(res.text)
        mic(res)
    # check_api()

    def try_completion():
        model = 'text-ada-001'  # fastest
        # model = 'text-davinci-002'  # most powerful

        payload = dict(
            model=model,
            prompt="Say this is a test",
            max_tokens=6,  # Generate w/ greedy decoding
        )
    try_completion()
