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
    # payload = dict(
    #     name='get_latest_day',
    #     nd=user_id,
    #     ctx=dict(
    #         before_date=before_date,
    #         show_report=1  # this must be 1 or otherwise the response will be empty
    #     )
    # )

    model = 'text-ada-001'  # fastest
    # model = 'text-davinci-002'  # most powerful
    res = requests.get(url='https://api.openai.com/v1/models', headers=headers)
    mic(res)
    res = json.loads(res.text)
    mic(res)
