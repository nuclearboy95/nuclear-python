import requests
import json


__all__ = ['get']


def get(url, headers=None, params=None):
    try:
        response = requests.get(url, headers=headers, params=params)
    except requests.ConnectTimeout:
        raise
    except requests.ReadTimeout:
        raise
    except requests.ConnectionError:
        raise
    except requests.exceptions.ChunkedEncodingError:
        raise
    except ConnectionResetError:
        raise
    except:
        raise

    response_code = response.status_code
    return json.loads(response.text)
