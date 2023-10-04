import requests
import sys, os

def post_requests(usr_query):
    headers = {
        'Content-Type': 'application/json',
    }

    json_data = {
        'config_id': 'vm',
        'messages': [
            {
                'role': 'user',
                'content': usr_query,
            },
        ]}
    
    response = requests.post('nemollm_service_api_call', headers=headers, json=json_data)
    return response

if __name__ == '__main__':
    usr_query=str(sys.argv[1])
    resp =post_requests(usr_query)
    if resp == "<Response [200]>":
        out=resp.json()
        print(out['content'])
    else:
        print("some error --- ", resp)