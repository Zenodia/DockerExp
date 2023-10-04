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
    
    response = requests.post('nemollm_api_calls', headers=headers, json=json_data)
    if response.status_code == 200:
        output=response.json()
    else:
        output=str(response.status_code)
    return output
if __name__ == '__main__':
    f=open('queries.txt','r')
    lines=f.readlines()
    k=len(lines)
    i=0
    for line in lines:
        if i <= k: 
            usr_query=line
            print("user query is : \n", usr_query)
            response= post_requests(usr_query)
            print("server response is :\n", response)
            print("-----"*10 )
        else:break
        i+=1

        
    #usr_query=str(sys.argv[1])
    