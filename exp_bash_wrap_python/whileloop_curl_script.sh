#!/bin/bash

FILE_NAME="queries.txt"


function save_log()
{
  printf '%s\n' \
    "Header Code  : $1" \
	"Query : $3" \
    "Executed at  : $(date)" \
    "Response Body  : $2" \
    "====================================================================================================="$'\r\n\n'  >> output.log
}


while IFS= read -r line || [[ -n "$line" ]]; 
    do
	  echo "user query ---> "
	  echo $line	
	  #jo_message=$( jo "role=user" "content=$line")	  
	  #json_w_vars=$( jo "config_id=vm" "messages=[$jo_message]")
	  
	  echo $json_w_vars
	  result=$(python3 curl2request.py $line )
	  echo "response from chatbot ---> "
	  echo $result
	  echo " ############### break for 5 secs ############### "
	  #curl --location 'https://api-prod.nvidia.com/gtc-chatbot/v1/chat/completions' \
	  #--header 'Content-Type: application/json' \
	  #--data $json_w_vars
      #HTTP_STATUS=$(echo $HTTP_RESPONSE | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
      #echo $HTTP_RESPONSE
      #save_log "$HTTP_STATUS" "$HTTP_BODY" "$line"
	  sleep 15
done < $FILE_NAME
