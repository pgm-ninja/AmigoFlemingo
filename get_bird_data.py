import requests
from datetime import date, timedelta
import os
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv('EBIRD_API_KEY')
region_code = "IN-KL"
# start_date = date(2025, 1, 1)
# end_date = date(2025, 4, 26)

birds_names = set()

headers = {
    'X-eBirdApiToken': api_key
}



# url = f"https://api.ebird.org/v2/data/obs/{region_code}/recent/bcrthr1"
# url = f"https://api.ebird.org/v2/data/obs/{region_code}/recent/notable?detail=full"
# url = f"https://api.ebird.org/v2/data/obs/{region_code}/historic/2025/01/01"
url = f"https://api.ebird.org/v2/data/obs/{region_code}/recent"

payload={}
headers = {
  'X-eBirdApiToken': api_key
}

response = requests.request("GET", url, headers=headers, data=payload)

response_json = response.json()

for data in response_json:
    common_name = data['comName']
    birds_names.add(common_name)


import os

for bird in birds_names:
    folder_name = bird.lower().replace(" ", "_")
    folder_path = 'new_birds'
    full_path = os.path.join(folder_path, folder_name)
    check_path = os.path.join('dataset', folder_name)

    if not os.path.exists(check_path):  # Corrected check
        os.makedirs(full_path)
        file_path = os.path.join(full_path, 'description.txt')
        with open(file_path, 'a') as file_:
            pass
