import requests

url = "https://www.csie.ntu.edu.tw/%7Ecjlin/libsvmtools/datasets/regression/cpusmall_scale"
filename = "cpusmall_scale"

response = requests.get(url)

with open(filename, 'wb') as file:
    file.write(response.content)

print(f"Downloaded {filename}")