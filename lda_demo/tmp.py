import requests

ret = requests.post('http://127.0.0.1:8888/cangjie_gateway/cargo_filter_web', data={'sentence': '\\t'})
print(ret)
