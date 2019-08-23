import requests
import time
for i in range(100):
    url = 'http://dev.ymm56.com/cangjie_gateway/action_sequence?userId=600608457567166602'
    t1 = time.time()
    res = requests.get(url)
    print(time.time() - t1)
# print(res.content)