import pandas as pd
import os



df = pd.DataFrame()
d = {1:2}
import time
def wait(fp):
    while not os.path.exists(fp):
        print("waiting creation")
        time.sleep(1)
    s = os.path.getsize(fp)
    time.sleep(.2)
    while (s != os.path.getsize(fp)):
        s= os.path.getsize(fp)
        print("waiting done writing")
        time.sleep(.1)
    return

if __name__ == "__main__":
    print('ok')
