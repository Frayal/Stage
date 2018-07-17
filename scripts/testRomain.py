from romain import df, wait
import pandas as pd
from threading import Thread
import time
import numpy as np
def write():
    time.sleep(2)
    df = pd.Series(np.arange(10**9))
    print("start writing")
    with open("truc","w") as file:
        for i in range(20000):
            file.write(str(i))
            file.flush()
            time.sleep(.2)
    print("done")

if __name__ == "__main__":
    t = Thread(target=write)
    t.start()
    wait("truc")
