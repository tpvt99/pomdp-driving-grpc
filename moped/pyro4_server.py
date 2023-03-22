#!/home/cunjun/anaconda3/envs/conda38/bin/python


import Pyro4
import numpy as np  # Your motion prediction service module

@Pyro4.expose
class MotionPredictionService(object):
    def predict(self, data):
        output = np.random.rand()
        print(f"Receiving {data} Output to return is {output}")
        return output

def main():
    Pyro4.Daemon.serveSimple(
        {
            MotionPredictionService: "motion_prediction_service",
        },
        #host="0.0.0.0",
        port=5001,
        ns=False,
    )

if __name__ == "__main__":
    main()
