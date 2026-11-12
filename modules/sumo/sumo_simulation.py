import traci
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import traceback


class SumoSimulation():
    def __init__(self,
                 cfg_file,
                 output_dir,
                 tl_ids,
                 ):
        
        self.cfg_file=cfg_file
        self.output_dir=output_dir
        self.tl_ids=tl_ids

    def simulation(self,epoch_index,rep,demand_scaling_factor=None):

        if demand_scaling_factor is not None:
            command=[
                    "sumo",
                    "-c", f"./{self.cfg_file}",
                    "--scale",f"{demand_scaling_factor}",
                    "--verbose","--seed", f"{rep}",
                    "--output-prefix", f"{self.output_dir}{epoch_index}_{rep}_"
                    ]
        else:
            command=[
                    "sumo",
                    "-c", f"./{self.cfg_file}",
                    "--verbose","--seed", f"{rep}",
                    "--output-prefix", f"{self.output_dir}{epoch_index}_{rep}_"
                    ]
        try:
            traci.start(command)
            for tl in self.tl_ids:
                traci.trafficlight.setProgram(tl, "1")
                program = traci.trafficlight.getAllProgramLogics(tl)
                program_used=traci.trafficlight.getProgram(tl)

            
            traci.simulationStep(61200)

        except Exception as e:
            print(f"Erreur SUMO/TraCI : {e}")
            traceback.print_exc()

        finally:
            traci.close()

        df_sim=pd.read_xml(f"./Ingolstadt Network/{self.output_dir}{epoch_index}_{rep}_InTAS.travel_times.xml",iterparse={"edge":["id","sampledSeconds","traveltime",\
                                                                                                                   "density","occupancy","speed","waitingTime",\
                                                                                                                    "timeLoss","departed","arrived","entered","left",\
                                                                                                                        "flow"]}).rename(columns={"id":"edge_id"})
        if df_sim.shape[0]!=len(df_sim["edge_id"].unique()):
            raise ValueError('This code asssumes there is  a single sampling time interval in the edge travel time xml file. If an edge id appears multiple times it is because there are multiple sampling intervals')

        df_sim.to_csv(f"{self.output_dir}{epoch_index}_{rep}_InTAS.travel_times.csv")

        
                
    
