import traci
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


class SumoSimulation():
    def __init__(self,
                 num_replications,
                 cfg_file,
                 output_dir,
                 tl_ids,
                 gt=False):
        
        self.num_replications=num_replications
        self.cfg_file=cfg_file
        self.output_dir=output_dir
        self.tl_ids=tl_ids

    def simulation(self,epoch_index):
        df_all=None

        for s1 in range(self.num_replications):
            try:
                traci.start([
                            "sumo",
                            "-c", f"./{self.cfg_file}",
                            "--verbose","--seed", f"{s1}",
                            "--output-prefix", f"{self.output_dir}{epoch_index}_{s1}_"
                            ])
                for tl in self.tl_ids:
                    traci.trafficlight.setProgram(tl, "1")
                    program = traci.trafficlight.getAllProgramLogics(tl)
                    program_used=traci.trafficlight.getProgram(tl)
                    print(program)
                    print(f"Program used : {program_used}")

                for i in range(3600):
                    traci.simulationStep()
            finally:
                traci.close()

            df_sim=pd.read_xml(f"./Ingolstadt Network/{self.output_dir}{epoch_index}_{s1}_InTAS.travel_times.xml",iterparse={"edge":["id","sampledSeconds","traveltime",\
                                                                                                                   "density","occupancy","speed","waitingTime",\
                                                                                                                    "timeLoss","departed","arrived","entered","left",\
                                                                                                                        "flow"]}).rename(columns={"id":"edge_id"})
            if df_sim.shape[0]!=len(df_sim["edge_id"].unique()):
                raise ValueError('This code asssumes there is  a single sampling time interval in the edge travel time xml file. If an edge id appears multiple times it is because there are multiple sampling intervals')

            df_sim.to_csv(f"{self.output_dir}{epoch_index}_{s1}_InTAS.travel_times.csv")

            if df_all is None:
                df_all=df_sim
            else:
                df_all=pd.concat([df_all,df_sim],axis=0)

        df_all_agg=df_all.groupby("edge_id").agg({"sampledSeconds":['mean', 'std'],"traveltime":['mean', 'std'],\
                                                    "density":['mean', 'std'],"occupancy":['mean', 'std'],"speed":['mean', 'std'],"waitingTime":['mean', 'std'],\
                                                    "timeLoss":['mean', 'std'],"departed":['mean', 'std'],"arrived":['mean', 'std'],"entered":['mean', 'std'],"left":['mean', 'std'],\
                                                       "flow":['mean', 'std']}).reset_index()
        df_all_agg=df_all_agg.droplevel(1,axis=1)
        new_column_names=["edge_id"]
        for i,col in enumerate(df_all_agg.columns[1:]):
            if i%2==0:
                new_column_names.append(col+"_mean")
            else:
                new_column_names.append(col+"_std")

        df_all_agg.columns=new_column_names
        df_all_agg.to_csv(f"{self.output_dir}{epoch_index}_InTAS.travel_times.csv")

        
                
    
