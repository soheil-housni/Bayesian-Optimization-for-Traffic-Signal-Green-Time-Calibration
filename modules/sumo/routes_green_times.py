import numpy as np
import sumolib
import pandas as pd

class GetRouteGreenTime():
    def __init__(self,
                 net_file,
                 initial_tl_df,
                 ):
        
        self.net=sumolib.net.readNet(net_file,withPrograms=True)
        self.initial_tl_df=initial_tl_df

    def get_route_green_time(self,
                             corridors_df):
        corridors_df["green_phases_info"]=corridors_df.apply(self.extract_route_green_phases,axis=1)
        corridors_df["route_green_phases"]=corridors_df["green_phases_info"].str[0]
        corridors_df["route_green_link_indices"]=corridors_df["green_phases_info"].str[1]

        corridors_df["main_green_phase_info"]=corridors_df.apply(self.extract_main_green_phase,axis=1)
        corridors_df["main_green_phase"]=corridors_df["main_green_phase_info"].str[0]
        corridors_df["main_green_phase_duration"]=corridors_df["main_green_phase_info"].str[1]
        corridors_df["main_green_phase_to_change"]=corridors_df["main_green_phase_info"].str[2]
        corridors_df["main_green_phase_initial_duration"]=corridors_df["main_green_phase_info"].str[3]

        return corridors_df

    def extract_main_green_phase(self,row):
        route_green_phases=row["route_green_phases"]
        tl=row["tl"]
        initial_tl_df_copy=self.initial_tl_df.copy()
        initial_tl_df_filtered=initial_tl_df_copy[(initial_tl_df_copy["tl"]==tl) & (initial_tl_df_copy["state"].isin(route_green_phases))]
        main_green_phase=initial_tl_df_filtered.loc[initial_tl_df_filtered["initial_duration"]==initial_tl_df_filtered["initial_duration"].max()]["state"].iloc[0]
        main_green_phase_duration=initial_tl_df_filtered.loc[initial_tl_df_filtered["initial_duration"]==initial_tl_df_filtered["initial_duration"].max()]["initial_duration"].iloc[0]
        main_green_phase_to_change=initial_tl_df_filtered.loc[initial_tl_df_filtered["initial_duration"]==initial_tl_df_filtered["initial_duration"].max()]["to_change"].iloc[0]
        initial_duration=initial_tl_df_filtered.loc[(initial_tl_df_filtered["tl"]==tl) & (initial_tl_df_filtered["state"]==main_green_phase)]["initial_duration"].iloc[0]
        return (main_green_phase,main_green_phase_duration,main_green_phase_to_change,initial_duration)
        

    def extract_route_green_phases(self,row):
        route_last_edge_id=row["route_last_edge"]
        tl=row["tl"]

        route_last_edge=self.net.getEdge(route_last_edge_id)
        traffic_light = self.net.getTLS(tl)
        to_edges=route_last_edge.getOutgoing()

        green_link_indices=[]

        for to_edge in to_edges:
            connections=route_last_edge.getConnections(to_edge)
            for connection in connections:
                if (connection.getFrom().getID()==route_last_edge_id) and (connection.getTLSID() == tl):
                    link_index = connection.getTLLinkIndex()
                    green_link_indices.append(link_index)

        route_green_phases=[]

        for program_id, program in traffic_light.getPrograms().items():
            for i, phase in enumerate(program.getPhases()):
                phase_state=phase.state
                for link_index in green_link_indices:
                    linkIndexPhaseElem=phase_state[link_index]
                    if (linkIndexPhaseElem=="G") or (linkIndexPhaseElem=="g"):
                        route_green_phases.append(phase_state)
                        break

        return (route_green_phases,green_link_indices)