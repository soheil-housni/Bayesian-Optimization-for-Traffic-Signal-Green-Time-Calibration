
import numpy as np
import sumolib
import pandas as pd


class CompleteCorridorsConnexions():

    def __init__(self,
                route_files_list,
                net_file,
                corridors_df,
                ):

        self.net=sumolib.net.readNet(net_file,withPrograms=True)

        self.routes_df=None
        for route_file in route_files_list:
            sub_routes_df=pd.read_xml(route_file, iterparse={"route":["edges","probability"]}).rename(columns={"edges":"route_edges"})
            if self.routes_df is None:
                self.routes_df=sub_routes_df
            else:
                self.routes_df=pd.concat([self.routes_df,sub_routes_df],axis=0)

        self.corridors_df=corridors_df


    def create_connexion_corridors_df(self):
        connections_df=self.creation_connexion_df()
        corridors_df_merged=pd.merge(self.corridors_df,connections_df,how="left",on="route_last_edge")
        corridors_df_merged["green_phases"]=corridors_df_merged.apply(self.extract_green_phases,args=(self.net,),axis=1)
        print(corridors_df_merged.columns)
        connexions_demand_df=self.extract_connexion_demand(corridors_df_merged)
        connexion_corridors_df=pd.merge(corridors_df_merged,connexions_demand_df,how="left",on=["tl","route_str","route_last_edge","to_edge","link_index"])
        route_connexion_demand_df=connexion_corridors_df.groupby("route_str").agg({"connexion_demand":"sum"}).reset_index().rename(columns={"connexion_demand":"route_connexion_demand"})
        connexion_corridors_df=pd.merge(connexion_corridors_df,route_connexion_demand_df,on="route_str",how="left")
        connexion_corridors_df=connexion_corridors_df.reset_index().rename(columns={"index":"connexion_id"})
        return connexion_corridors_df


    def extract_connexion_demand(self,corridors_df):
        connexions_demand_dict=[]
        for index,row in corridors_df[["route","tl","route_str","route_last_edge","to_edge","link_index"]].drop_duplicates(subset=["tl","route_str","route_last_edge","to_edge","link_index"]).iterrows():
            complete_route=row["route"]+[row["to_edge"]]
            complete_route_str_str=" ".join(complete_route)
            connexion_demand=self.routes_df.loc[self.routes_df["route_edges"].str.contains(complete_route_str_str,regex=False),"probability"].sum()
            connexions_demand_dict.append({
                "tl":row["tl"],
                "route_str":row["route_str"],
                "route_last_edge":row["route_last_edge"],
                "to_edge":row["to_edge"],
                "link_index":row["link_index"],
                "connexion_demand":connexion_demand,
            })

        connexions_demand_df=pd.DataFrame(connexions_demand_dict)
        return connexions_demand_df

    def extract_green_phases(self,row,net):
        from_edge=row["route_last_edge"]
        to_edge=row["to_edge"]
        tl=row["tl"]
        linkIndex=row["link_index"]
        traffic_light = net.getTLS(tl)
        green_phases=[]
        for program_id, program in traffic_light.getPrograms().items():
            for i, phase in enumerate(program.getPhases()):
                phase_state=phase.state
                linkIndexPhaseElem=phase_state[linkIndex]
                if (linkIndexPhaseElem=="G") or (linkIndexPhaseElem=="g"):
                    green_phases.append(phase_state)

        return green_phases


    def creation_connexion_df(self):
        last_edges_list=[]
        to_edges_list=[]
        link_index_list=[]

        for edge_id in self.corridors_df["route_last_edge"].unique():
            edge = self.net.getEdge(edge_id)

            to_edges = edge.getOutgoing()

            for to_edge in to_edges:
                to_edge_id = to_edge.getID()

                connections=edge.getConnections(to_edge)
                for connection in connections:
                    if connection.getFrom().getID()==edge_id:
                        link_index = connection.getTLLinkIndex()

                        last_edges_list.append(edge_id)
                        to_edges_list.append(to_edge_id)
                        link_index_list.append(link_index)

        connections_df=pd.DataFrame({"route_last_edge":last_edges_list,"to_edge":to_edges_list,"link_index":link_index_list})
        return connections_df