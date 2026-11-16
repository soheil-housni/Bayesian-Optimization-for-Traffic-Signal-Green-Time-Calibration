
import sumolib
import pandas as pd
import numpy as np
from .corridors_extraction import CorridorExtractor

class TLIncomingCorridors():

    def __init__(self,
                 net_file,
                 route_files_list,
                 ):

        self.net = sumolib.net.readNet(net_file,withInternal=False)

        self.corridors_extractor = CorridorExtractor(self.net)

        self.routes_df=None
        for route_file in route_files_list:
            sub_routes_df=pd.read_xml(route_file, iterparse={"route":["edges","probability"]}).rename(columns={"edges":"route_edges"})
            if self.routes_df is None:
                self.routes_df=sub_routes_df
            else:
                self.routes_df=pd.concat([self.routes_df,sub_routes_df],axis=0)

    def get_corridors_df(self):
        print(self.routes_df.shape)
        corridors_df=self.creation_tl_corridors_dataframe()
        print("CORRIDORS DF GENERATED")
        
        routes_demand_df=self.extract_route_demand(corridors_df)
        print("ROUTES DEMAND EXTRACTED")
        corridors_df=pd.merge(corridors_df,routes_demand_df,how="left",on="route_str")
        tl_demand=corridors_df.drop_duplicates(["tl","route_str"]).groupby("tl").agg({"route_demand":"sum"}).reset_index().rename(columns={"route_demand":"tl_demand"})
        corridors_df=pd.merge(corridors_df,tl_demand,how="left",on="tl")
        corridors_df=corridors_df.loc[corridors_df["route_demand"]>0]

        
        corridors_df=self.get_n_route_edges(corridors_df)
        corridors_df["add_info"]=corridors_df.apply(self.get_n_previous_tl,args=(corridors_df,),axis=1)
        corridors_df["n_previous_tl"]=corridors_df["add_info"].str[0]
        corridors_df["n_previous_routes"]=corridors_df["add_info"].str[1]
        corridors_df["n_previous_edges"]=corridors_df["add_info"].str[2]
        corridors_df["previous_demand"]=corridors_df["add_info"].str[3]


        corridors_df["next_info"]=corridors_df.apply(self.to_edges,args=(self.net,corridors_df,),axis=1)
        corridors_df["direct_next_edges"]=corridors_df["next_info"].str[0]
        corridors_df["direct_next_tls"]=corridors_df["next_info"].str[1]

        corridors_df["from_info"]=corridors_df.apply(self.from_edges,args=(self.net,corridors_df,),axis=1)
        corridors_df["direct_from_edges"]=corridors_df["from_info"].str[0]
        corridors_df["direct_previous_tls"]=corridors_df["from_info"].str[1]

        corridors_df["direct_next_demand"]=corridors_df.apply(self.extract_next_demand,args=(corridors_df,),axis=1)

        corridors_df["direct_previous_demand"]=corridors_df.apply(self.extract_previous_demand,args=(corridors_df,),axis=1)

        corridors_df["direct_from_n_junction_inc_lanes"]=corridors_df.apply(self.extract_junction_lanes,axis=1)

        corridors_df["route_id"]=1
        for tl in corridors_df["tl"].unique():
            corridors_df.loc[corridors_df["tl"]==tl,"route_id"]=[i for i in range(len(corridors_df.loc[corridors_df["tl"]==tl]))]

        

        return corridors_df


    def extract_junction_lanes(self,row):
        route=row["route"]
        n_junction_inc_lanes=0

        for edge_id in route[:-1]:
            edge=self.net.getEdge(edge_id)
            to_junction=edge.getToNode()

            n_incoming_lanes = np.sum(
            [len(edge.getLanes())
            for edge in to_junction.getIncoming()]
            )

            n_junction_inc_lanes+=n_incoming_lanes

        return n_junction_inc_lanes

    def extract_previous_demand(self,row,corridors_df):
        previous_demand=0
        curr_tl=row["tl"]
        previous_tls=row["direct_previous_tls"]
        for tl in previous_tls:
            if tl!=curr_tl:
                previous_demand+=corridors_df.loc[(corridors_df["origin"]==tl),"route_demand"].sum()

        return previous_demand

    def from_edges(self,row,net,corridors_df):
        from_edges_list=[]
        from_tls_list=[]
        edge_id=row["route_first_edge"]
        edge=net.getEdge(edge_id)
        for from_edge in edge.getIncoming():
            from_edge_id=from_edge.getID()
            from_tls=list(corridors_df.loc[corridors_df["route_last_edge"]==from_edge_id,"origin"].unique())
            from_tls_list.append(from_tls)
            from_edges_list.append([from_edge_id]*len(from_tls))

        from_edges_list=[elem for sublist in from_edges_list for elem in sublist]
        from_tls_list=[elem for sublist in from_tls_list for elem in sublist]

        return from_edges_list,from_tls_list

    def extract_next_demand(self,row,corridors_df):
        next_demand=0
        curr_tl=row["tl"]
        next_tls=row["direct_next_tls"]
        for tl in next_tls:
            if tl!=curr_tl:
                next_demand+=corridors_df.loc[(corridors_df["tl"]==tl),"route_demand"].sum()

        return next_demand

    def to_edges(self,row,net,corridors_df):
        to_edges_list=[]
        next_tls_list=[]
        edge_id=row["route_last_edge"]
        edge=net.getEdge(edge_id)
        for to_edge in edge.getOutgoing():
            to_edge_id=to_edge.getID()
            next_tls=list(corridors_df.loc[corridors_df["route_first_edge"]==to_edge_id,"tl"].unique())
            next_tls_list.append(next_tls)
            to_edges_list.append([to_edge_id]*len(next_tls))

        to_edges_list=[elem for sublist in to_edges_list for elem in sublist]
        next_tls_list=[elem for sublist in next_tls_list for elem in sublist]

        return to_edges_list,next_tls_list


    def extract_route_demand(self,corridors_df):
        routes_demand_dict={}
        for index,row in corridors_df[["route_str","route_str_str"]].drop_duplicates().iterrows():
            route_demand=self.routes_df.loc[self.routes_df["route_edges"].str.contains(row["route_str_str"],regex=False),"probability"].sum()
            routes_demand_dict[row["route_str"]]=route_demand

        routes_demand_df=pd.DataFrame({"route_str":[key for key in routes_demand_dict.keys()],"route_demand":[value for value in routes_demand_dict.values()]})
        return routes_demand_df


    def get_n_previous_tl(self, row, corridor_df):
        base_tl=row["tl"]
        origin = row["origin"]

        visited = {base_tl,origin}
        previous_tl = set()
        origins_to_visit = []
        if origin!="dead_end":
            previous_tl.add(origin)
            origins_to_visit.append(origin)

        n_previous_routes = 0
        n_previous_edges = 0
        previous_demand=0

        def get_previous(origins):

            nonlocal n_previous_routes
            nonlocal n_previous_edges
            nonlocal previous_demand

            filtered_df = corridor_df[
                (corridor_df["tl"]==origins) & (~corridor_df["origin"].isin(visited))
            ]

            n_previous_routes += len(filtered_df)
            n_previous_edges += filtered_df["route_n_edges"].sum()
            previous_demand +=filtered_df["route_demand"].sum()

            previous_origins = filtered_df.loc[
                filtered_df["origin"] != "dead_end",
                "origin"
            ].unique()

            return previous_origins

        while origins_to_visit:

            current_origins = origins_to_visit.pop()

            previous_origins = get_previous(current_origins)

            for previous_origin in previous_origins:

                if previous_origin not in visited:

                    visited.add(previous_origin)
                    previous_tl.add(previous_origin)
                    origins_to_visit.append(previous_origin)

        return [
            len(previous_tl),
            n_previous_routes,
            n_previous_edges,
            previous_demand
        ]

    def get_n_route_edges(self,corridors_df):
        n_routes_agg_df=corridors_df.drop_duplicates(["tl","route_str"]).groupby("tl").agg({"route_str":"count"}).rename(columns={"route_str":"tl_n_routes"})
        n_edges_agg_df=corridors_df.drop_duplicates(["tl","route_str"]).groupby("tl").agg({"route_n_edges":"sum"}).rename(columns={"route_n_edges":"tl_n_edges"})
        corridors_df=pd.merge(pd.merge(corridors_df,n_routes_agg_df,how="left",on="tl"),n_edges_agg_df,how="left",on="tl")
        return corridors_df


    def creation_tl_corridors_dataframe(self):

        tl_corridors = self.corridors_extractor.tl_corridors_extraction()
        print("CORRIDORS EXTRACTED")

        all_routes=[]
        all_tls=[]

        for tl,routes in tl_corridors.items():
            for route in routes.values():
                all_routes.append(route)
                all_tls.append(tl)

        corridors_dataframe=pd.DataFrame({"tl":all_tls,"route":all_routes})
        corridors_dataframe["route_last_edge"]=corridors_dataframe["route"].str[-1]

        corridors_dataframe["route_first_edge"]=corridors_dataframe["route"].str[0]


        corridors_dataframe["route_length_ff_tt"]=corridors_dataframe["route"].apply(self.get_route_length_ff_tt)

        corridors_dataframe["route_sum_length"]=corridors_dataframe["route_length_ff_tt"].str[0]

        corridors_dataframe["route_sum_ff_tt"]=corridors_dataframe["route_length_ff_tt"].str[1]

        corridors_dataframe["route_ff_speed"]=corridors_dataframe["route_sum_length"]/corridors_dataframe["route_sum_ff_tt"]

        corridors_dataframe["origin"]=corridors_dataframe["route"].apply(self.get_origin)

        corridors_dataframe["route_n_edges"]=corridors_dataframe["route"].apply(len)

        corridors_dataframe["route_str"]=corridors_dataframe["route"].astype(str)

        corridors_dataframe["route_str"]=corridors_dataframe["route"].astype(str)
        corridors_dataframe["route_str_str"]=corridors_dataframe["route"].str.join(" ")

        #corridors_dataframe=corridors_dataframe.sort_values(["tl","route_first_edge","route_ff_tt"],ascending=[True,True,True])
        #corridors_dataframe=corridors_dataframe.drop_duplicates(["tl","route_first_edge"],keep="first")
        #corridors_dataframe=corridors_dataframe.loc[corridors_dataframe["tl"]!=corridors_dataframe["origin"]]

        #corridors_dataframe["route_id"]=1
        #for tl in corridors_dataframe["tl"].unique():
            #corridors_dataframe.loc[corridors_dataframe["tl"]==tl,"route_id"]=[i for i in range(len(corridors_dataframe.loc[corridors_dataframe["tl"]==tl]))]

        return corridors_dataframe



    def get_origin(self,route):
        first_edge_id=route[0]
        first_edge=self.net.getEdge(first_edge_id)
        first_from_node=first_edge.getFromNode()
        first_from_node_type=first_from_node.getType()
        if first_from_node_type == "traffic_light":
            connections=first_from_node.getConnections()
            origin_tl_id=connections[0].getTLSID()
            return origin_tl_id
        else:
            return "dead_end"

    def get_route_length_ff_tt(self,route):
        route_lengths_array=np.array([self.net.getEdge(edge_id).getLength() for edge_id in route])
        route_ff_speeds_array=np.array([self.net.getEdge(edge_id).getSpeed() for edge_id in route])
        route_ff_tt_array=route_lengths_array/route_ff_speeds_array

        route_sum_length=np.sum(route_lengths_array)
        route_sum_ff_tt=np.sum(route_ff_tt_array)

        return route_sum_length,route_sum_ff_tt

    def tl_corridors_extraction(self):
        corridors={}
        starting_edges=self.creation_starting_edges()
        for starting_edge in starting_edges:
            route=[]
            visited=set()
            self.find_corridors(starting_edge, route=route , visited=visited , corridors=corridors)

        return corridors
        

    def creation_starting_edges(self):
        starting_edges=[]
        for edge in self.net.getEdges():
            if edge.getID().startswith(":"):
                continue
            from_node=edge.getFromNode()
            from_node_type=from_node.getType()
            if (from_node_type == "traffic_light") or (from_node_type == "dead_end"):
                starting_edges.append(edge)

        return starting_edges

    def find_corridors(self,starting_edge, route, visited, corridors):
        if starting_edge in visited:
            return

        visited.add(starting_edge)
        route = route + [starting_edge.getID()]
        to_node=starting_edge.getToNode()

        to_node_type=to_node.getType()

        if to_node_type=="traffic_light":
            connections=to_node.getConnections()
            tl=connections[0].getTLSID()
            if corridors.get(tl) is None:
                corridors[tl]={"route_0":route}
            else:
                corridors[tl][f"route_{len(corridors[tl])}"]=route

            return

        for next_edge in starting_edge.getOutgoing():
            self.find_corridors(
                next_edge,
                route.copy(),
                visited.copy(),
                corridors
            )