from modules.sumo import TLIncomingCorridors,CompleteCorridorsConnexions
import torch

net_xml="./Ingolstadt Network/ingolstadt.net.xml"
route_files_list=["./Ingolstadt Network/InTAS_016.rou.xml","./Ingolstadt Network/InTAS_017.rou.xml","./Ingolstadt Network/InTAS_018.rou.xml"]

corridors_generator=TLIncomingCorridors(net_file=net_xml,route_files_list=route_files_list)
corridors_df=corridors_generator.get_corridors_df()
corridors_df=corridors_df.loc[(corridors_df["origin"]!="other_type") & (corridors_df["route_demand"]>0)]
torch.save(corridors_df,"./corridors_savings/corridors_df.pt")

corridors_df=torch.load("./corridors_savings/corridors_df.pt",weights_only=False)

complete_corridors_connexions_generator=CompleteCorridorsConnexions(route_files_list=route_files_list,net_file=net_xml,corridors_df=corridors_df)
complete_corridors_connexions_df=complete_corridors_connexions_generator.create_connexion_corridors_df()
complete_corridors_connexions_df=complete_corridors_connexions_df.loc[(complete_corridors_connexions_df["origin"]!="other_type") & (complete_corridors_connexions_df["connexion_demand"]>0)]
torch.save(complete_corridors_connexions_df,"./corridors_savings/complete_corridors_connexions_df.pt")
