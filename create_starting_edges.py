from modules.sumo import TLIncomingCorridors
import torch
import sumolib

if __name__=="__main__":

    net_xml="./Ingolstadt Network/ingolstadt.net.xml"
    route_files_list=["./Ingolstadt Network/InTAS_016.rou.xml","./Ingolstadt Network/InTAS_017.rou.xml","./Ingolstadt Network/InTAS_018.rou.xml"]
    corridors_generator=TLIncomingCorridors(net_file=net_xml,route_files_list=route_files_list)


    starting_edges=corridors_generator.creation_starting_edges()
    starting_edges_ids=[edge.getID() for edge in starting_edges]
    torch.save(starting_edges_ids,"./corridors_savings/starting_edges_ids.pt")

