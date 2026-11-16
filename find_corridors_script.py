from modules.sumo import TLIncomingCorridors
import torch

if __name__=="__main__":

    net_xml="./Ingolstadt Network/ingolstadt.net.xml"
    route_files_list=["./Ingolstadt Network/InTAS_016.rou.xml","./Ingolstadt Network/InTAS_017.rou.xml","./Ingolstadt Network/InTAS_018.rou.xml"]
    corridors_generator=TLIncomingCorridors(net_file=net_xml,route_files_list=route_files_list)


    tl_corridors=corridors_generator.tl_corridors_extraction()
    torch.save(tl_corridors,"./corridors_savings/extracted_corrdiors.pt")

