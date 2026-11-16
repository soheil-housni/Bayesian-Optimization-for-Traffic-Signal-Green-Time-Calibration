from modules.sumo import SumoSimulation
import sumolib

if __name__=="__main__":
    net=sumolib.net.readNet("Ingolstadt Network/ingolstadt.net.xml")
    tl_ids=[tl.getID() for tl in net.getTrafficLights()]
    sumo_simulator=SumoSimulation(
        cfg_file="./Ingolstadt Network/InTAS_gt.sumocfg",
        output_dir="gt_data/",
        tl_ids=tl_ids
        
    )

    sumo_simulator.simulation(epoch_index="test",rep=-1)