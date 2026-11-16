from collections import deque

import sumolib
import torch

class CorridorExtractor:
    """
    Extraction des corridors entre :
      - un traffic light et un autre traffic light
      - un dead-end et un traffic light

    Méthode : un BFS par starting edge, qui s'arrête dès qu'une edge
    aboutit sur un traffic light. Chaque edge n'est visitée qu'une seule fois
    par BFS -> O(V + E) par départ, au lieu d'une énumération exponentielle
    de tous les chemins.

    Format de sortie (identique à ton code actuel) :
        corridors = {
            "tls_id_1": {"route_0": [edge_id, ...], "route_1": [...], ...},
            "tls_id_2": {...},
        }
    Chaque route est le PLUS COURT chemin (en nombre d'edges) depuis une
    starting edge jusqu'à une edge qui arrive sur ce traffic light.
    """

    # Types de noeuds SUMO considérés comme des feux
    TL_TYPES = {"traffic_light"}
    START_TYPES = TL_TYPES | {"dead_end"}

    def __init__(self, net):
        self.net = net
        self._build_cache()

    # ------------------------------------------------------------------
    # Pré-calcul (une seule fois) : évite tous les appels sumolib répétés
    # ------------------------------------------------------------------
    def _build_cache(self):
        self._out = {}          # edge_id -> [edge_id sortantes]
        self._end_tl = {}       # edge_id -> tls_id (seulement si l'edge arrive sur un feu)
        self._starting_edges = []  # [edge_id]

        node_tl_cache = {}      # node_id -> tls_id ou None

        for edge in self.net.getEdges():
            eid = edge.getID()
            if eid.startswith(":"):
                continue

            # Adjacence (on ignore les edges internes)
            self._out[eid] = [
                o.getID() for o in edge.getOutgoing()
                if not o.getID().startswith(":")
            ]

            # Edge de départ ?
            from_type = edge.getFromNode().getType()
            if from_type in self.START_TYPES:
                self._starting_edges.append(eid)

            # Edge qui termine sur un feu ?
            to_node = edge.getToNode()
            if to_node.getType() in self.TL_TYPES:
                nid = to_node.getID()
                if nid not in node_tl_cache:
                    node_tl_cache[nid] = self._node_tls_id(to_node)
                self._end_tl[eid] = node_tl_cache[nid]

    @staticmethod
    def _node_tls_id(node):
        """Retourne l'id du TLS associé à un noeud (fallback : id du noeud)."""
        for conn in node.getConnections():
            tls = conn.getTLSID()
            if tls:
                return tls
        return node.getID()

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def tl_corridors_extraction(self):
        corridors = {}
        for start_id in self._starting_edges:
            self._bfs_corridors(start_id, corridors)
        return corridors

    # ------------------------------------------------------------------
    # BFS depuis une starting edge
    # ------------------------------------------------------------------
    def _bfs_corridors(self, start_id, corridors):
        parent = {start_id: None}   # sert aussi de "visited"
        queue = deque([start_id])

        while queue:
            eid = queue.popleft()

            tl = self._end_tl.get(eid)
            if tl is not None:
                # On a atteint un feu : reconstruire la route et ne pas traverser
                route = []
                cur = eid
                while cur is not None:
                    route.append(cur)
                    cur = parent[cur]
                route.reverse()

                tl_routes = corridors.setdefault(tl, {})
                tl_routes[f"route_{len(tl_routes)}"] = route
                continue

            for nxt in self._out[eid]:
                if nxt not in parent:
                    parent[nxt] = eid
                    queue.append(nxt)


if __name__ == "__main__":
    net = sumolib.net.readNet("./Ingolstadt Network/ingolstadt.net.xml", withInternal=False)
    extractor = CorridorExtractor(net)
    tl_corridors = extractor.tl_corridors_extraction()

    print(f"{len(tl_corridors)} traffic lights atteints")
    print(f"{sum(len(r) for r in tl_corridors.values())} corridors au total")

    torch.save(tl_corridors,"./corridors_savings/tl_corridors.pt")