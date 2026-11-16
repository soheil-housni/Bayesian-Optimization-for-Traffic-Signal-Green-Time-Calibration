from collections import deque


class CorridorExtractor:

    TL_TYPES = {"traffic_light"}
    START_TYPES = TL_TYPES | {"dead_end"}

    def __init__(self, net):
        self.net = net
        self._build_cache()

    def _build_cache(self):
        self._out = {}          
        self._end_tl = {}       
        self._starting_edges = []  

        node_tl_cache = {} 

        for edge in self.net.getEdges():
            eid = edge.getID()
            if eid.startswith(":"):
                continue

            self._out[eid] = [
                o.getID() for o in edge.getOutgoing()
                if not o.getID().startswith(":")
            ]

            from_type = edge.getFromNode().getType()
            if from_type in self.START_TYPES:
                self._starting_edges.append(eid)

            to_node = edge.getToNode()
            if to_node.getType() in self.TL_TYPES:
                nid = to_node.getID()
                if nid not in node_tl_cache:
                    node_tl_cache[nid] = self._node_tls_id(to_node)
                self._end_tl[eid] = node_tl_cache[nid]

    @staticmethod
    def _node_tls_id(node):
        for conn in node.getConnections():
            tls = conn.getTLSID()
            if tls:
                return tls
        return None

    def tl_corridors_extraction(self):
        corridors = {}
        for start_id in self._starting_edges:
            self._bfs_corridors(start_id, corridors)
        return corridors


    def _bfs_corridors(self, start_id, corridors):
        parent = {start_id: None}   
        queue = deque([start_id])

        while queue:
            eid = queue.popleft()

            tl = self._end_tl.get(eid)
            if tl is not None:
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