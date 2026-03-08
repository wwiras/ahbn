import networkx as nx
import random


def generate_topology(num_nodes=20, clusters=2):
    nodes = [f"N{i}" for i in range(1, num_nodes + 1)]

    # m must be < num_nodes
    m = min(3, max(1, num_nodes - 1))
    G = nx.barabasi_albert_graph(num_nodes, m)

    topology = {
        "clusters": {},
        "nodes": {},
        "link_delays": {},
        "processing_delay": {}
    }

    # --- assign nodes to clusters ---
    cluster_map = {}
    cluster_heads = []

    base_size = num_nodes // clusters
    remainder = num_nodes % clusters

    start = 0
    for c in range(clusters):
        extra = 1 if c < remainder else 0
        end = start + base_size + extra

        cluster_nodes = nodes[start:end]
        cluster_id = f"C{c+1}"
        topology["clusters"][cluster_id] = cluster_nodes

        ch = cluster_nodes[0]
        cluster_heads.append(ch)

        for node in cluster_nodes:
            cluster_map[node] = {
                "cluster_id": cluster_id,
                "cluster_head": ch
            }

        start = end

    # --- initialize node records using graph neighbors ---
    for i, node in enumerate(nodes):
        graph_neighbors = [nodes[n] for n in G.neighbors(i)]

        topology["nodes"][node] = {
            "cluster_id": cluster_map[node]["cluster_id"],
            "role": "CH" if node == cluster_map[node]["cluster_head"] else "member",
            "neighbors": list(graph_neighbors),
            "cluster_head": None if node == cluster_map[node]["cluster_head"] else cluster_map[node]["cluster_head"],
            "gateway_peers": []
        }

    # --- add gateway links between CHs ---
    for i in range(len(cluster_heads) - 1):
        ch1 = cluster_heads[i]
        ch2 = cluster_heads[i + 1]

        topology["nodes"][ch1]["gateway_peers"].append(ch2)
        topology["nodes"][ch2]["gateway_peers"].append(ch1)

        if ch2 not in topology["nodes"][ch1]["neighbors"]:
            topology["nodes"][ch1]["neighbors"].append(ch2)
        if ch1 not in topology["nodes"][ch2]["neighbors"]:
            topology["nodes"][ch2]["neighbors"].append(ch1)

    # --- ensure CH has links to all members in its cluster ---
    for cluster_id, members in topology["clusters"].items():
        ch = members[0]
        for member in members[1:]:
            if member not in topology["nodes"][ch]["neighbors"]:
                topology["nodes"][ch]["neighbors"].append(member)
            if ch not in topology["nodes"][member]["neighbors"]:
                topology["nodes"][member]["neighbors"].append(ch)

    # --- create link delays for ALL neighbor edges ---
    for node in nodes:
        topology["link_delays"][node] = {}
        for neigh in topology["nodes"][node]["neighbors"]:
            topology["link_delays"][node][neigh] = round(random.uniform(0.8, 1.5), 3)

    # --- processing delay ---
    for node in nodes:
        if topology["nodes"][node]["role"] == "CH":
            topology["processing_delay"][node] = 0.6
        else:
            topology["processing_delay"][node] = 0.05

    return topology