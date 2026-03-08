from topology_generator import generate_topology
import random
import heapq, json, csv
from message import Message
from node import Node
import os, statistics

class Simulation:
    def __init__(self,topology,mode):
        self.topology=topology
        self.mode=mode
        self.nodes={}
        self.events=[]
        self.seq=0

        for nid,data in topology["nodes"].items():
            self.nodes[nid]=Node(
                nid,
                data["cluster_id"],
                data["role"],
                data["neighbors"],
                data.get("cluster_head"),
                data.get("gateway_peers",[]),
                mode
            )

    def push(self,t,target,msg,sender):
        self.seq+=1
        heapq.heappush(self.events,(t,self.seq,target,msg,sender))

    def run(self,origin,payload):
        m=Message.new(origin,payload)
        self.push(0,origin,m,origin)

        while self.events:
            t,_,target,msg,sender=heapq.heappop(self.events)
            ev=self.nodes[target].receive(t,msg,sender,self.topology)
            for e in ev:
                self.push(*e)

        reached=sum(1 for n in self.nodes.values() if n.first_receive_time is not None)
        dup=sum(n.duplicate_count for n in self.nodes.values())
        time=max(n.first_receive_time or 0 for n in self.nodes.values())
        return reached,dup,time

# def load():
def load(num_nodes):
    # with open("topology.json") as f:
    #     return json.load(f)
    return generate_topology(num_nodes=num_nodes, clusters=2)

def main():

    NUM_NODES = 50   # change to 10, 20, 100 later
    RUNS = 10

    topo = load(NUM_NODES)

    modes = ["gossip", "cluster", "ahbn"]

    results = {m: {"dup": [], "time": []} for m in modes}

    for r in range(RUNS):

        origin = random.choice(list(topo["nodes"].keys()))

        print(f"\nRun {r+1} origin={origin}")

        for m in modes:

            sim = Simulation(topo, m)

            reached, dup, t = sim.run(origin, f"TX{r}")

            results[m]["dup"].append(dup)
            results[m]["time"].append(t)

            print(m, "duplicates:", dup, "time:", round(t,3))

    print("\nAVERAGE RESULTS")

    for m in modes:

        avg_dup = sum(results[m]["dup"]) / RUNS
        avg_time = sum(results[m]["time"]) / RUNS

        print(m, "avg duplicates:", round(avg_dup,2),
                 "avg time:", round(avg_time,2))

# def main():
#     topo=load()
#     modes=["gossip","cluster","ahbn"]
#     rows=[]

#     for m in modes:
#         sim=Simulation(topo,m)
#         r,d,t=sim.run("N2","TX001")
#         rows.append([m,r,d,t])

#     with open("results/summary.csv","w",newline="") as f:
#         w=csv.writer(f)
#         w.writerow(["mode","nodes_reached","duplicates","propagation_time"])
#         for r in rows:
#             w.writerow(r)

#     print("Results")
#     for r in rows:
#         print(r)

# def main():

#     topo = load()

#     modes = ["gossip", "cluster", "ahbn"]

#     RUNS = 10   # change this to 20 or 50 later

#     results = {mode: {"duplicates": [], "time": []} for mode in modes}

#     for i in range(RUNS):

#         print(f"\nRun {i+1}")

#         for mode in modes:

#             sim = Simulation(topo, mode)

#             r, d, t = sim.run("N2", f"TX{i}")

#             results[mode]["duplicates"].append(d)
#             results[mode]["time"].append(t)

#             print(mode, "duplicates:", d, "time:", t)

#     os.makedirs("results", exist_ok=True)

#     with open("results/summary.csv", "w", newline="") as f:

#         writer = csv.writer(f)

#         writer.writerow([
#             "mode",
#             "avg_duplicates",
#             "avg_propagation_time",
#             "runs"
#         ])

#         for mode in modes:

#             avg_dup = statistics.mean(results[mode]["duplicates"])
#             avg_time = statistics.mean(results[mode]["time"])

#             writer.writerow([
#                 mode,
#                 round(avg_dup, 3),
#                 round(avg_time, 3),
#                 RUNS
#             ])

#     print("\nAVERAGE RESULTS")

#     for mode in modes:

#         avg_dup = statistics.mean(results[mode]["duplicates"])
#         avg_time = statistics.mean(results[mode]["time"])

#         print(
#             mode,
#             "avg duplicates:", round(avg_dup, 3),
#             "avg time:", round(avg_time, 3)
#         )

if __name__=="__main__":
    main()
