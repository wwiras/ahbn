import random

def run_simulation(protocol,
                   node_count,
                   fanout=None,
                   ch_count=None,
                   ba_m=2,
                   scenario="normal",
                   seed=0):

    random.seed(seed)

    if protocol == "gossip":
        base_delay = 6.0 - 0.35 * (fanout or 1)
        propagation_time = max(2.0, base_delay + random.uniform(-0.3, 0.3))
        duplicate_messages = max(0, int((node_count * (fanout or 1) * 0.9) + random.uniform(-20, 20)))

    elif protocol == "cluster":
        ch = ch_count or 2
        optimal = max(2, node_count // 25)
        imbalance_penalty = abs(ch - optimal) * 0.45
        propagation_time = 4.0 + imbalance_penalty + random.uniform(-0.2, 0.2)
        duplicate_messages = max(0, int((ch * 8) + random.uniform(-5, 5)))

    elif protocol == "ahbn":
        f = fanout or 2
        propagation_time = max(2.2, 4.8 - 0.20 * f + random.uniform(-0.25, 0.25))
        duplicate_messages = max(0, int((node_count * f * 0.45) + random.uniform(-15, 15)))

    else:
        propagation_time = random.uniform(3.0, 6.0)
        duplicate_messages = int(random.uniform(50, 300))

    propagation_time = max(1.5, propagation_time - 0.08 * ba_m)
    duplicate_messages = int(duplicate_messages + ba_m * 10)

    if scenario == "overload":
        propagation_time += 1.2
    elif scenario == "failure":
        propagation_time += 2.0

    total_transmissions = duplicate_messages + node_count
    delivery_ratio = 0.85 if scenario == "failure" else 1.0

    return {
        "protocol": protocol,
        "node_count": node_count,
        "fanout": fanout,
        "ch_count": ch_count,
        "ba_m": ba_m,
        "scenario": scenario,
        "seed": seed,
        "propagation_time": round(propagation_time, 4),
        "duplicate_messages": duplicate_messages,
        "total_transmissions": total_transmissions,
        "delivery_ratio": delivery_ratio
    }