
AHBN Prototype (Milestone Implementation)

Folders
-------
sim/            deterministic AHBN simulator
async_demo/     placeholder for async node version

Concept
-------
Cluster heads include additional processing delay to emulate coordination overhead.
This exposes the latency vs duplication tradeoff even in a small network.

Run
---
cd sim
python simulate.py

Output
------
results/summary.csv

Expected Pattern
----------------
gossip  : fastest but many duplicates
cluster : clean but slower
AHBN    : balanced
