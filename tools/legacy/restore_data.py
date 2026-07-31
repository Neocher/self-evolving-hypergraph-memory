#!/usr/bin/env python3
"""Restore SHM data from JSON export to RyuGraph DB."""
import json, sys, os, time

cd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cd)
os.chdir(cd)

from graph.ryu_store import RyuStore

INPUT = os.path.join(cd, 'data', 'kuzu_migration_full.json')

def main():
    print(f'读取: {INPUT}')
    with open(INPUT) as f:
        data = json.load(f)

    store = RyuStore()
    store.connect()
    print('RyuStore OK - connected')

    nodes = data.get('nodes', {}).get('EpisodeNode', [])
    hyperedges = data.get('nodes', {}).get('HyperedgeNode', [])
    edges = data.get('edges', {})
    print(f'EpisodeNodes: {len(nodes)}, Hyperedges: {len(hyperedges)}')

    # Restore EpisodeNodes
    for i, raw in enumerate(nodes):
        n = raw if 'id' in raw else raw.get('e', raw)
        nid = n.get('id', '')
        content = n.get('content', '')
        if not content or not nid:
            continue
        try:
            store.create_episode({
                'id': nid,
                'content': content,
                'created_at': n.get('created_at', time.time()),
                'tau_initial': n.get('tau_value') or n.get('tau_initial', 0.5),
                'source': n.get('source', 'restore'),
            })
        except Exception as e:
            err = str(e)
            if 'already exists' in err:
                pass  # skip duplicates
            else:
                print(f'  node[{i}]: {err[:80]}')
        if (i+1) % 10 == 0:
            print(f'  EpisodeNode: {i+1}/{len(nodes)}')
    print(f'  EpisodeNodes done: {len(nodes)}')

    # Restore Hyperedges
    for i, raw in enumerate(hyperedges):
        h = raw if 'id' in raw else raw.get('h', raw)
        hid = h.get('id', '')
        if not hid:
            continue
        meta = h.get('metadata', '{}')
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except:
                meta = {'raw': meta}
        try:
            store.create_hyperedge_node({
                'id': hid,
                'type': h.get('type', 'episode'),
                'created_at': h.get('created_at', time.time()),
                'gate_value': h.get('gate_value', 1.0),
                'metadata': json.dumps(meta),
            })
        except Exception as e:
            err = str(e)
            if 'already exists' in err:
                pass
            else:
                print(f'  hyperedge[{i}]: {err[:80]}')
        if (i+1) % 50 == 0:
            print(f'  Hyperedge: {i+1}/{len(hyperedges)}')
    print(f'  Hyperedges done: {len(hyperedges)}')

    store.close()
    print('\n恢复完成！')

if __name__ == '__main__':
    main()
