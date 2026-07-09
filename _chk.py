import os
BASE = os.path.dirname(os.path.abspath(__file__))
labels_dir = os.path.join(BASE, 'datasets', 'bvn', 'labels', 'train')
counts = {}
total = 0
names = {0: 'hand_up', 1: 'abnormal', 2: 'sleep', 3: 'phone'}

for fn in os.listdir(labels_dir):
    if fn.endswith('.txt'):
        with open(os.path.join(labels_dir, fn)) as f:
            for line in f:
                cid = line.strip().split()[0]
                counts[cid] = counts.get(cid, 0) + 1
                total += 1

print(f"Total files: {len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])}")
print(f"Total labels: {total}")
for cid in sorted(counts.keys(), key=lambda x: int(x)):
    name = names.get(int(cid), cid)
    print(f"  {name}: {counts[cid]} instances ({counts[cid]/total*100:.1f}%)")
