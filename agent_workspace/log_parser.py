import re
logs = ['2026-10-15 14:30:00; 8.0.0.1; 192.168.1.1'
counts = {'8.0.0.1': 1, '192.168.1.1': 1}
for l in logs:
    m = re.match(r'^(\d{1,3}\.){3}{}$','{l}')
    if m: 
        ip = m.group()
        counts[ip] = counts.get(ip, 0) + 1
    else: 
        counts[ip] = 1
assert counts == {8.0.0.1: 1, 192.168.1.1: 1}
print(f'UNIQUE_IPS')