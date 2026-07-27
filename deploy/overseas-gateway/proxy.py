#!/usr/bin/env python3
"""简易 HTTP CONNECT 代理 - 帮 Docker 过防火墙"""
import socket, threading, sys, os

def handle(client, addr):
    req = client.recv(4096)
    if not req:
        client.close()
        return
    lines = req.split(b'\r\n')
    first = lines[0].decode()
    parts = first.split()
    if len(parts) < 3:
        client.close()
        return
    method, target = parts[0], parts[1]
    
    if method == 'CONNECT':
        host, _, port_s = target.partition(':')
        port = int(port_s) if port_s else 443
        try:
            remote = socket.create_connection((host, port), timeout=30)
            client.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n')
            # 双向转发
            def forward(src, dst):
                try:
                    while True:
                        data = src.recv(65536)
                        if not data: break
                        dst.sendall(data)
                except: pass
            threading.Thread(target=forward, args=(client, remote), daemon=True).start()
            threading.Thread(target=forward, args=(remote, client), daemon=True).start()
        except Exception as e:
            client.sendall(f'HTTP/1.1 502 Bad Gateway\r\n\r\n{e}'.encode())
            client.close()
    else:
        # HTTP 直连转发
        import urllib.request
        try:
            r = urllib.request.urlopen(urllib.request.Request(target, headers={
                k.decode(): v.decode() for k, v in [l.split(b':', 1) for l in lines[1:] if b':' in l]
            }), timeout=30)
            resp = f'HTTP/1.1 {r.status} {r.reason}\r\n'
            for k, v in r.headers.items():
                resp += f'{k}: {v}\r\n'
            client.sendall(resp.encode() + b'\r\n' + r.read())
        except Exception as e:
            client.sendall(f'HTTP/1.1 502 Bad Gateway\r\n\r\n'.encode())
        client.close()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8083
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', port))
    s.listen(100)
    print(f'Proxy on :{port}', flush=True)
    while True:
        client, addr = s.accept()
        threading.Thread(target=handle, args=(client, addr), daemon=True).start()
