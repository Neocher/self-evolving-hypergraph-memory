import http.server, json, subprocess, re, signal, sys, os, threading

signal.signal(signal.SIGCHLD, signal.SIG_IGN)

GT_PROXY = os.environ.get('GT_PROXY', 'socks5://172.18.0.1:40001')
PID_FILE = '/tmp/gtproxy.pid'

URLS = {
    'bbc': 'https://www-bbc-com.translate.goog/news/?_x_tr_sl=en&_x_tr_tl=zh-CN',
    'aj': 'https://www-aljazeera-com.translate.goog/news/?_x_tr_sl=en&_x_tr_tl=zh-CN',
    'nhk': 'https://www3-nhk-or-jp.translate.goog/nhkworld/data/en/news/all.json?_x_tr_sl=en&_x_tr_tl=zh-CN',
}

def fetch(site):
    r = subprocess.run(['curl', '-sL', '--max-time', '20', '-x', GT_PROXY, URLS[site]],
        capture_output=True, text=True, timeout=25)
    articles, seen = [], set()
    if site == 'nhk':
        for m in re.finditer(r'"alt":\s*"([^"]+)"', r.stdout):
            t = m.group(1).replace('\\u0027', "'").replace('\\u0026', '&')
            t = re.sub(r'\s+', ' ', t).strip()
            if len(t) > 15 and t not in seen:
                seen.add(t)
                articles.append({'title': t})
    else:
        for url2, txt in re.findall(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r.stdout, re.DOTALL):
            t2 = re.sub(r'<[^>]+>', '', txt).strip()
            t2 = re.sub(r'\s+', ' ', t2)
            if 20 < len(t2) < 200 and t2 not in seen:
                seen.add(t2)
                articles.append({'title': t2})
                if len(articles) >= 20:
                    break
    return articles[:20]

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        site = self.path.strip('/').split('/')[0]
        if site == 'health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok', 'sites': list(URLS.keys())}).encode())
            return
        if site not in URLS:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(f'unknown: {site}'.encode())
            return
        try:
            articles = fetch(site)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'site': site,
                'count': len(articles),
                'articles': articles
            }, ensure_ascii=False).encode())
        except Exception as e:
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
    def log_message(self, *a):
        pass

if __name__ == '__main__':
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    server = http.server.HTTPServer(('0.0.0.0', 8084), Handler)
    print(f'GTProxy running on :8084, PID={os.getpid()}', flush=True)
    server.serve_forever()
