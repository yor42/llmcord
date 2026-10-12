import json, os, subprocess, sys, time, urllib.request, signal
sys.path.insert(0, os.path.dirname(__file__))
from cases import C
RT = os.path.expanduser('~/feat39-bench-data/rt/llama-b11559')
MD = os.path.expanduser('~/feat39-bench-data/models')
OUT = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT, exist_ok=True)
DIRECTOR = 'You direct a casual group skit. Pick speaker IDs only from the supplied cast. Choose one speaker normally and up to three when a short exchange improves the joke. For ambient chat, choose no speaker unless the group clearly invites a character. Do not invent IDs or choose a non-cast character.'
SCHEMA = {"type": "object", "properties": {"speakers": {"type": "array", "items": {"type": "integer"}}}, "required": ["speakers"], "additionalProperties": False}
SYSTEM = DIRECTOR + '\n\nReturn only the required structured result. Output schema: ' + json.dumps(SCHEMA)
PORT = 8099

def memavail():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'): return int(l.split()[1]) // 1024

def payload(c):
    recent = [{'message_id': 1000 + i, 'author_id': 500 + (hash(a) % 90), 'author_label': a, 'text': t, 'mentions': []} for i, (a, t) in enumerate(c['recent'])][-6:]
    return json.dumps({'cast': [{'id': i, 'name': n} for i, n in c['cast']], 'recent': recent, 'latest': c['latest'],
        'latest_author': {'author_id': 777, 'author_label': c['author']}, 'ambient': c['ambient'], 'forced': None}, ensure_ascii=False)

def post(body):
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t = time.time()
    with urllib.request.urlopen(req, timeout=300) as r: d = json.load(r)
    return d, time.time() - t

def hwm(pid):
    for l in open(f'/proc/{pid}/status'):
        if l.startswith('VmHWM'): return int(l.split()[1]) // 1024

def run(model):
    path = f'{MD}/{model}.gguf'
    env = dict(os.environ, LD_LIBRARY_PATH=RT)
    t0 = time.time()
    p = subprocess.Popen([f'{RT}/llama-server', '-m', path, '-t', '3', '-c', '2048', '-np', '1', '--port', str(PORT), '--host', '127.0.0.1', '--no-webui', '--no-warmup'],
                         env=env, stdout=open(f'{OUT}/{model}.server.log', 'w'), stderr=subprocess.STDOUT)
    try:
        while True:
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2); break
            except Exception: time.sleep(0.5)
            if p.poll() is not None: raise SystemExit('server died')
        load = time.time() - t0
        rows = []
        def call(c, constrained):
            body = {'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': payload(c)}], 'temperature': 0, 'max_tokens': 64, 'cache_prompt': False}
            if constrained: body['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'choose_speakers', 'schema': SCHEMA, 'strict': True}}
            d, wall = post(body)
            tm = d['timings']
            return {'text': d['choices'][0]['message']['content'], 'wall': wall, 'prompt_n': tm['prompt_n'], 'prefill_ms': tm['prompt_ms'], 'decode_n': tm['predicted_n'], 'decode_ms': tm['predicted_ms']}
        call(C[0], True)  # warm-up
        for mode in (True, False):
            for i, c in enumerate(C):
                if memavail() < 600: raise SystemExit('MemAvailable < 600 MB, stopping')
                r = call(c, mode); r.update(case=i, constrained=mode); rows.append(r)
        res = {'model': model, 'load_s': load, 'peak_rss_mb': hwm(p.pid), 'rows': rows, 'temp': subprocess.run(['vcgencmd', 'measure_temp'], capture_output=True, text=True).stdout.strip(),
               'throttled': subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True, text=True).stdout.strip(), 'min_memavail_mb_end': memavail()}
        json.dump(res, open(f'{OUT}/{model}.json', 'w'), ensure_ascii=False, indent=1)
        print(model, 'load', round(load, 1), 'rss', res['peak_rss_mb'], res['temp'], res['throttled'])
    finally:
        p.send_signal(signal.SIGTERM); p.wait(30)

if __name__ == '__main__':
    for m in sys.argv[1:]: run(m)
