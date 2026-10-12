"""Tiny fake Ollama for dry-running the harness offline: python stub_ollama.py [port]. Answers are canned and meaningless."""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

TAG = 'qwen3.5:4b'


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self.send({'models': [{'name': TAG}]} if self.path == '/api/tags' else {'version': 'stub'} if self.path == '/api/version' else {'models': []})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        fmt = body.get('format')
        props = (fmt or {}).get('properties', {})
        if 'speakers' in props:
            out = {'speakers': [3]}
        elif 'answer' in props:
            out = {'answer': props['answer']['enum'][0]}
        elif props:
            out = {'shared_facts': ['stub fact'], 'personal_facts': [], 'encounter_facts': []}
        else:
            out = 'Stub summary of the skit.'
        self.send({'message': {'content': out if isinstance(out, str) else json.dumps(out)}, 'prompt_eval_count': 100, 'prompt_eval_duration': 1000000,
                   'eval_count': 5, 'eval_duration': 500000, 'load_duration': 0, 'total_duration': 2000000})


if __name__ == '__main__':
    HTTPServer(('127.0.0.1', int(sys.argv[1]) if len(sys.argv) > 1 else 11999), H).serve_forever()
