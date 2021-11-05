import json
import subprocess

def daphne(args, cwd='C:/Users/jlovr/CS532-HW4/BBVI/daphne'):
    proc = subprocess.run(['lein','run','-f','json'] + args,
                          capture_output=True, cwd=cwd, shell=True)
    if(proc.returncode != 0):
        raise Exception(proc.stdout.decode() + proc.stderr.decode())
    return json.loads(proc.stdout)
