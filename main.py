import subprocess

process = subprocess.Popen(['python', 'src/main.py', 'process', '--repo-url', 'https://github.com/RAravindDS/mastering-ci-cd'])
process.wait()