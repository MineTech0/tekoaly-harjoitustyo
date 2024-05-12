import invoke

@invoke.task
def start(ctx):
    ctx.run("flask --app web-app/app.py run")

@invoke.task
def test(ctx):
    ctx.run("python -m unittest discover -s own_model/tests -v")
    
@invoke.task
def coverage(ctx):
    ctx.run("coverage run -m unittest discover -s own_model/tests")
    ctx.run("coverage report")
    
@invoke.task
def train(ctx):
    ctx.run("python own_model/training.py", pty=True)
    
@invoke.task
def preprocess(ctx):
    ctx.run("python preprocess_data.py",pty=True)