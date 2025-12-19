from flask import Flask
from blueprints.ui import ui_bp
from blueprints.projects import projects_bp
from blueprints.training import training_bp


app = Flask(__name__)
app.secret_key = "dev"  # replace later if needed

# Register blueprints
app.register_blueprint(ui_bp)
app.register_blueprint(projects_bp)
app.register_blueprint(training_bp)

if __name__ == "__main__":
    app.run(debug=False)