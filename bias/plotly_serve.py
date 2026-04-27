import dash
from dash import Dash, html, dcc

from surface_explore_interactive import app as surface_explore_app

# Initialize the app.
# We set pages_folder="" so Dash doesn't go looking for a 'pages/' directory.
app = Dash(__name__, use_pages=True, pages_folder="")

# MANUALLY REGISTER THE PAGES
# The first argument is a unique module name (can be anything unique).
dash.register_page("surface", layout=surface_explore_app.layout, path="/surface", name="Surface Explore Interactive")
# dash.register_page("bar_module", layout=bar.layout, path="/bar", name="Bar App")

app.layout = html.Div([
    html.H1('Main Multi-App Container'),

    # Navigation generated from the registry
    html.Div([
        html.Div(
            dcc.Link(f"Go to {page['name']}", href=page["relative_path"])
        ) for page in dash.page_registry.values()
    ]),

    # This is where foo.layout or bar.layout will be injected
    dash.page_container
])

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8889)