def mpl_apply():
    """
    Aplica un estilo personalizado a las gráficas de Matplotlib y Seaborn.
    from style import apply
    mpl_apply()
    """
    import seaborn as sns

    sns.set(
        style="whitegrid",
        palette="muted",
        font="EB Garamond",
        font_scale=1.5,
        rc={
            "grid.linestyle": "--",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "grid.color": "lightgray",
            "figure.figsize": (16/2, 9/2),
            "axes.titlesize": 16,
            "axes.labelsize": 10,

            # Tick labels: tamaño + color
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.labelcolor": "black",

            # Ticks: asegúrate de que se dibujan
            "xtick.bottom": True,        # mostrar ticks abajo
            "xtick.top": False,
            "ytick.left": True,
            "ytick.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,       # longitud de la marca
            "ytick.major.size": 4,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,

            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "axes.titlepad": 15,

            # Legend: fondo sí, sin borde visible, esquinas cuadradas
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.facecolor": "white",
            "legend.edgecolor": "none",
            "legend.framealpha": 0.8,

            # Spines: solo abajo y a la izquierda, en negro
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,

            # Límites y márgenes
            "axes.autolimit_mode": "round_numbers",
            "axes.xmargin": 0.0,    # sin margen extra
            "axes.ymargin": 0.0,
        },
    )

def set_style():
    mpl_apply()




def plotly_apply(palette=["#ffa600", "#ffd380"], fontsize=18, fontstack="EB Garamond, Garamond, Georgia, 'Times New Roman', serif"):
    """
    Aplica un estilo personalizado a las gráficas de Plotly, poner:
    from style import plotly_apply
    plotly_apply()
    """
    import pandas as pd
    pd.options.plotting.backend = "plotly"
    import plotly.io as pio
    pio.templates.default = "gridon"
    import plotly.graph_objects as go
    import plotly.express as px

    # Parte de 'gridon' y personalizaa
    base = pio.templates["gridon"]
    custom = go.layout.Template(base)

    font_stack = fontstack
    font_size = fontsize

    custom.layout.update(
    colorway=palette,                # Colores discretos por defecto
    font=dict(family=font_stack, size=font_size, color="#2b2b2b"),
    paper_bgcolor="#181818",
    plot_bgcolor="#181818",

    coloraxis=dict(colorscale="Blues"),  # Escala continua por defecto

    title=dict(font=dict(family=font_stack, size=font_size * 1.3, color="white")),
    xaxis=dict(title_font=dict(family=font_stack, size=font_size),
            tickfont=dict(family=font_stack, size=font_size * 0.857), 
            gridcolor="#e5e5e5", zerolinecolor="#cccccc", color="white"),
    yaxis=dict(title_font=dict(family=font_stack, size=font_size),
            tickfont=dict(family=font_stack, size=font_size * 0.857), 
            gridcolor="#e5e5e5", zerolinecolor="#cccccc", color="white"),
    legend=dict(font=dict(family=font_stack, size=font_size * 0.857, color="white"))
    )

    # Registra y usa como template global
    pio.templates["mi_tema"] = custom
    pio.templates.default = "mi_tema"

    # (Opcional) Defaults de Plotly Express
    px.defaults.template = "mi_tema"
    px.defaults.color_discrete_sequence = palette
    px.defaults.color_continuous_scale = "Blues"