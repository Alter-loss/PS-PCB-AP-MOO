import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ast
import math
import base64
from io import BytesIO
from deap import base, creator, tools, algorithms
import os
import random


# Safe evaluation environment
safe_env = {
    '__builtins__': None,
    'math': math,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'exp': math.exp,
    'log': math.log,
    'log10': math.log10,
    'sqrt': math.sqrt,
    'pi': math.pi,
    'e': math.e,
    'abs': abs,
    'min': min,
    'max': max,
    'sum': sum,
    'avg': lambda x: sum(x)/len(x),
    'np': np
}

def safe_eval(expr, context):
    """Safely evaluate an expression"""
    try:
        local_env = safe_env.copy()
        local_env.update(context)
        node = ast.parse(expr, mode='eval')
        code = compile(node, filename='<string>', mode='eval')
        return eval(code, {"__builtins__": None}, local_env)
    except Exception as e:
        st.error(f"Calculation error: {e}\nExpression: {expr}")
        return None

def calculate_all(variable_values, intermediates_def, objectives_def):
    """
    Unified calculation for all intermediates and objectives.
    Args:
        variable_values: dict, {var_name: value}
        intermediates_def: dict, {name: {expression: ...}}
        objectives_def: dict, {name: {expression: ...}}
    Returns:
        (dict_of_intermediates, dict_of_objectives)
    """
    context = variable_values.copy()
    intermediates_results = {}

    # Calculate intermediates (with dependency resolution)
    changed = True
    while changed:
        changed = False
        for name, inter in intermediates_def.items():
            if name in intermediates_results:
                continue
            expr = inter['expression']
            try:
                result = safe_eval(expr, context)
                intermediates_results[name] = result
                context[name] = result
                changed = True
            except NameError:
                pass
            except Exception as e:
                intermediates_results[name] = None

    for name, inter in intermediates_def.items():
        if name not in intermediates_results:
            try:
                expr = inter['expression']
                result = safe_eval(expr, context)
                intermediates_results[name] = result
                context[name] = result
            except Exception as e:
                intermediates_results[name] = None

    # Calculate objectives
    objectives_results = {}
    for name, obj in objectives_def.items():
        expr = obj['expression']
        try:
            result = safe_eval(expr, context)
            objectives_results[name] = result
        except Exception as e:
            objectives_results[name] = None

    return intermediates_results, objectives_results

# Page settings
st.set_page_config(
    page_title="Multi-Objective Optimization Configuration Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = None
if 'variables' not in st.session_state:
    st.session_state.variables = {}
if 'intermediates' not in st.session_state:
    st.session_state.intermediates = {}
if 'objectives' not in st.session_state:
    st.session_state.objectives = {}
if 'results' not in st.session_state:
    st.session_state.results = None

# Title and introduction
st.title("📊 Multi-Objective Optimization Tool")
st.markdown("""
Use the NSGA-II algorithm of the DEAP library for multi-objective optimization. Upload a configuration file to define variables, intermediate variables, and objective functions,
set initial values, and run the optimization.
""")

# Configuration file upload
with st.expander("📤 Upload Configuration File", expanded=True):
    config_file = st.file_uploader("Upload a JSON configuration file", type=["json"])
    if config_file:
        try:
            config = json.load(config_file)
            st.session_state.config = config
            st.success("Configuration file parsed successfully!")
            st.session_state.variables = {v['name']: v for v in config.get('variables', [])}
            st.session_state.intermediates = {i['name']: i for i in config.get('intermediates', [])}
            st.session_state.objectives = {o['name']: o for o in config.get('objectives', [])}
            # Reset all memory/results when loading a new config!
            st.session_state.results = None
            # Optionally also reset variable inputs if you want to clear all user input:
            for key in list(st.session_state.keys()):
                if key.startswith("input_") or key.startswith("weight_"):
                    del st.session_state[key]
        except Exception as e:
            st.error(f"Configuration file parsing error: {e}")

    if st.button("📥 Load Config", use_container_width=True):
        if os.path.exists('optimization_config.json'):
            with open('optimization_config.json', 'r') as f:
                config = json.load(f)
                st.session_state.config = config
                st.success("Configuration file parsed successfully!")
                st.session_state.variables = {v['name']: v for v in config.get('variables', [])}
                st.session_state.intermediates = {i['name']: i for i in config.get('intermediates', [])}
                st.session_state.objectives = {o['name']: o for o in config.get('objectives', [])}
                # Reset all memory/results when loading a new config!
                st.session_state.results = None
                # Optionally also reset variable inputs if you want to clear all user input:
                for key in list(st.session_state.keys()):
                    if key.startswith("input_") or key.startswith("weight_"):
                        del st.session_state[key]
        else:
            st.error(f"No config file found, please use Browse file to upload")

# Display configuration informations
if st.session_state.config:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Variable Configuration")
        variables_df = pd.DataFrame(st.session_state.variables.values())
        columns_to_show = ['name', 'description', 'type', 'lb', 'ub']
        if 'categories' in variables_df.columns:
            columns_to_show.append('categories')
        st.dataframe(variables_df[columns_to_show], height=300)
    with col2:
        st.subheader("Objective Functions")
        objectives_df = pd.DataFrame(st.session_state.objectives.values())
        st.dataframe(objectives_df[['name','description','target', 'expression']], height=300)
if st.session_state.intermediates:  
    st.subheader("Intermediate Variables")
    intermediates_df = pd.DataFrame(st.session_state.intermediates.values())
    if not intermediates_df.empty:
        columns_to_show = [col for col in ['name', 'description', 'expression'] if col in intermediates_df.columns]
        if columns_to_show:
            st.dataframe(intermediates_df[columns_to_show], height=100)
        else:
            st.warning("No columns to display in intermediate variables")
    else:
        st.warning("No intermediate variables defined")
else:
    st.warning("No intermediate variables defined")

# Initial design input
if st.session_state.variables:
    st.header("📝 Initial Design Input")
    st.markdown("Set initial values for each variable, and the system will automatically calculate the values of intermediate variables and objective functions.")
    # Create variable input controls
    inputs = {}
    cols = st.columns(3)
    col_idx = 0
    for i, (name, var) in enumerate(st.session_state.variables.items()):
        with cols[col_idx]:
            range_hint = ""
            if var['type'] in ['float', 'int']:
                range_hint = f" (Range: {var['lb']} - {var['ub']})"
            label = f"{name}{range_hint}"
            description = var.get('description', '')
            if description:
                label = f"{label}\n_{description}_"
            if var['type'] == 'float':
                default_val = (float(var['lb']) + float(var['ub'])) / 2
                default_val = max(float(var['lb']), min(float(var['ub']), default_val))
                inputs[name] = st.number_input(
                    label,
                    min_value=float(var['lb']),
                    max_value=float(var['ub']),
                    value=default_val,
                    step=0.01,
                    key=f"input_{name}"
                )
            elif var['type'] == 'int':
                default_val = (int(var['lb']) + int(var['ub'])) // 2
                default_val = max(int(var['lb']), min(int(var['ub']), default_val))
                inputs[name] = st.number_input(
                    label,
                    min_value=int(var['lb']),
                    max_value=int(var['ub']),
                    value=default_val,
                    step=1,
                    key=f"input_{name}"
                )
            elif var['type'] == 'categories':
                options = var['categories']
                inputs[name] = st.selectbox(
                    label,
                    options=options,
                    index=0,
                    key=f"input_{name}"
                )
        col_idx = (col_idx + 1) % 3

    if inputs:
        # --- UNIFIED CALCULATION ---
        intermediates_results, objectives_results = calculate_all(
            inputs,
            st.session_state.intermediates,
            st.session_state.objectives
        )

        # Display results
        st.subheader("📌 Calculation Results")
        st.markdown("#### Objective Function Values")
        if objectives_results:
            obj_cols = st.columns(len(objectives_results))
            for i, (name, value) in enumerate(objectives_results.items()):
                with obj_cols[i]:
                    st.markdown(f"**{name}**")
                    if isinstance(value, (int, float)):
                        st.markdown(f"<div style='background-color:#e6f7ff; padding:15px; border-radius:10px;'>{value:.2f}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='background-color:#e6f7ff; padding:15px; border-radius:10px;'>{value}</div>", unsafe_allow_html=True)
        else:
            st.warning("No objective function values calculated")
        st.write("")
        with st.expander("#### Intermediate Variable Values", expanded=False):
            if intermediates_results:
                var_groups = [list(intermediates_results.items())[i:i+6] 
                            for i in range(0, len(intermediates_results), 6)]
                for group in var_groups:
                    cols = st.columns(6)
                    for i, (name, value) in enumerate(group):
                        with cols[i]:
                            st.markdown(f"**{name}**")
                            if isinstance(value, (int, float)):
                                st.markdown(f"<div style='background-color:#f0f2f6; padding:15px; border-radius:10px;'>{value:.2f}</div>", 
                                            unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div style='background-color:#f0f2f6; padding:15px; border-radius:10px;'>{value}</div>", 
                                            unsafe_allow_html=True)
                st.markdown("<br><br>", unsafe_allow_html=True)
            else:
                st.warning("No intermediate variable values calculated")

# Application instructions
st.sidebar.markdown("""
**Instructions:**
1. Upload a JSON configuration file.
2. Set initial values in the variable area.
3. View the initial design results.
4. Set optimization parameters in the sidebar (optional).
5. Click "Start Optimization" to run the NSGA-II algorithm.
6. View and download the optimization results.
""")
st.sidebar.markdown("---")
# Optimization parameter settings
with st.sidebar.expander("⚙️ Optimization Parameter Settings (optional)", expanded=False):
    pop_size = st.number_input("Population Size", min_value=1, max_value=50000, value=50, step=1)
    num_gen = st.number_input("Number of Generations", min_value=1, max_value=1000, value=50, step=1)
    cx_prob = st.number_input("Crossover Probability", min_value=0.1, max_value=1.0, value=0.7, step=0.01, format="%.2f")
    mut_prob = st.number_input("Mutation Probability", min_value=0.01, max_value=1.0, value=0.2, step=0.01, format="%.2f")

# --- OPTIMIZATION LOGIC ---
if st.sidebar.button("🚀 Start Optimization", use_container_width=True):
    if not st.session_state.config:
        st.sidebar.error("Please upload a configuration file first!")
    else:
        with st.spinner("Optimization in progress, please wait..."):
            # Get the objectives and their targets
            objectives = list(st.session_state.objectives.values())
            # Build weights: -1.0 for "minimize", +1.0 for "maximize"
            weights = tuple(-1.0 if obj.get('target', 'minimize').lower() == 'minimize' else 1.0 for obj in objectives)

            # Safely re-create only if not already created with correct weights (avoid DEAP errors)
            if hasattr(creator, 'FitnessMin'):
                del creator.FitnessMin
            if hasattr(creator, 'Individual'):
                del creator.Individual
            creator.create("FitnessMin", base.Fitness, weights=weights)
            creator.create("Individual", list, fitness=creator.FitnessMin)
            toolbox = base.Toolbox()
            float_vars, int_vars, cat_vars = [], [], []
            variables = list(st.session_state.variables.values())
            for var in variables:
                if var['type'] == 'float':
                    float_vars.append(var)
                elif var['type'] == 'int':
                    int_vars.append(var)
                elif var['type'] == 'categories':
                    cat_vars.append(var)
            for var in float_vars:
                toolbox.register(
                    f"attr_{var['name']}", 
                    np.random.uniform, 
                    float(var['lb']), 
                    float(var['ub'])
                )
            for var in int_vars:
                toolbox.register(
                    f"attr_{var['name']}", 
                    np.random.randint, 
                    int(var['lb']), 
                    int(var['ub']) + 1
                )
            for var in cat_vars:
                toolbox.register(
                    f"attr_{var['name']}", 
                    np.random.choice, 
                    var['categories']
                )
            attr_generators = []
            for var in variables:
                attr_generators.append(getattr(toolbox, f"attr_{var['name']}"))
            toolbox.register("individual", tools.initCycle, 
                            creator.Individual, attr_generators, n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            # --- USE UNIFIED CALCULATION IN evaluate() ---
            def evaluate(individual):
                var_dict = {}
                for var, val in zip(variables, individual):
                    if var['type'] == 'int':
                        int_val = round(val)
                        lb = int(var['lb'])
                        ub = int(var['ub'])
                        int_val = max(lb, min(ub, int_val))
                        var_dict[var['name']] = int_val
                    else:
                        var_dict[var['name']] = val
                intermediates_vals, objectives_vals_dict = calculate_all(
                    var_dict, st.session_state.intermediates, st.session_state.objectives
                )
                objectives_tuple = tuple(objectives_vals_dict.get(name, float('inf')) for name in st.session_state.objectives.keys())
                if len(objectives_tuple) != len(st.session_state.objectives):
                    objectives_tuple = [float('inf')] * len(st.session_state.objectives)
                return tuple(objectives_tuple)

            toolbox.register("evaluate", evaluate)

            def hybrid_mate(ind1, ind2):
                child1, child2 = toolbox.clone(ind1), toolbox.clone(ind2)
                for i, var in enumerate(variables):
                    if var['type'] in ['float', 'int']:
                        new_val1, new_val2 = tools.cxSimulatedBinaryBounded(
                            [child1[i]], [child2[i]], 
                            low=float(var['lb']),
                            up=float(var['ub']),
                            eta=20.0
                        )
                        child1[i] = new_val1[0]
                        child2[i] = new_val2[0]
                    elif var['type'] == 'categories':
                        if random.random() < 0.5:
                            child1[i], child2[i] = child2[i], child1[i]
                return child1, child2

            def hybrid_mutate(individual):
                mutant = toolbox.clone(individual)
                for i, var in enumerate(variables):
                    mut_prob = 0.2 if var['type'] == 'categories' else 0.1
                    if random.random() < mut_prob:
                        if var['type'] == 'float':
                            new_val, = tools.mutPolynomialBounded(
                                [mutant[i]], 
                                low=float(var['lb']),
                                up=float(var['ub']),
                                eta=20.0,
                                indpb=1.0
                            )
                            mutant[i] = new_val[0]
                        elif var['type'] == 'int':
                            lb, ub = int(var['lb']), int(var['ub'])
                            delta = max(1, int(0.1 * (ub - lb)))
                            new_val = mutant[i] + random.randint(-delta, delta)
                            mutant[i] = min(ub, max(lb, new_val))
                        elif var['type'] == 'categories':
                            choices = [c for c in var['categories'] if c != mutant[i]]
                            if choices:
                                mutant[i] = random.choice(choices)
                return mutant,

            toolbox.register("mate", hybrid_mate)
            toolbox.register("mutate", hybrid_mutate)
            toolbox.register("select", tools.selNSGA2)

            pop = toolbox.population(n=pop_size)
            hof = tools.ParetoFront()
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)
            pop, logbook = algorithms.eaMuPlusLambda(
                pop, toolbox, mu=pop_size, lambda_=pop_size,
                cxpb=cx_prob, mutpb=mut_prob, ngen=num_gen,
                stats=stats, halloffame=hof, verbose=True
            )
            st.session_state.results = {
                "pop": pop,
                "hof": hof,
                "logbook": logbook,
                "variables": variables
            }

if st.session_state.results:
    st.header("📈 Optimization Results")
    pareto_front = st.session_state.results["hof"]
    objectives_names = list(st.session_state.objectives.keys())
    num_objectives = len(objectives_names)

    # ==== USER INPUT: OBJECTIVE WEIGHTS ====
    st.sidebar.markdown("### 🎯 Set Objective Weights")
    weights = {}
    total_weight = 0
    for name in objectives_names:
        default_weight = 1.0 / len(objectives_names)
        weight = st.sidebar.number_input(
            f"Weight for '{name}'", min_value=0.0, max_value=1.0, value=default_weight, step=0.01, key=f"weight_{name}"
        )
        weights[name] = weight
        total_weight += weight
    # Normalize weights if sum > 0
    if total_weight > 0:
        for name in weights:
            weights[name] /= total_weight

    # ==== USER PLOTTING CHOICE ====
    axes = []
    plot_type = None
    if num_objectives == 1:
        st.info("Only one objective: displaying line plot.")
    elif num_objectives == 2:
        st.info("Two objectives: displaying 2D scatter plot.")
        axes = objectives_names[:2]
    elif num_objectives >= 3:
        plot_type = st.radio("Select plot type:", ["2D", "3D"])
        if plot_type == "2D":
            axes = st.multiselect("Select 2 objectives to plot", objectives_names, default=objectives_names[:2])
            if len(axes) != 2:
                st.warning("Please select exactly 2 objectives.")
        else:  # 3D
            axes = st.multiselect("Select 3 objectives to plot (3D)", objectives_names, default=objectives_names[:3])
            if len(axes) != 3:
                st.warning("Please select exactly 3 objectives.")

    # ==== INITIAL DESIGN POINT ====
    initial_inputs = {}
    for name, var in st.session_state.variables.items():
        input_key = f"input_{name}"
        if input_key in st.session_state:
            initial_inputs[name] = st.session_state[input_key]
        else:
            if var['type'] == 'float':
                initial_inputs[name] = (float(var['lb']) + float(var['ub'])) / 2
            elif var['type'] == 'int':
                initial_inputs[name] = (int(var['lb']) + int(var['ub'])) // 2
            elif var['type'] == 'categories':
                initial_inputs[name] = var['categories'][0]
    _, initial_objectives = calculate_all(
        initial_inputs, st.session_state.intermediates, st.session_state.objectives
    )

    # ==== GATHER ALL RESULTS (WITH SCORE) ====
    results_data = []
    for i, ind in enumerate(pareto_front):
        var_values = {}
        for var, val in zip(st.session_state.variables.values(), ind):
            if var['type'] == 'int':
                int_val = round(val)
                lb = int(var['lb'])
                ub = int(var['ub'])
                int_val = max(lb, min(ub, int_val))
                var_values[var['name']] = int_val
            else:
                var_values[var['name']] = val
        # --- UNIFIED CALCULATION ---
        intermediates_vals, objectives_vals = calculate_all(
            var_values, st.session_state.intermediates, st.session_state.objectives
        )
        results_data.append({
            **var_values,
            **intermediates_vals,
            **objectives_vals
        })

    # ==== NORMALIZATION AND SCORING ====
    # Find min/max for each objective
    obj_min = {name: min([res[name] for res in results_data]) for name in objectives_names}
    obj_max = {name: max([res[name] for res in results_data]) for name in objectives_names}
    # Score for each solution
    scores = []
    for res in results_data:
        score = 0
        for name in objectives_names:
            val = res[name]
            if obj_max[name] != obj_min[name]:
                if st.session_state.objectives[name]['target'].lower() == 'minimize':
                    norm = (obj_max[name] - val) / (obj_max[name] - obj_min[name])
                else:
                    norm = (val - obj_min[name]) / (obj_max[name] - obj_min[name])
            else:
                norm = 1
            score += weights[name] * norm
        scores.append(score)
    for res, score in zip(results_data, scores):
        res['Score'] = score
    # Index of best solution
    best_idx = int(np.argmax(scores))

    # ==== PARETO PLOT SECTION ====
    import matplotlib.pyplot as plt
    import numpy as np

    if num_objectives == 1:
        fig, ax = plt.subplots()
        y = [ind.fitness.values[0] for ind in pareto_front]
        ax.plot(range(len(y)), y, 'o', label="Pareto Front")
        ax.scatter([0], [initial_objectives.get(objectives_names[0])], 
                   c='blue', marker='o', s=100, label="Initial Design")
        # Highlight best solution
        ax.scatter([best_idx], [y[best_idx]], c='gold', marker='.', s=250, label="Best Solution")
        ax.set_ylabel(objectives_names[0])
        ax.set_xlabel("Solution Index")
        ax.set_title("Pareto Front (1D)")
        ax.legend()
        st.pyplot(fig)

    elif num_objectives == 2 or (num_objectives >= 3 and axes and len(axes) == 2):
        idx0 = objectives_names.index(axes[0])
        idx1 = objectives_names.index(axes[1])
        x = [ind.fitness.values[idx0] for ind in pareto_front]
        y = [ind.fitness.values[idx1] for ind in pareto_front]
        fig, ax = plt.subplots()
        ax.scatter(x, y, c='red', s=80, label="Pareto Front")
        initial_point = [initial_objectives.get(axes[0]), initial_objectives.get(axes[1])]
        ax.scatter(initial_point[0], initial_point[1],
                   c='blue', marker='o', s=120, label="Initial Design")
        # Highlight best solution
        ax.scatter([x[best_idx]], [y[best_idx]], c='gold', marker='.', s=250, label="Best Solution")
        ax.set_xlabel(axes[0])
        ax.set_ylabel(axes[1])
        ax.set_title("Pareto Front (2D)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    elif num_objectives >= 3 and axes and len(axes) == 3:
        import plotly.graph_objects as go
        idx0 = objectives_names.index(axes[0])
        idx1 = objectives_names.index(axes[1])
        idx2 = objectives_names.index(axes[2])
        x = [ind.fitness.values[idx0] for ind in pareto_front]
        y = [ind.fitness.values[idx1] for ind in pareto_front]
        z = [ind.fitness.values[idx2] for ind in pareto_front]
        initial_point = [initial_objectives.get(axes[0]), initial_objectives.get(axes[1]), initial_objectives.get(axes[2])]
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Pareto Front'
        ))
        # Initial design point
        fig.add_trace(go.Scatter3d(
            x=[initial_point[0]], y=[initial_point[1]], z=[initial_point[2]],
            mode='markers',
            marker=dict(size=10, color='blue', symbol='diamond'),
            name='Initial Design'
        ))
        # Best solution
        fig.add_trace(go.Scatter3d(
            x=[x[best_idx]], y=[y[best_idx]], z=[z[best_idx]],
            mode='markers',
            marker=dict(size=14, color='gold', symbol='circle'),
            name='Best Solution'
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title=axes[0],
                yaxis_title=axes[1],
                zaxis_title=axes[2]
            ),
            title="Pareto Front (3D Interactive)",
            legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ==== RESULTS TABLE ====
    st.subheader("Pareto Front Solutions")
    results_df = pd.DataFrame(results_data)
    for col in results_df.select_dtypes(include=['object']).columns:
        results_df[col] = results_df[col].astype(str)

    st.dataframe(results_df.style.apply(
        lambda row: ['background-color: gold' if row.name == best_idx else '' for _ in row], axis=1
    ))

    st.markdown(f"**🏅 Best Solution (highest score):**")
    st.write(results_df.iloc[best_idx])

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name='pareto_front_results.csv',
        mime='text/csv'
    )

# Configuration file management link

st.sidebar.markdown("---")
st.sidebar.markdown("### Configuration File Management")
st.sidebar.markdown("Please click config manager")
st.sidebar.markdown("---")
st.sidebar.markdown("**Developed by PS/PCB-AP**")